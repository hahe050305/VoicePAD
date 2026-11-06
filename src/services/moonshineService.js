// src/services/moonshineService.js
import { ensureMoonshineModelCached } from "./modelCache.js";

// Client service for on-device Moonshine Speech-to-Text.
// Captures 16kHz mono PCM audio via Web Audio API and runs transcription in an offline Web Worker.

let moonshineStatus = { state: "idle", progress: null, error: null };
const statusListeners = new Set();

export function subscribeMoonshineStatus(callback) {
  statusListeners.add(callback);
  callback(moonshineStatus);
  return () => statusListeners.delete(callback);
}

function updateStatus(status) {
  moonshineStatus = { ...moonshineStatus, ...status };
  statusListeners.forEach((listener) => listener(moonshineStatus));
}

// ---------------------------------------------------------------------------
// Worker management
let worker = null;
let requestId = 0;
const pendingRequests = new Map();

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("../workers/moonshineWorker.js", import.meta.url), {
    type: "module",
  });

  worker.onmessage = ({ data }) => {
    if (data.type === "STATUS") {
      updateStatus({
        state: data.state,
        progress: data.progress ?? null,
        error: data.error ?? null,
      });
      return;
    }

    const request = pendingRequests.get(data.id);
    if (!request) return;
    pendingRequests.delete(data.id);

    if (data.error) {
      request.reject(new Error(data.error));
    } else {
      request.resolve(data.result);
    }
  };

  const handleError = (event) => {
    const error = new Error(event.message || "Moonshine worker crashed.");
    updateStatus({ state: "error", progress: null, error: error.message });
    for (const req of pendingRequests.values()) req.reject(error);
    pendingRequests.clear();
    worker?.terminate();
    worker = null;
  };

  worker.onerror = handleError;
  worker.onmessageerror = handleError;
  return worker;
}

export function prewarmMoonshine() {
  if (typeof window === "undefined") return;
  // Ensure model assets are cached locally before warm‑up
  ensureMoonshineModelCached()
    .catch((e) => console.warn("[Moonshine] Cache priming failed:", e))
    .finally(() => {
      setTimeout(() => {
        runTask("warmup").catch((err) => {
          console.warn("[Moonshine] Background model pre‑warming skipped:", err);
        });
      }, 2500);
    });
}

function runTask(type, payload) {
  const w = getWorker();
  const id = ++requestId;
  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { resolve, reject });
    if (type === "transcribe") {
      w.postMessage({ id, type, audioData: payload });
    } else {
      w.postMessage({ id, type });
    }
  });
}

// ---------------------------------------------------------------------------
// Audio Recording & 16kHz PCM Float32 Extraction
let activeAudioContext = null;
let activeMediaStream = null;
let activeProcessor = null;
let recordedAudioChunks = [];

/**
 * Starts recording audio from the user's microphone with hardware noise suppression,
 * echo cancellation, and auto-gain.
 */
export async function startAudioCapture() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Microphone capture is not supported in this browser.");
  }

  // 1. Hardware-level constraints for noise reduction
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: { ideal: true },
      noiseSuppression: { ideal: true },
      autoGainControl: { ideal: true },
      channelCount: 1,
      sampleRate: { ideal: 16000 },
    },
  });

  activeMediaStream = stream;
  recordedAudioChunks = [];

  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioCtx({ sampleRate: 16000 });
  activeAudioContext = ctx;

  const source = ctx.createMediaStreamSource(stream);

  // High-pass filter cuts fan vibration, desk rumble below 85Hz
  const highPass = ctx.createBiquadFilter();
  highPass.type = "highpass";
  highPass.frequency.value = 85;
  highPass.Q.value = 0.7;

  // Vocal boost filter around speech resonance
  const vocalBoost = ctx.createBiquadFilter();
  vocalBoost.type = "peaking";
  vocalBoost.frequency.value = 2000;
  vocalBoost.Q.value = 1.0;
  vocalBoost.gain.value = 3.0;

  // Buffer chunk collector (4096 samples per chunk = ~256ms intervals at 16kHz)
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  activeProcessor = processor;

  processor.onaudioprocess = (e) => {
    const inputData = e.inputBuffer.getChannelData(0);
    // Clone buffer
    recordedAudioChunks.push(new Float32Array(inputData));
  };

  source.connect(highPass);
  highPass.connect(vocalBoost);
  vocalBoost.connect(processor);
  processor.connect(ctx.destination);

  return stream;
}

/**
 * Stops recording, merges captured PCM frames into a single Float32Array,
 * and transcribes using the offline Moonshine Web Worker.
 */
export async function stopAudioCaptureAndTranscribe() {
  if (activeProcessor) {
    try {
      activeProcessor.disconnect();
    } catch {}
    activeProcessor = null;
  }

  if (activeMediaStream) {
    try {
      activeMediaStream.getTracks().forEach((track) => track.stop());
    } catch {}
    activeMediaStream = null;
  }

  if (activeAudioContext) {
    try {
      if (activeAudioContext.state !== "closed") {
        await activeAudioContext.close();
      }
    } catch {}
    activeAudioContext = null;
  }

  if (recordedAudioChunks.length === 0) {
    return "";
  }

  // Flatten collected chunks into a contiguous Float32Array
  const totalLength = recordedAudioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const audioData = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of recordedAudioChunks) {
    audioData.set(chunk, offset);
    offset += chunk.length;
  }
  recordedAudioChunks = [];

  // Minimum 0.4 seconds of speech to avoid transcribing micro-clicks
  if (audioData.length < 16000 * 0.4) {
    return "";
  }

  const result = await runTask("transcribe", audioData);
  return result?.text || "";
}

export function cancelAudioCapture() {
  if (activeProcessor) {
    try { activeProcessor.disconnect(); } catch {}
    activeProcessor = null;
  }
  if (activeMediaStream) {
    try { activeMediaStream.getTracks().forEach((track) => track.stop()); } catch {}
    activeMediaStream = null;
  }
  if (activeAudioContext) {
    try {
      if (activeAudioContext.state !== "closed") activeAudioContext.close();
    } catch {}
    activeAudioContext = null;
  }
  recordedAudioChunks = [];
}
