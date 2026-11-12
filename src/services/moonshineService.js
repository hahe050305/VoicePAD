// src/services/moonshineService.js
import { Capacitor } from "@capacitor/core";
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

export function prewarmMoonshine(delay = 1800) {
  if (typeof window === "undefined") return;
  // Ensure model assets are cached locally before warm‑up
  ensureMoonshineModelCached()
    .catch((e) => console.warn("[Moonshine] Cache priming failed:", e))
    .finally(() => {
      setTimeout(() => {
        runTask("warmup").catch((err) => {
          console.warn("[Moonshine] Background model pre‑warming skipped:", err);
        });
      }, delay);
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
let activeMuteGain = null;
let recordedAudioChunks = [];
let capturedSampleRate = 16000;

function resampleTo16k(audioBuffer, sourceSampleRate) {
  if (!sourceSampleRate || sourceSampleRate === 16000) return audioBuffer;
  const ratio = sourceSampleRate / 16000;
  const newLength = Math.max(1, Math.round(audioBuffer.length / ratio));
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const originIdx = i * ratio;
    const idx0 = Math.floor(originIdx);
    const idx1 = Math.min(idx0 + 1, audioBuffer.length - 1);
    const weight = originIdx - idx0;
    result[i] = audioBuffer[idx0] * (1 - weight) + audioBuffer[idx1] * weight;
  }
  return result;
}

/**
 * Starts recording audio from the user's microphone with hardware noise suppression,
 * echo cancellation, and auto-gain.
 */
export async function startAudioCapture() {
  if (Capacitor.isNativePlatform()) {
    try {
      const { VoiceRecorder } = await import("capacitor-voice-recorder");
      const perm = await VoiceRecorder.hasAudioRecordingPermission();
      if (!perm?.value) {
        await VoiceRecorder.requestAudioRecordingPermission();
      }
    } catch (e) {
      console.warn("[Moonshine] Native mic permission check:", e);
    }
  }

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
  let ctx;
  try {
    ctx = new AudioCtx({ sampleRate: 16000 });
  } catch {
    ctx = new AudioCtx();
  }

  if (ctx.state === "suspended") {
    await ctx.resume();
  }

  activeAudioContext = ctx;
  capturedSampleRate = ctx.sampleRate || 16000;

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
  // Keep ScriptProcessor callbacks alive without playing the microphone back.
  const muteGain = ctx.createGain();
  muteGain.gain.value = 0;
  processor.connect(muteGain);
  muteGain.connect(ctx.destination);
  activeMuteGain = muteGain;

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
    } catch { /* The processor may already be disconnected. */ }
    activeProcessor = null;
  }
  if (activeMuteGain) {
    try { activeMuteGain.disconnect(); } catch { /* The output node may already be disconnected. */ }
    activeMuteGain = null;
  }

  if (activeMediaStream) {
    try {
      activeMediaStream.getTracks().forEach((track) => track.stop());
    } catch { /* Tracks can already be stopped by the browser. */ }
    activeMediaStream = null;
  }

  const sampleRate = capturedSampleRate || activeAudioContext?.sampleRate || 16000;

  if (activeAudioContext) {
    try {
      if (activeAudioContext.state !== "closed") {
        await activeAudioContext.close();
      }
    } catch { /* The audio context may already be closed. */ }
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

  // Resample from hardware rate to 16kHz for Moonshine
  const pcm16k = resampleTo16k(audioData, sampleRate);

  // Minimum 0.3 seconds of speech to avoid transcribing micro-clicks
  if (pcm16k.length < 16000 * 0.3) {
    return "";
  }

  const result = await runTask("transcribe", pcm16k);
  return result?.text || "";
}

export function cancelAudioCapture() {
  if (activeProcessor) {
    try { activeProcessor.disconnect(); } catch { /* The processor may already be disconnected. */ }
    activeProcessor = null;
  }
  if (activeMuteGain) {
    try { activeMuteGain.disconnect(); } catch { /* The output node may already be disconnected. */ }
    activeMuteGain = null;
  }
  if (activeMediaStream) {
    try { activeMediaStream.getTracks().forEach((track) => track.stop()); } catch { /* Tracks can already be stopped. */ }
    activeMediaStream = null;
  }
  if (activeAudioContext) {
    try {
      if (activeAudioContext.state !== "closed") activeAudioContext.close();
    } catch { /* The audio context may already be closed. */ }
    activeAudioContext = null;
  }
  recordedAudioChunks = [];
}
