// src/services/moonshineService.js
import { Capacitor } from "@capacitor/core";

// Client service for on-device Moonshine Speech-to-Text.
// Captures 16kHz mono PCM audio via Web Audio API and runs transcription in an offline Web Worker.

let moonshineStatus = { state: "idle", progress: null, error: null };
const statusListeners = new Set();

export function isMoonshineReady() {
  return moonshineStatus.state === "ready";
}

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

let moonshineWarmupPromise = null;

export function prewarmMoonshine(delay = 0) {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (!moonshineWarmupPromise) {
    updateStatus({ state: "loading", progress: 0, error: null });
    moonshineWarmupPromise = new Promise((resolve) => setTimeout(resolve, delay))
      .then(() => runTask("warmup"))
      .catch((error) => {
        moonshineWarmupPromise = null;
        console.warn("[Moonshine] Background model pre-warming skipped:", error);
        return null;
      });
  }
  return moonshineWarmupPromise;
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
let liveSnapshotInFlight = false;

export function resampleTo16k(audioBuffer, sourceSampleRate) {
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

function mergeChunks(chunks) {
  const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const audioData = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    audioData.set(chunk, offset);
    offset += chunk.length;
  }
  return audioData;
}

/**
 * Removes DC offset, trims leading/trailing silence, and gently normalizes
 * quiet microphone speech for Moonshine's raw-waveform frontend.
 * Returns null if the buffer contains only silence or is too short.
 */
export function preparePcmForMoonshine(pcm16k, minSamples = 16000 * 0.2) {
  if (!pcm16k || pcm16k.length < minSamples) return null;

  // 1. Remove DC offset and find peak amplitude
  let sum = 0;
  for (let i = 0; i < pcm16k.length; i++) {
    sum += pcm16k[i];
  }
  const mean = sum / pcm16k.length;

  const centered = new Float32Array(pcm16k.length);
  let peak = 0;
  for (let i = 0; i < pcm16k.length; i++) {
    const v = pcm16k[i] - mean;
    centered[i] = v;
    const abs = Math.abs(v);
    if (abs > peak) peak = abs;
  }

  // Pure silence / dead mic
  if (peak < 0.002) return null;

  // 2. Trim only true dead silence with generous 250ms padding so soft initial
  // consonants (e.g. "b", "p", "k") are never clipped.
  const threshold = Math.min(0.008, peak * 0.015);
  const padSamples = 4000; // 250ms at 16kHz
  let start = 0;
  while (start < centered.length && Math.abs(centered[start]) < threshold) {
    start++;
  }
  let end = centered.length - 1;
  while (end > start && Math.abs(centered[end]) < threshold) {
    end--;
  }

  if (end <= start) return null;

  const trimStart = Math.max(0, start - padSamples);
  const trimEnd = Math.min(centered.length, end + 1 + padSamples);
  const trimmed = centered.subarray(trimStart, trimEnd);

  if (trimmed.length < minSamples) return null;

  // 3. Gently normalize quiet speech without amplifying background hiss
  const gain = peak < 0.35 ? Math.min(8, 0.5 / peak) : 1;
  if (gain === 1) {
    return new Float32Array(trimmed);
  }

  const normalized = new Float32Array(trimmed.length);
  for (let i = 0; i < trimmed.length; i++) {
    const scaled = trimmed[i] * gain;
    normalized[i] = scaled > 1 ? 1 : scaled < -1 ? -1 : scaled;
  }
  return normalized;
}

/**
 * Starts recording audio from the user's microphone with hardware noise suppression,
 * echo cancellation, and auto-gain.
 */
export async function startAudioCapture() {
  // Create AudioContext synchronously during the user gesture before awaiting permissions/mic
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  let ctx = null;
  if (AudioCtx) {
    try {
      ctx = new AudioCtx({ sampleRate: 16000 });
    } catch {
      ctx = new AudioCtx();
    }
  }

  try {
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

    if (!ctx) {
      ctx = new AudioCtx();
    }
    if (ctx.state === "suspended") {
      await ctx.resume();
    }

    activeAudioContext = ctx;
    capturedSampleRate = ctx.sampleRate || 16000;

    const source = ctx.createMediaStreamSource(stream);

    // Gentle high-pass filter cuts sub-60Hz fan vibration / desk rumble without altering speech formants
    const highPass = ctx.createBiquadFilter();
    highPass.type = "highpass";
    highPass.frequency.value = 60;
    highPass.Q.value = 0.7;

    // Buffer chunk collector (4096 samples per chunk = ~256ms intervals at 16kHz)
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    activeProcessor = processor;

    processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      // Clone buffer
      recordedAudioChunks.push(new Float32Array(inputData));
    };

    source.connect(highPass);
    highPass.connect(processor);
    // Keep ScriptProcessor callbacks alive without playing the microphone back.
    const muteGain = ctx.createGain();
    muteGain.gain.value = 0;
    processor.connect(muteGain);
    muteGain.connect(ctx.destination);
    activeMuteGain = muteGain;

    return stream;
  } catch (err) {
    if (ctx && ctx.state !== "closed") {
      try { await ctx.close(); } catch { /* ignore */ }
    }
    throw err;
  }
}

/**
 * Non-destructive live transcription of the audio recorded so far.
 * Used while the mic is active so filtering works even when Web Speech API is unavailable.
 */
export async function transcribeCurrentAudioSnapshot() {
  if (liveSnapshotInFlight || recordedAudioChunks.length === 0) {
    return "";
  }
  const chunksSnapshot = recordedAudioChunks.slice();
  const sampleRate = capturedSampleRate || activeAudioContext?.sampleRate || 16000;
  const audioData = mergeChunks(chunksSnapshot);
  const pcm16k = resampleTo16k(audioData, sampleRate);
  const prepared = preparePcmForMoonshine(pcm16k, 16000 * 0.3);
  if (!prepared) return "";

  liveSnapshotInFlight = true;
  try {
    const result = await runTask("transcribe", prepared);
    return result?.text || "";
  } catch {
    return "";
  } finally {
    liveSnapshotInFlight = false;
  }
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
  const audioData = mergeChunks(recordedAudioChunks);
  recordedAudioChunks = [];

  // Resample from hardware rate to 16kHz for Moonshine
  const pcm16k = resampleTo16k(audioData, sampleRate);
  const prepared = preparePcmForMoonshine(pcm16k, 16000 * 0.2);
  if (!prepared) {
    return "";
  }

  const result = await runTask("transcribe", prepared);
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
