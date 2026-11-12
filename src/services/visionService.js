// src/services/visionService.js
// Provides client interface to the CLIP vector embedding worker and thumbnail generation.

let visionModelStatus = { state: "idle", progress: null, error: null };
const visionStatusListeners = new Set();

export function subscribeVisionModelStatus(callback) {
  visionStatusListeners.add(callback);
  callback(visionModelStatus);
  return () => visionStatusListeners.delete(callback);
}

export function isVisionModelReady() {
  return visionModelStatus.state === "ready";
}

function updateVisionModelStatus(status) {
  visionModelStatus = { ...visionModelStatus, ...status };
  visionStatusListeners.forEach((listener) => listener(visionModelStatus));
}

/** Prewarm the CLIP models in the background */
let visionWarmupPromise = null;

export function prewarmVisionModel(delay = 0) {
  if (typeof window === "undefined") return;
  if (!visionWarmupPromise) {
    visionWarmupPromise = new Promise((resolve) => setTimeout(resolve, delay))
      .then(() => runVisionTask("warmup", null))
      .catch((err) => {
        visionWarmupPromise = null;
        console.warn("[Vision] CLIP Model pre-warming skipped:", err);
        return null;
      });
  }
  return visionWarmupPromise;
}

/** Extract 512-dimensional normalized vector embedding from an image */
export function embedImage(imageSource) {
  return runVisionTask("embedImage", imageSource);
}

/**
 * Extract 512-dimensional normalized vector embedding from a text search query.
 * Non-blocking: If model is still downloading or not ready, returns null immediately
 * so search filtering is never stalled.
 */
export function embedText(text, { timeout = 250 } = {}) {
  if (visionModelStatus.state !== "ready") {
    return Promise.resolve(null);
  }
  const task = runVisionTask("embedText", text);
  const timeoutPromise = new Promise((resolve) => setTimeout(() => resolve(null), timeout));
  return Promise.race([task, timeoutPromise]).catch(() => null);
}

/** Backward compatibility placeholders */
export function getDetectedLabelsAndTags() {
  return Promise.resolve({ labels: [], tags: [], detections: [] });
}
export function detectPlateRegion() {
  return Promise.resolve(null);
}

export async function prepareImageForStorage(imageSource, { includeThumbnail = true } = {}) {
  try {
    const blob = imageSource instanceof Blob ? imageSource : await (await fetch(imageSource)).blob();
    const bitmap = await createImageBitmap(blob);
    try {
      const origW = bitmap.width;
      const origH = bitmap.height;
      const maxDim = 1440; // High clarity for phone zoom & OCR, but eliminates 20MB bloat
      const scale = Math.min(1, maxDim / Math.max(origW, origH));
      const width = Math.max(1, Math.round(origW * scale));
      const height = Math.max(1, Math.round(origH * scale));

      let optimizedBlob = blob;
      // Only scale down if larger than maxDim or if blob is heavy (> 600KB)
      if (scale < 1 || blob.size > 600 * 1024) {
        if (typeof OffscreenCanvas === "function") {
          const canvas = new OffscreenCanvas(width, height);
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(bitmap, 0, 0, width, height);
            optimizedBlob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.82 });
          }
        } else if (typeof document !== "undefined") {
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(bitmap, 0, 0, width, height);
            optimizedBlob = await new Promise((resolve) => {
              canvas.toBlob((b) => resolve(b || blob), "image/jpeg", 0.82);
            });
          }
        }
      }

      // Generate compact thumbnail for card preview in feed (max 400px, quality 0.75)
      let thumbnailBlob = null;
      if (includeThumbnail) {
        const thumbMax = 400;
        const thumbScale = Math.min(1, thumbMax / Math.max(origW, origH));
        const thumbW = Math.max(1, Math.round(origW * thumbScale));
        const thumbH = Math.max(1, Math.round(origH * thumbScale));

        if (typeof OffscreenCanvas === "function") {
          const thumbCanvas = new OffscreenCanvas(thumbW, thumbH);
          const thumbCtx = thumbCanvas.getContext("2d");
          if (thumbCtx) {
            thumbCtx.drawImage(bitmap, 0, 0, thumbW, thumbH);
            thumbnailBlob = await thumbCanvas.convertToBlob({ type: "image/jpeg", quality: 0.75 });
          }
        } else if (typeof document !== "undefined") {
          const thumbCanvas = document.createElement("canvas");
          thumbCanvas.width = thumbW;
          thumbCanvas.height = thumbH;
          const thumbCtx = thumbCanvas.getContext("2d");
          if (thumbCtx) {
            thumbCtx.drawImage(bitmap, 0, 0, thumbW, thumbH);
            thumbnailBlob = await new Promise((resolve) => {
              thumbCanvas.toBlob((b) => resolve(b), "image/jpeg", 0.75);
            });
          }
        }
      }

      return {
        optimizedBlob: optimizedBlob || blob,
        thumbnailBlob: thumbnailBlob || optimizedBlob || blob,
      };
    } finally {
      bitmap.close();
    }
  } catch (err) {
    console.warn("[Vision] Image optimization fallback:", err);
    const rawBlob = imageSource instanceof Blob ? imageSource : await (await fetch(imageSource)).blob();
    return { optimizedBlob: rawBlob, thumbnailBlob: rawBlob };
  }
}

export async function createImageThumbnail(imageSource) {
  // Fast, non-blocking client-side thumbnail generation
  try {
    const blob = imageSource instanceof Blob ? imageSource : await (await fetch(imageSource)).blob();
    const bitmap = await createImageBitmap(blob);
    try {
      const maxDim = 480;
      const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
      const width = Math.max(1, Math.round(bitmap.width * scale));
      const height = Math.max(1, Math.round(bitmap.height * scale));

      if (typeof OffscreenCanvas === "function") {
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(bitmap, 0, 0, width, height);
          return await canvas.convertToBlob({ type: "image/jpeg", quality: 0.80 });
        }
      }

      if (typeof document !== "undefined") {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(bitmap, 0, 0, width, height);
          return await new Promise((resolve) => {
            canvas.toBlob((b) => resolve(b || blob), "image/jpeg", 0.80);
          });
        }
      }
      return blob;
    } finally {
      bitmap.close();
    }
  } catch {
    return runVisionTask("thumbnail", imageSource).catch(() => null);
  }
}

// ---------------------------------------------------------------------------
// Worker management
let visionWorker = null;
let visionIdleTimer = null;
let visionRequestId = 0;
let visionQueue = Promise.resolve();
const visionRequests = new Map();

function runVisionTask(type, payload) {
  const task = visionQueue.then(async () => {
    if (visionIdleTimer) clearTimeout(visionIdleTimer);
    visionIdleTimer = null;
    const worker = getVisionWorker();
    const id = ++visionRequestId;
    const result = new Promise((resolve, reject) => {
      visionRequests.set(id, { resolve, reject });
      if (type === "embedImage" || type === "thumbnail") {
        worker.postMessage({ id, type, imageSource: payload });
      } else if (type === "embedText") {
        worker.postMessage({ id, type, text: payload });
      } else {
        worker.postMessage({ id, type });
      }
    });
    try {
      return await result;
    } finally {
      visionRequests.delete(id);
    }
  });
  visionQueue = task.catch(() => {});
  return task;
}

function getVisionWorker() {
  if (visionWorker) return visionWorker;
  visionWorker = new Worker(new URL("../workers/visionWorker.js", import.meta.url), {
    type: "module",
  });
  visionWorker.onmessage = ({ data }) => {
    if (data.type === "MODEL_STATUS") {
      updateVisionModelStatus({
        state: data.state,
        progress: data.progress ?? null,
        error: data.error ?? null,
      });
      return;
    }
    const request = visionRequests.get(data.id);
    if (!request) return;
    visionRequests.delete(data.id);
    if (data.error) request.reject(new Error(data.error));
    else request.resolve(data.result);
  };
  const rejectPending = (event) => {
    const error = new Error(event.message || "Vision worker failed.");
    updateVisionModelStatus({ state: "error", progress: null, error: error.message });
    for (const request of visionRequests.values()) request.reject(error);
    visionRequests.clear();
    visionWorker?.terminate();
    visionWorker = null;
  };
  visionWorker.onerror = rejectPending;
  visionWorker.onmessageerror = rejectPending;
  return visionWorker;
}
