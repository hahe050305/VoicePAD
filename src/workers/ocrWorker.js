// src/workers/ocrWorker.js
// Dedicated Web Worker for Deep Learning TrOCR + Entity Normalization.
// Runs off the main browser thread to guarantee ZERO UI freezing.

import { pipeline, env } from "@huggingface/transformers";

// Use browser Cache API / IndexedDB
env.allowLocalModels = false;
env.useBrowserCache = true;

let ocrPipeline = null;

function normalizeRegistrationPlate(text) {
  if (!text) return null;
  const compact = text.toUpperCase().replace(/[^A-Z0-9]/g, "");
  // Indian & standard format: 2 State letters + 1-2 District digits + 0-3 Series letters + 3-4 Digits
  // e.g. BR01CJ6440, DL3CAF1234, MH12DE1433
  const plateMatch = compact.match(/([A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4})/);
  return plateMatch ? plateMatch[1] : null;
}

async function getPipeline() {
  if (ocrPipeline) return ocrPipeline;

  const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
  const device = hasWebGPU ? "webgpu" : "wasm";
  const dtype = hasWebGPU ? "fp32" : "q8";

  try {
    ocrPipeline = await pipeline("image-to-text", "Xenova/trocr-small-printed", {
      device,
      dtype,
    });
  } catch (err) {
    // Fallback to WASM quantized if WebGPU fails in worker context
    ocrPipeline = await pipeline("image-to-text", "Xenova/trocr-small-printed", {
      device: "wasm",
      dtype: "q8",
    });
  }

  return ocrPipeline;
}

self.onmessage = async (e) => {
  const { id, type, imageSource } = e.data;

  if (type === "PREWARM") {
    try {
      await getPipeline();
      self.postMessage({ id, type: "PREWARM_DONE", success: true });
    } catch (err) {
      self.postMessage({ id, type: "PREWARM_DONE", success: false, error: err.message });
    }
    return;
  }

  if (type === "RECOGNIZE") {
    try {
      const pipe = await getPipeline();
      const output = await pipe(imageSource);
      const rawText = (output?.[0]?.generated_text || "").trim();
      const plate = normalizeRegistrationPlate(rawText);

      self.postMessage({
        id,
        type: "SUCCESS",
        rawText,
        plate,
      });
    } catch (err) {
      self.postMessage({
        id,
        type: "ERROR",
        error: err.message || "OCR worker recognition failed",
      });
    }
  }
};
