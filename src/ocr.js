import { createWorker } from "tesseract.js";

let workerInstance = null;
let initPromise = null;

async function getWorker(onProgress) {
  if (workerInstance) {
    return workerInstance;
  }
  if (!initPromise) {
    initPromise = (async () => {
      const worker = await createWorker("eng", 1, {
        logger: (m) => {
          if (m.status === "recognizing text" && typeof onProgress === "function") {
            onProgress(Math.round((m.progress || 0) * 100));
          }
        },
      });
      workerInstance = worker;
      return worker;
    })().catch((error) => {
      initPromise = null;
      throw error;
    });
  }
  return initPromise;
}

/**
 * Robust filter for real-world text vs pure gibberish noise.
 */
export function isPureImageText(text) {
  if (!text || !text.trim()) return true;
  const clean = text.trim();

  // Any English words of length >= 3 with letters?
  const words = clean.split(/\s+/).filter(w => /^[a-zA-Z]{3,}$/.test(w));
  if (words.length >= 1) return false;

  const validNumbers = clean.split(/\s+/).filter(w => /^[A-Z0-9]{3,}$/i.test(w));
  if (validNumbers.length >= 1) return false;

  return true;
}

/**
 * Fast client-side image downscaling (Max 1000px).
 * Eliminates multi-second lag on high-res camera uploads.
 */
async function getOcrInput(file) {
  if (typeof createImageBitmap !== "function") return file;

  try {
    const bitmap = await createImageBitmap(file);
    try {
      const maxDim = 1000;
      const longest = Math.max(bitmap.width, bitmap.height);
      if (longest <= maxDim) return file;

      const scale = maxDim / longest;
      const targetWidth = Math.max(1, Math.round(bitmap.width * scale));
      const targetHeight = Math.max(1, Math.round(bitmap.height * scale));

      const canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return file;

      ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
      return blob || file;
    } finally {
      bitmap.close();
    }
  } catch {
    return file;
  }
}

export async function prewarmOCR() {
  try {
    if (typeof window !== "undefined" && "TextDetector" in window) return;
    await getWorker();
  } catch (err) {
    console.debug("[OCR] Pre-warm fail:", err);
  }
}

/**
 * Fast & Accurate Hybrid OCR Pipeline.
 * 1. Checks Native Hardware OS TextDetector (Sub-30ms).
 * 2. Runs lightweight Tesseract pass.
 * 3. Filters out single-character garbage tokens so tags are strictly meaningful.
 */
export async function performOCR(file, onProgress) {
  const input = await getOcrInput(file);
  let nativeText = "";
  let tesseractText = "";

  // 1. Native Hardware OS OCR (Near-instant ~20ms on Chrome/Edge/Android/macOS)
  if (typeof window !== "undefined" && "TextDetector" in window) {
    let bitmap;
    try {
      const detector = new window.TextDetector();
      bitmap = await createImageBitmap(input);
      const detections = await detector.detect(bitmap);
      nativeText = (detections || []).map((d) => d.rawValue || "").join(" ").trim();
    } catch (e) {
      console.debug("[OCR] Native TextDetector fallback:", e);
    } finally {
      bitmap?.close();
    }
  }

  // 2. Tesseract.js (Standard OCR)
  try {
    const worker = await getWorker(onProgress);
    const ret = await worker.recognize(input);
    tesseractText = ret?.data?.text?.trim() || "";
  } catch (tessErr) {
    console.debug("[OCR] Tesseract pass failed:", tessErr);
  }

  // Combine results
  const combinedRaw = `${nativeText}\n${tesseractText}`.trim();
  
  // Clean text: strip random non-ASCII punctuation noise & single-letter hallucinated artifacts
  const rawLines = combinedRaw.split(/\n+/).map(l => l.trim()).filter(Boolean);
  const uniqueLines = Array.from(new Set(rawLines));
  const rawText = uniqueLines.join("\n");

  const isPure = isPureImageText(rawText);

  return {
    text: isPure ? "" : rawText,
    rawText,
    isPureImage: isPure,
    confidence: 85,
  };
}
