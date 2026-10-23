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
      // Allow a later OCR attempt to retry if worker startup fails.
      initPromise = null;
      throw error;
    });
  }
  return initPromise;
}

/**
 * Detects whether OCR output is pure noise/gibberish produced from a photograph
 * or non-text image (e.g. portraits, objects, scenes).
 * @param {string} text
 * @returns {boolean}
 */
export function isPureImageText(text) {
  if (!text || !text.trim()) return true;
  const clean = text.trim();

  // If there are no alphanumeric characters at all, it's a pure image
  const letters = clean.match(/[a-zA-Z0-9]/g) || [];
  if (letters.length === 0) return true;

  // Split lines
  const lines = clean.split("\n").map((l) => l.trim()).filter(Boolean);
  const totalLineLen = lines.reduce((acc, l) => acc + l.length, 0);
  const avgLineLen = lines.length > 0 ? totalLineLen / lines.length : 0;

  // Weird symbol ratio common in hallucinated OCR on photos: §, |, ©, {, }, <, >, %, etc.
  const weirdSymbols = clean.match(/[^a-zA-Z0-9\s.,?!'"-]/g) || [];
  const symbolRatio = weirdSymbols.length / clean.length;

  // Word token quality
  const words = clean.split(/\s+/).filter(Boolean);
  const validWords = words.filter((w) => /^[a-zA-Z0-9]{3,}$/.test(w));
  const validWordRatio = words.length > 0 ? validWords.length / words.length : 0;

  // Hallucinated noise heuristics on photos:
  // 1. Many fragmented vertical lines (>= 5 lines with avg length < 5 chars per line)
  if (lines.length >= 5 && avgLineLen < 5) return true;

  // 2. High ratio of symbols/punctuation noise (> 10%)
  if (symbolRatio > 0.10) return true;

  // 3. Multi-line noise with low valid word ratio (e.g. random 1-2 char fragments)
  if (lines.length >= 4 && words.length >= 4 && validWordRatio < 0.3) return true;

  // Otherwise, it has readable text (like "Hash IHR", "Harish JHR", notes, signs)
  return false;
}

async function getOcrInput(file) {
  const isGif = file?.type === "image/gif" || /\.gif$/i.test(file?.name || "");
  if (typeof createImageBitmap !== "function") return file;

  try {
    const bitmap = await createImageBitmap(file);
    const maxDim = 1280;
    if (!isGif && bitmap.width <= maxDim && bitmap.height <= maxDim) {
      bitmap.close();
      return file;
    }
    try {
      // Explicitly rasterize GIFs to a still image; Tesseract reads only one frame.
      const ratio = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
      const targetWidth = Math.max(1, Math.round(bitmap.width * ratio));
      const targetHeight = Math.max(1, Math.round(bitmap.height * ratio));
      const canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return file;

      ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
      const mimeType = isGif ? "image/png" : "image/jpeg";
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, mimeType, 0.85));
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
    console.log("[OCR] Pre-warmed background worker.");
  } catch (err) {
    console.debug("[OCR] Pre-warm idle fail:", err);
  }
}

/**
 * Runs OCR on an image file or blob.
 * @param {Blob | File} file
 * @param {(progressPercent: number) => void} [onProgress]
 * @returns {Promise<{text: string, rawText: string, isPureImage: boolean, confidence: number}>}
 */
export async function performOCR(file, onProgress) {
  const input = await getOcrInput(file);
  // 1. Instant Hardware Native OS OCR (Chrome/Edge/Android/macOS) -> ~20ms
  if (typeof window !== "undefined" && "TextDetector" in window) {
    let bitmap;
    try {
      const detector = new window.TextDetector();
      bitmap = await createImageBitmap(input);
      const detections = await detector.detect(bitmap);
      const nativeText = (detections || []).map((d) => d.rawValue || "").join(" ").trim();
      if (nativeText) {
        const isPure = isPureImageText(nativeText);
        if (!isPure) {
          return {
            text: nativeText,
            rawText: nativeText,
            isPureImage: false,
            confidence: 98,
          };
        }
      }
    } catch (e) {
      console.debug("[OCR] Native TextDetector fallback to WASM:", e);
    } finally {
      bitmap?.close();
    }
  }
  // 2. Fallback to pre-warmed Tesseract.js with downscaling
  const worker = await getWorker(onProgress);
  const ret = await worker.recognize(input);
  const rawText = ret?.data?.text?.trim() || "";
  const confidence = ret?.data?.confidence || 0;
  const isPure = isPureImageText(rawText);
  return {
    text: isPure ? "" : rawText,
    rawText,
    isPureImage: isPure,
    confidence,
  };
}
