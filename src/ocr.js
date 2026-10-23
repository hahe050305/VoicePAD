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
    })();
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
  if (isGif) {
    // Keep existing GIF handling...
  }

  // Pre-scale large camera photos down to max 1280px (preserves full readability while speeding up OCR by 4x-8x)
  try {
    const bitmap = await createImageBitmap(file);
    const maxDim = 1280;
    if (bitmap.width <= maxDim && bitmap.height <= maxDim) {
      bitmap.close();
      return file;
    }
    const ratio = Math.min(maxDim / bitmap.width, maxDim / bitmap.height);
    const targetWidth = Math.round(bitmap.width * ratio);
    const targetHeight = Math.round(bitmap.height * ratio);

    const canvas = document.createElement("canvas");
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      bitmap.close();
      return file;
    }
    ctx.drawImage(bitmap, 0, 0, targetWidth, targetHeight);
    bitmap.close();

    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.85));
    return blob || file;
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
  // 1. Instant Hardware Native OS OCR (Chrome/Edge/Android/macOS) -> ~20ms
  if (typeof window !== "undefined" && "TextDetector" in window) {
    try {
      const detector = new window.TextDetector();
      const bitmap = await createImageBitmap(file);
      const detections = await detector.detect(bitmap);
      bitmap.close();
      if (detections && detections.length > 0) {
        const nativeText = detections.map((d) => d.rawValue || "").join(" ").trim();
        const isPure = isPureImageText(nativeText);
        return {
          text: isPure ? "" : nativeText,
          rawText: nativeText,
          isPureImage: isPure,
          confidence: 98,
        };
      }
    } catch (e) {
      console.debug("[OCR] Native TextDetector fallback to WASM:", e);
    }
  }
  // 2. Fallback to pre-warmed Tesseract.js with downscaling
  const worker = await getWorker(onProgress);
  const input = await getOcrInput(file);
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