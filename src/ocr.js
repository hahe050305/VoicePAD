import { createWorker } from "tesseract.js";
import { getDeepLearningOCR, normalizeRegistrationPlate } from "./services/deepOcrService.js";

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

  // If a valid registration plate pattern exists, it's definitely NOT pure noise
  if (normalizeRegistrationPlate(clean)) {
    return false;
  }

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
  if (lines.length >= 5 && avgLineLen < 5) return true;
  if (symbolRatio > 0.10) return true;
  if (lines.length >= 4 && words.length >= 4 && validWordRatio < 0.3) return true;

  return false;
}

async function getOcrInput(file) {
  const isGif = file?.type === "image/gif" || /\.gif$/i.test(file?.name || "");
  if (typeof createImageBitmap !== "function") return file;

  try {
    const bitmap = await createImageBitmap(file);
    try {
      const maxDim = 1600;
      const minDim = 900;
      const longest = Math.max(bitmap.width, bitmap.height);
      const ratio = Math.min(maxDim / longest, Math.max(1, minDim / longest));
      const targetWidth = Math.max(1, Math.round(bitmap.width * ratio));
      const targetHeight = Math.max(1, Math.round(bitmap.height * ratio));
      if (!isGif && ratio === 1) return file;
      const canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return file;

      ctx.filter = "grayscale(1) contrast(1.2)";
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
 * Runs OCR using our Hybrid Deep-Learning + Native + Tesseract Pipeline.
 * 1. Checks Native OS TextDetector (Hardware GPU/NPU).
 * 2. Runs Deep-Learning Vision Transformer (TrOCR) on the image.
 * 3. Falls back to Tesseract.js.
 * 4. Normalizes structured entities like vehicle plates (e.g. BR01C J6440).
 * 
 * @param {Blob | File} file
 * @param {(progressPercent: number) => void} [onProgress]
 * @returns {Promise<{text: string, rawText: string, isPureImage: boolean, confidence: number, plate: string|null}>}
 */
export async function performOCR(file, onProgress) {
  const input = await getOcrInput(file);
  let nativeText = "";
  let trocrText = "";
  let tesseractText = "";

  // 1. Native Hardware OS OCR (instant ~20ms on Chrome/Edge/macOS/Android)
  if (typeof window !== "undefined" && "TextDetector" in window) {
    let bitmap;
    try {
      const detector = new window.TextDetector();
      bitmap = await createImageBitmap(input);
      const detections = await detector.detect(bitmap);
      nativeText = (detections || []).map((d) => d.rawValue || "").join("\n").trim();
    } catch (e) {
      console.debug("[OCR] Native TextDetector fallback:", e);
    } finally {
      bitmap?.close();
    }
  }

  // 2. Deep-Learning Vision Transformer (TrOCR) via Transformers.js
  try {
    const pipe = await getDeepLearningOCR(onProgress);
    let imageUrl = null;
    if (file instanceof Blob || file instanceof File) {
      imageUrl = URL.createObjectURL(file);
    }
    const trocrOutput = await pipe(imageUrl || file);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    trocrText = (trocrOutput?.[0]?.generated_text || "").trim();
  } catch (deepErr) {
    console.debug("[OCR] Deep learning TrOCR skipped or failed, using standard engine:", deepErr);
  }

  // 3. Classical OCR fallback / supplement (Tesseract.js)
  try {
    const worker = await getWorker(onProgress);
    const ret = await worker.recognize(input);
    tesseractText = ret?.data?.text?.trim() || "";
  } catch (tessErr) {
    console.debug("[OCR] Tesseract pass failed:", tessErr);
  }

  // Combine and deduplicate extracted lines across all deep-learning & classical engines
  const allCandidateLines = [
    ...trocrText.split(/\n+/),
    ...nativeText.split(/\n+/),
    ...tesseractText.split(/\n+/),
  ]
    .map((l) => l.trim())
    .filter(Boolean);

  const uniqueLines = allCandidateLines.filter(
    (line, idx, arr) => arr.findIndex((c) => c.toLowerCase() === line.toLowerCase()) === idx
  );

  let rawText = uniqueLines.join("\n");

  // 4. Structured Entity Normalization (Plate / Card / Code Detection)
  const plate = normalizeRegistrationPlate(rawText);
  if (plate && !rawText.toUpperCase().includes(plate)) {
    rawText = `${plate}\n${rawText}`;
  }

  const isPure = isPureImageText(rawText);

  return {
    text: isPure ? "" : rawText,
    rawText,
    isPureImage: isPure,
    confidence: plate ? 99 : 85,
    plate,
  };
}
