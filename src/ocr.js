// src/ocr.js
import { Capacitor } from "@capacitor/core";
import { Filesystem, Directory } from "@capacitor/filesystem";
import { TextRecognition } from "@capacitor-mlkit/text-recognition";
import { parsePlateString } from "./services/plateService.js";
import { scoreOcrText } from "./services/ocrRanking.js";
import { createOcrVariants } from "./services/ocrPreprocessService.js";
import { recognizeWithTesseractEnhancements } from "./services/tesseractService.js";

export function isPureImageText(text) {
  if (!text || !text.trim()) return true;
  const clean = text.trim();
  if (extractVehiclePlate(clean) || extractStructuredIds(clean).length > 0) return false;

  const words = clean.split(/\s+/).filter((w) => /^[a-zA-Z]{3,}$/.test(w));
  if (words.length >= 1) return false;

  const validNumbers = clean.split(/\s+/).filter((w) => /^[A-Z0-9]{3,}$/i.test(w));
  if (validNumbers.length >= 1) return false;

  return true;
}

export function extractVehiclePlate(rawText) {
  return parsePlateString(rawText);
}

function hasPlateLikeText(text) {
  const tokens = String(text || "").toUpperCase().match(/[A-Z0-9]+/g) || [];
  for (let start = 0; start < tokens.length; start++) {
    let candidate = "";
    for (let count = 0; count < 5 && start + count < tokens.length; count++) {
      candidate += tokens[start + count];
      if (candidate.length > 12) break;
      if (
        candidate.length >= 6 &&
        (candidate.match(/[A-Z]/g) || []).length >= 2 &&
        (candidate.match(/\d/g) || []).length >= 3
      ) {
        return true;
      }
    }
  }
  return false;
}

export function extractStructuredIds(rawText) {
  if (!rawText) return [];
  const found = [];
  const upper = rawText.toUpperCase();
  const compact = upper.replace(/[^A-Z0-9]/g, "");

  const panMatch = compact.match(/([A-Z]{5}\d{4}[A-Z])/);
  if (panMatch) found.push({ type: "pan", value: panMatch[1], tag: "#document" });

  const aadhaarMatch = compact.match(/(\d{12})/);
  if (aadhaarMatch) found.push({ type: "aadhaar", value: aadhaarMatch[1], tag: "#document" });

  const dlMatch = compact.match(/([A-Z]{2}\d{13,15})/);
  if (dlMatch) found.push({ type: "dl", value: dlMatch[1], tag: "#document" });

  return found;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const dataUrl = reader.result;
      const base64 = typeof dataUrl === "string" ? dataUrl.split(",")[1] : "";
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * 1. Native On-Device OCR (Android / iOS):
 * Uses Google Play Services ML Kit via @capacitor-mlkit/text-recognition.
 * The native plugin handles recognition asynchronously and supports arbitrary angles and 2-line plates.
 */
async function recognizeWithNativeMLKit(file) {
  const extension = file.type === "image/png" ? "png" : "jpg";
  const tempName = `ocr_${Date.now()}_${Math.random().toString(36).slice(2, 7)}.${extension}`;
  try {
    const base64 = await blobToBase64(file);
    await Filesystem.writeFile({
      path: tempName,
      directory: Directory.Cache,
      data: base64,
    });

    const fileUri = await Filesystem.getUri({
      path: tempName,
      directory: Directory.Cache,
    });

    const result = await TextRecognition.processImage({
      path: fileUri.uri,
    });

    const fullText = result?.text?.trim() || "";
    return {
      text: fullText,
      confidence: 0,
    };
  } finally {
    await Filesystem.deleteFile({
      path: tempName,
      directory: Directory.Cache,
    }).catch((error) => {
      console.warn("[OCR] Temporary image cleanup failed:", error);
    });
  }
}

async function recognizeWithNativeEnhancements(file) {
  let processed = {
    baseline: null,
    variants: [],
    perspectiveVariants: [],
    focusedRegions: [],
    rotated: [],
  };
  try {
    processed = await createOcrVariants(file);
  } catch (error) {
    console.warn("[OCR] Native image enhancement unavailable; using original:", error);
  }

  let bestResult = null;
  let bestScore = -1;
  const candidates = [
    processed.baseline || file,
    ...processed.perspectiveVariants,
    ...processed.variants,
    ...processed.focusedRegions,
  ].slice(0, 5);
  for (const candidate of candidates) {
    try {
      const result = await recognizeWithNativeMLKit(candidate);
      const score = scoreOcrText(result.text);
      if (score > bestScore) {
        bestResult = result;
        bestScore = score;
      }
    } catch (error) {
      console.warn("[OCR] Native ML Kit pass failed; continuing with other variants:", error);
    }
  }
  if (bestResult) return bestResult;
  throw new Error("All native OCR passes failed.");
}

/**
 * 2. Web / Browser Dev Fallback:
 * Uses browser native Shape Detection API (window.TextDetector) or a Tesseract worker pass.
 */
async function recognizeWithWeb(file, onProgress) {
  let nativeText = "";
  if (typeof window !== "undefined" && "TextDetector" in window) {
    let bitmap;
    try {
      const detector = new window.TextDetector();
      bitmap = await createImageBitmap(file);
      const detections = await detector.detect(bitmap);
      nativeText = (detections || []).map((d) => d.rawValue || "").join("\n").trim();
    } catch (error) {
      console.warn("[OCR] Browser text detection failed; trying Tesseract:", error);
    } finally {
      bitmap?.close();
    }
  }

  // Tesseract compares the original with contrast and adaptive-threshold variants.
  const ret = await recognizeWithTesseractEnhancements(file, { plate: false, onProgress }).catch((error) => {
    console.warn("[OCR] Tesseract recognition failed:", error);
    return null;
  });
  const raw = ret?.data?.text?.trim() || "";
  const conf = Math.round(Number(ret?.data?.confidence) || 0);
  return { text: raw || nativeText, confidence: raw ? conf : 0 };
}

async function recognizePlateCandidate(file, text, onProgress) {
  if (extractVehiclePlate(text) || !hasPlateLikeText(text)) {
    return { text, attempted: false };
  }
  try {
    const result = await recognizeVehiclePlate(file, onProgress);
    return {
      text: result.plate ? `${text}\n${result.rawText}` : text,
      attempted: true,
    };
  } catch (error) {
    console.warn("[OCR] Plate-focused recognition failed; keeping primary OCR text:", error);
    return { text, attempted: true };
  }
}

export async function recognizeVehiclePlate(file, onProgress) {
  const result = await recognizeWithTesseractEnhancements(file, { plate: true, onProgress });
  const rawText = result?.data?.text?.trim() || "";
  return {
    rawText,
    plate: extractVehiclePlate(rawText),
    confidence: Math.round(Number(result?.data?.confidence) || 0),
  };
}

/**
 * High-performance, on-device OCR pipeline:
 * Automatically routes to native Google ML Kit on mobile or browser engine in development.
 */
let ocrQueue = Promise.resolve();

export async function performOCR(file, onProgress) {
  const task = ocrQueue.then(() => _performOCR(file, onProgress));
  ocrQueue = task.catch(() => {});
  return task;
}

// Internal OCR implementation separated from the exported queue wrapper.
async function _performOCR(file, onProgress) {
  let result;
  try {
    if (Capacitor.isNativePlatform()) {
      result = await recognizeWithNativeEnhancements(file);
    } else {
      result = await recognizeWithWeb(file, onProgress);
    }
  } catch (err) {
    console.warn("[OCR] Primary recognition error, falling back to Web OCR:", err);
    result = await recognizeWithWeb(file, onProgress).catch((fallbackError) => {
      console.warn("[OCR] Web OCR fallback failed:", fallbackError);
      return { text: "", confidence: 0 };
    });
  }

  const plateRecognition = await recognizePlateCandidate(file, result.text, onProgress);
  const recognizedText = plateRecognition.text;
  const confidence = result.confidence;
  const plate = extractVehiclePlate(recognizedText);
  const structuredIds = extractStructuredIds(recognizedText);

  const cleanDisplayText = recognizedText.trim();
  const displayText = plate && !cleanDisplayText.includes(plate)
    ? `Vehicle Plate: ${plate}\n${cleanDisplayText}`.trim()
    : cleanDisplayText;

 

  return {
    text: displayText,
    rawText: recognizedText,
    isPureImage: isPure,
    confidence,
    plate,
    plateOcrAttempted: plateRecognition.attempted,
    structuredIds,
  };
}
