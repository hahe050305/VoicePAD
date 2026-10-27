// src/ocr.js
import { Capacitor } from "@capacitor/core";
import { Filesystem, Directory } from "@capacitor/filesystem";
import { TextRecognition } from "@capacitor-mlkit/text-recognition";
import { parsePlateString } from "./services/plateService.js";
import { recognizeWithTesseract } from "./services/tesseractService.js";

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
 * Hardware accelerated, zero UI freeze, handles arbitrary angles and 2-line plates.
 */
async function recognizeWithNativeMLKit(file) {
  const tempName = `ocr_${Date.now()}_${Math.random().toString(36).slice(2, 7)}.jpg`;
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
      confidence: fullText ? 95 : 0,
    };
  } finally {
    await Filesystem.deleteFile({
      path: tempName,
      directory: Directory.Cache,
    }).catch(() => {});
  }
}

/**
 * 2. Web / Browser Dev Fallback:
 * Uses browser native Shape Detection API (window.TextDetector) or single Tesseract pass.
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
    } catch (_) {
    } finally {
      bitmap?.close();
    }
  }

  if (nativeText && nativeText.length >= 4) {
    return { text: nativeText, confidence: 90 };
  }

  // Tesseract single pass fallback (PSM 6: Uniform block of multi-line text)
  const ret = await recognizeWithTesseract(file, { plate: false, onProgress }).catch(() => null);
  const raw = ret?.data?.text?.trim() || "";
  const conf = Math.round(Number(ret?.data?.confidence) || 0);
  return { text: raw, confidence: conf };
}

/**
 * High-performance, on-device OCR pipeline:
 * Automatically routes to native Google ML Kit on mobile or browser engine in development.
 */
export async function performOCR(file, onProgress) {
  // Use a dedicated OCR queue to avoid flooding the UI thread.
  if (!window.__ocrQueue) {
    window.__ocrQueue = Promise.resolve();
  }
  const task = window.__ocrQueue.then(() => _performOCR(file, onProgress)).catch(() => ({ text: "", confidence: 0 }));
  window.__ocrQueue = task;
  return task;
}

// Internal OCR implementation separated from the exported queue wrapper.
async function _performOCR(file, onProgress) {
  let recognizedText = "";
  let confidence = 0;
  try {
    if (Capacitor.isNativePlatform()) {
      const mlKitResult = await recognizeWithNativeMLKit(file);
      recognizedText = mlKitResult.text;
      confidence = mlKitResult.confidence;
    } else {
      const webResult = await recognizeWithWeb(file, onProgress);
      recognizedText = webResult.text;
      confidence = webResult.confidence;
    }
  } catch (err) {
    console.warn("[OCR] Primary recognition error, falling back to Web OCR:", err);
    const webFallback = await recognizeWithWeb(file, onProgress).catch(() => ({ text: "", confidence: 0 }));
    recognizedText = webFallback.text;
    confidence = webFallback.confidence;
  }

  const plate = extractVehiclePlate(recognizedText);
  const structuredIds = extractStructuredIds(recognizedText);

  const WATERMARK_WORDS = new Set([
    "shutterstock",
    "shutterstockcom",
    "getty",
    "gettyimages",
    "alamy",
    "stock",
    "adobe",
    "adobestock",
    "depositphotos",
    "vector",
    "watermark",
    "photo",
    "image",
    "copyright",
    "com",
  ]);

  const validWords = recognizedText
    .replace(/[^a-zA-Z\s]/g, " ")
    .split(/\s+/)
    .map((w) => w.trim().toLowerCase())
    .filter((w) => w.length >= 3 && /[aeiou]/.test(w) && !WATERMARK_WORDS.has(w));

  const uniqueWords = Array.from(new Set(validWords));

  let cleanDisplayText = "";
  if (plate) {
    cleanDisplayText = `Vehicle Plate: ${plate}`;
  } else if (structuredIds.length > 0) {
    cleanDisplayText = structuredIds.map((id) => `${id.type.toUpperCase()}: ${id.value}`).join("\n");
  } else {
    cleanDisplayText = uniqueWords.join(" ");
  }

  const isPure = !plate && structuredIds.length === 0 && cleanDisplayText.length === 0;

  return {
    text: cleanDisplayText,
    rawText: recognizedText,
    isPureImage: isPure,
    confidence: plate ? Math.max(90, confidence) : confidence,
    plate,
    structuredIds,
  };
}
  let recognizedText = "";
  let confidence = 0;

  try {
    if (Capacitor.isNativePlatform()) {
      const mlKitResult = await recognizeWithNativeMLKit(file);
      recognizedText = mlKitResult.text;
      confidence = mlKitResult.confidence;
    } else {
      const webResult = await recognizeWithWeb(file, onProgress);
      recognizedText = webResult.text;
      confidence = webResult.confidence;
    }
  } catch (err) {
    console.warn("[OCR] Primary recognition error, falling back to Web OCR:", err);
    const webFallback = await recognizeWithWeb(file, onProgress).catch(() => ({ text: "", confidence: 0 }));
    recognizedText = webFallback.text;
    confidence = webFallback.confidence;
  }

  const plate = extractVehiclePlate(recognizedText);
  const structuredIds = extractStructuredIds(recognizedText);

  const WATERMARK_WORDS = new Set([
    "shutterstock", "shutterstockcom", "getty", "gettyimages",
    "alamy", "stock", "adobe", "adobestock", "depositphotos",
    "vector", "watermark", "photo", "image", "copyright", "com"
  ]);

  const validWords = recognizedText
    .replace(/[^a-zA-Z\s]/g, " ")
    .split(/\s+/)
    .map((w) => w.trim().toLowerCase())
    .filter((w) => w.length >= 3 && /[aeiou]/.test(w) && !WATERMARK_WORDS.has(w));

  const uniqueWords = Array.from(new Set(validWords));

  let cleanDisplayText = "";
  if (plate) {
    cleanDisplayText = `Vehicle Plate: ${plate}`;
  } else if (structuredIds.length > 0) {
    cleanDisplayText = structuredIds.map((id) => `${id.type.toUpperCase()}: ${id.value}`).join("\n");
  } else {
    cleanDisplayText = uniqueWords.join(" ");
  }

  const isPure = !plate && structuredIds.length === 0 && cleanDisplayText.length === 0;

  return {
    text: cleanDisplayText,
    rawText: recognizedText,
    isPureImage: isPure,
    confidence: plate ? Math.max(90, confidence) : confidence,
    plate,
    structuredIds,
  };
}
