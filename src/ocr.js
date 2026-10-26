import { runAggressiveANPR, parsePlateString } from "./services/plateService.js";
import { recognizeWithTesseract } from "./services/tesseractService.js";

export function isPureImageText(text) {
  if (!text || !text.trim()) return true;
  const clean = text.trim();
  if (extractVehiclePlate(clean) || extractStructuredIds(clean).length > 0) return false;

  const words = clean.split(/\s+/).filter(w => /^[a-zA-Z]{3,}$/.test(w));
  if (words.length >= 1) return false;

  const validNumbers = clean.split(/\s+/).filter(w => /^[A-Z0-9]{3,}$/i.test(w));
  if (validNumbers.length >= 1) return false;

  return true;
}

const VALID_INDIAN_STATES = new Set([
  "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
  "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
  "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
]);

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

async function getOcrInput(file) {
  if (typeof createImageBitmap !== "function") return file;
  try {
    const bitmap = await createImageBitmap(file);
    try {
      const maxDim = 1200;
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
      canvas.width = 0;
      canvas.height = 0;
      return blob || file;
    } finally {
      bitmap.close();
    }
  } catch {
    return file;
  }
}

export async function performOCR(file, onProgress) {
  // 1. Aggressive Multi-Window ANPR Pass (Multi-crop + Contrast enhancement)
  let detectedPlate = null;
  try {
    detectedPlate = await runAggressiveANPR(file);
  } catch (_) {}

  const input = await getOcrInput(file);
  let nativeText = "";
  let tesseractText = "";
  let tesseractRaw = "";
  let tesseractConfidence = 0;

  if (typeof window !== "undefined" && "TextDetector" in window) {
    let bitmap;
    try {
      const detector = new window.TextDetector();
      bitmap = await createImageBitmap(input);
      const detections = await detector.detect(bitmap);
      nativeText = (detections || []).map((d) => d.rawValue || "").join(" ").trim();
    } catch (_) {
    } finally {
      bitmap?.close();
    }
  }

  // Always run the general OCR fallback when ANPR found no plate; native text
  // detection often returns unrelated scene text while missing small plates.
  if (!nativeText || nativeText.length < 5 || !detectedPlate) {
    try {
      const ret = await recognizeWithTesseract(input, { onProgress });
      tesseractRaw = ret?.data?.text?.trim() || "";
      const words = ret?.data?.words || [];
      const readableWords = words
        .filter((word) => Number(word.confidence) >= 45 && word.text?.trim())
        .map((word) => word.text.trim());
      tesseractText = words.length ? readableWords.join(" ") : tesseractRaw;
      tesseractConfidence = Math.round(Number(ret?.data?.confidence) || 0);
    } catch (_) {}
  }

  const fullRaw = `${nativeText}\n${tesseractRaw}`;
  const displayRaw = `${nativeText}\n${tesseractText}`;
  const plate = detectedPlate?.plate || extractVehiclePlate(fullRaw);
  const structuredIds = extractStructuredIds(fullRaw);

  const WATERMARK_WORDS = new Set([
    "shutterstock", "shutterstockcom", "getty", "gettyimages",
    "alamy", "stock", "adobe", "adobestock", "depositphotos",
    "vector", "watermark", "photo", "image", "copyright", "com"
  ]);

  const validWords = displayRaw
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
    rawText: cleanDisplayText,
    isPureImage: isPure,
    confidence: plate ? (detectedPlate?.confidence ?? tesseractConfidence) : tesseractConfidence,
    plate,
    structuredIds,
  };
}
