// src/ocr.js
import { Capacitor } from "@capacitor/core";
import { Filesystem, Directory } from "@capacitor/filesystem";
import { TextRecognition } from "@capacitor-mlkit/text-recognition";
import { parsePlateString } from "./services/plateService.js";
import { scoreOcrText } from "./services/ocrRanking.js";
import { createOcrVariants } from "./services/ocrPreprocessService.js";
import { recognizeWithTesseractEnhancements } from "./services/tesseractService.js";
import { runDeepOcrInWorker } from "./services/deepOcrService.js";

/**
 * Post-processing filter for OCR text:
 * 1. Strips isolated punctuation/symbols that are noise artifacts.
 * 2. Drops fragmented lines with low alphanumeric ratio or single-symbol noise.
 * 3. Enforces an overall alphanumeric ratio threshold (>= 60%) unless a verified plate or structured ID is found.
 */
export function sanitizeOcrText(rawText) {
  if (!rawText || typeof rawText !== "string") return "";
  const trimmed = rawText.trim();
  if (!trimmed) return "";

  // Check if plate or structured ID is present
  const plate = parsePlateString(trimmed);
  const structuredIds = extractStructuredIds(trimmed);

  const lines = trimmed.split(/[\r\n]+/);
  const cleanedLines = [];

  for (const rawLine of lines) {
    let line = rawLine.trim();
    if (!line) continue;

    // Token-based filtering:
    // Drop tokens that have zero alphanumeric characters (isolated symbols/punctuation)
    // and strip leading/trailing punctuation from remaining words
    const words = line.split(/\s+/).filter(Boolean);
    const validWords = [];
    for (const w of words) {
      if (!/[a-zA-Z0-9]/.test(w)) continue;
      const cleaned = w.replace(/^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$/g, "");
      if (cleaned) validWords.push(cleaned);
    }
    line = validWords.join(" ");

    if (!line) continue;

    const nonSpaceChars = line.replace(/\s+/g, "");
    if (!nonSpaceChars.length) continue;

    const alphanumChars = (nonSpaceChars.match(/[a-zA-Z0-9]/g) || []).length;
    const lineRatio = alphanumChars / nonSpaceChars.length;

    // Drop line if alphanumeric content is less than 50%
    if (lineRatio < 0.5) continue;

    // Drop single-char noise lines unless it's part of a detected structured ID or plate
    if (alphanumChars < 2 && !plate && structuredIds.length === 0) continue;

    cleanedLines.push(line);
  }

  const resultText = cleanedLines.join("\n").trim();
  if (!resultText) {
    return plate ? `Vehicle Plate: ${plate}` : "";
  }

  const totalNonSpace = resultText.replace(/\s+/g, "");
  const totalAlphanum = (totalNonSpace.match(/[a-zA-Z0-9]/g) || []).length;
  const overallRatio = totalNonSpace.length > 0 ? totalAlphanum / totalNonSpace.length : 0;

  // Treat outputs as empty if alphanumeric ratio is below 60% (and no plate/ID present)
  if (overallRatio < 0.60 && !plate && structuredIds.length === 0) {
    return "";
  }

  return resultText;
}

export function isPureImageText(text) {
  if (!text || !text.trim()) return true;
  const clean = sanitizeOcrText(text);
  if (!clean || !clean.trim()) return true;
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
  if (!text) return false;
  const upper = text.toUpperCase();
  for (const kw of [
    "BOOKMYSHOW", "CINEMA", "TICKET", "SCREEN", "SEAT", "INVOICE", "RECEIPT", "BILL",
    "HACKERRANK", "ORCHESTRATE", "ATTENDING", "PARTICIPANT", "CONFERENCE", "BADGE",
    "DOT COM", "INFOWAY", "CAFE", "COFFEE", "MENU", "LOVE", "FESTIVAL"
  ]) {
    if (upper.includes(kw)) return false;
  }

  const tokens = upper.match(/[A-Z0-9]{6,12}/g) || [];
  for (const t of tokens) {
    const letters = (t.match(/[A-Z]/g) || []).length;
    const digits = (t.match(/\d/g) || []).length;
    if (letters >= 2 && digits >= 3) {
      return true;
    }
  }
  return false;
}

export function extractStructuredIds(rawText) {
  if (!rawText) return [];
  const found = [];
  const upper = rawText.toUpperCase();

  // Documents must contain supportive textual keywords to avoid false positives on random image noise
  const hasGovKeywords = /(GOVERNMENT|INDIA|BHARAT|INCOME\s*TAX|PERMANENT\s*ACCOUNT|DIRECTORATE|DEPARTMENT|REPUBLIC)/i.test(upper);
  const hasAadhaarKeywords = /(AADHAAR|UIDAI|ENROLMENT|MERA\s*AADHAAR|FATHER|DOB|YEAR\s*OF\s*BIRTH)/i.test(upper);
  const hasDlKeywords = /(DRIVING|LICENCE|LICENSE|TRANSPORT|UNION\s*OF\s*INDIA|FORM\s*7)/i.test(upper);

  // PAN: 5 uppercase letters, 4 digits, 1 uppercase letter (e.g., ABCDE1234F)
  const panMatch = upper.match(/\b([A-Z]{5}\d{4}[A-Z])\b/);
  if (panMatch && (hasGovKeywords || upper.includes("PAN") || upper.includes("ACCOUNT"))) {
    found.push({ type: "pan", value: panMatch[1], tag: "#document" });
  }

  // Aadhaar: 12 digits, formatted as 4 4 4 (e.g. 1234 5678 9012) or with explicit Aadhaar keywords
  const formattedAadhaar = upper.match(/\b(\d{4}\s\d{4}\s\d{4})\b/);
  const compact = upper.replace(/[^A-Z0-9]/g, "");
  if (formattedAadhaar) {
    found.push({ type: "aadhaar", value: formattedAadhaar[1].replace(/\s+/g, ""), tag: "#document" });
  } else if (hasAadhaarKeywords) {
    const compactAadhaar = compact.match(/(\d{12})/);
    if (compactAadhaar) {
      found.push({ type: "aadhaar", value: compactAadhaar[1], tag: "#document" });
    }
  }

  // Driving License: State code + 13-15 digits
  const dlMatch = upper.match(/\b([A-Z]{2}[-\s]?\d{13,15})\b/);
  if (dlMatch && (hasDlKeywords || upper.includes("DL") || hasGovKeywords)) {
    found.push({ type: "dl", value: dlMatch[1].replace(/[-\s]/g, ""), tag: "#document" });
  }

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
 * 2. Web / Browser Dev Mode:
 * Uses browser native Shape Detection API (window.TextDetector),
 * modern TrOCR deep learning worker, or enhanced Tesseract fallback.
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
      console.warn("[OCR] Browser text detection failed; trying TrOCR / Tesseract:", error);
    } finally {
      bitmap?.close();
    }
  }

  if (nativeText && nativeText.length > 5) {
    return { text: nativeText, confidence: 90 };
  }

  // 1. Attempt TrOCR deep-learning model in dedicated web worker
  try {
    const deepResult = await runDeepOcrInWorker(file, 8000);
    const deepText = deepResult?.rawText?.trim() || "";
    if (deepText) {
      return {
        text: deepText,
        confidence: 85,
      };
    }
  } catch (deepErr) {
    console.warn("[OCR] TrOCR worker pass unavailable; falling back to Tesseract:", deepErr?.message || deepErr);
  }

  // 2. High-speed Tesseract fallback with antialiased enhancements
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
 * Automatically routes to native Google ML Kit on mobile or modern TrOCR/Tesseract on web.
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

  // Apply OCR post-processing filter to strip punctuation noise, speckles, and low-alphanumeric text
  const sanitizedText = sanitizeOcrText(recognizedText);
  const plate = extractVehiclePlate(sanitizedText) || extractVehiclePlate(recognizedText);
  const structuredIds = extractStructuredIds(sanitizedText);

  const cleanDisplayText = sanitizedText.trim();
  const displayText = plate && !cleanDisplayText.includes(plate)
    ? `Vehicle Plate: ${plate}\n${cleanDisplayText}`.trim()
    : cleanDisplayText;

  return {
    text: displayText,
    rawText: recognizedText,
    isPureImage: isPureImageText(displayText),
    confidence,
    plate,
    plateOcrAttempted: plateRecognition.attempted,
    structuredIds,
  };
}
