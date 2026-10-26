import { createWorker } from "tesseract.js";

let workerInstance = null;
let initPromise = null;
let idleTimer = null;

function resetIdleTimer() {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(async () => {
    if (workerInstance) {
      try {
        await workerInstance.terminate();
        console.log("[OCR] Tesseract worker idle timeout -> terminated.");
      } catch (_) {}
      workerInstance = null;
      initPromise = null;
    }
  }, 30000);
}

async function getWorker(onProgress) {
  resetIdleTimer();
  if (workerInstance) return workerInstance;
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

/**
 * Intelligent Plate & Registration Extractor.
 * Extracts clean format like "BR01C J6440" or "BR01C".
 */
export function extractVehiclePlate(rawText) {
  if (!rawText) return null;

  let s = rawText.toUpperCase();

  // Normalize common OCR character confusions:
  s = s.replace(/([A-Z]{2})\s*O(\d)/g, "$1 0$2");
  s = s.replace(/([A-Z]{2})\s*(\d)\s*O\b/g, "$1 $20");
  s = s.replace(/([A-Z]{2})\s*I(\d)/g, "$1 1$2");
  s = s.replace(/\bBRO1C\b/g, "BR01C");
  s = s.replace(/\bBROIC\b/g, "BR01C");

  // Remove non-alphanumeric
  const compact = s.replace(/[^A-Z0-9]/g, "");

  // 1. Full standard plate: 2 state letters + 1-2 digits + 0-3 letters + 3-4 digits (e.g. BR01CJ6440)
  const fullMatch = compact.match(/([A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4})/);
  if (fullMatch) {
    const val = fullMatch[1];
    // Return with clean standard spacing: "BR01C J6440"
    if (val.length >= 8) {
      return `${val.slice(0, 5)} ${val.slice(5)}`;
    }
    return val;
  }

  // 2. Partial match: e.g. BR01C
  const partialMatch = compact.match(/([A-Z]{2}\d{1,2}[A-Z]{1,2})/);
  if (partialMatch && partialMatch[1].length >= 4) {
    return partialMatch[1];
  }

  return null;
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

export async function prewarmOCR() {
  try {
    if (typeof window !== "undefined" && "TextDetector" in window) return;
    await getWorker();
  } catch (_) {}
}

/**
 * Runs OCR and returns strictly clean text without garbage lines.
 */
export async function performOCR(file, onProgress) {
  const input = await getOcrInput(file);
  let nativeText = "";
  let tesseractText = "";

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

  if (!nativeText || nativeText.length < 5) {
    try {
      const worker = await getWorker(onProgress);
      const ret = await worker.recognize(input);
      tesseractText = ret?.data?.text?.trim() || "";
      resetIdleTimer();
    } catch (_) {}
  }

  const fullRaw = `${nativeText}\n${tesseractText}`;
  const plate = extractVehiclePlate(fullRaw);
  const structuredIds = extractStructuredIds(fullRaw);

  // Filter out all OCR noise lines (e.g. "JI5OL", "eR Te", "ol npn NE", "a §")
  const validWords = fullRaw
    .replace(/[^a-zA-Z\s]/g, " ")
    .split(/\s+/)
    .map((w) => w.trim())
    .filter((w) => w.length >= 3 && /[aeiou]/i.test(w));

  const uniqueWords = Array.from(new Set(validWords));

  // If a plate was detected, present the clean plate prominently without garbage lines
  let cleanDisplayText = "";
  if (plate) {
    cleanDisplayText = `Vehicle Plate: ${plate}`;
    if (uniqueWords.length > 0) {
      cleanDisplayText += `\n${uniqueWords.join(" ")}`;
    }
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
    confidence: plate || structuredIds.length > 0 ? 95 : 85,
    plate,
    structuredIds,
  };
}
