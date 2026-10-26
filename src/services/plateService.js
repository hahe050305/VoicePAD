import { recognizeWithTesseract } from "./tesseractService.js";

const VALID_INDIAN_STATES = new Set([
  "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "GA", "GJ", "HP", "HR",
  "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL",
  "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
]);

const STATE_PREFIX = [...VALID_INDIAN_STATES].join("|");
const PLATE_PATTERN = new RegExp(`(?:${STATE_PREFIX})(?:[0-9]{1,2})(?:[A-Z]{0,3})(?:[0-9]{3,4})`);
const BH_PLATE_PATTERN = /\d{2}BH\d{4}[A-Z]{1,2}/;

function normalizeOcrLine(line) {
  return line
    .toUpperCase()
    .replace(/SHUTTERSTOCK|GETTY(?:IMAGES)?|ALAMY|WATERMARK|ADOBE(?:STOCK)?|DEPOSITPHOTOS/g, " ")
    .replace(/[^A-Z0-9]/g, "")
    // Correct common substitutions only where the expected character is numeric.
    .replace(/^([A-Z]{2})O(?=\d)/, (_, prefix) => `${prefix}0`)
    .replace(/^([A-Z]{2})I(?=\d)/, (_, prefix) => `${prefix}1`)
    .replace(/^(\d{2})8H(?=\d)/, "$1BH");
}

function formatPlate(value) {
  const stateEnd = 2;
  const district = value.slice(stateEnd).match(/^\d{1,2}/)?.[0] || "";
  const seriesStart = stateEnd + district.length;
  const series = value.slice(seriesStart).match(/^[A-Z]{0,3}/)?.[0] || "";
  const serial = value.slice(seriesStart + series.length);
  return `${value.slice(0, stateEnd)}${district}${series} ${serial}`.trim();
}

export function parsePlateString(raw) {
  if (!raw) return null;

  const lines = String(raw)
    .split(/[\r\n]+/)
    .map(normalizeOcrLine)
    .filter(Boolean);
  const candidates = [...lines];
  // Motorcycle plates are often printed on two rows; join only adjacent OCR lines.
  for (let index = 0; index < lines.length - 1; index += 1) {
    candidates.push(`${lines[index]}${lines[index + 1]}`);
  }

  for (const candidate of candidates) {
    const bhMatch = candidate.match(BH_PLATE_PATTERN);
    if (bhMatch) {
      const value = bhMatch[0];
      return `${value.slice(0, 4)} ${value.slice(4, 8)} ${value.slice(8)}`;
    }
    const match = candidate.match(PLATE_PATTERN);
    if (!match) continue;

    const value = match[0];
    if (VALID_INDIAN_STATES.has(value.slice(0, 2))) return formatPlate(value);
  }
  return null;
}

async function loadImageBitmap(imageSource) {
  if (typeof createImageBitmap === "function") {
    return { bitmap: await createImageBitmap(imageSource), cleanup: () => {} };
  }

  const url = URL.createObjectURL(imageSource);
  try {
    const image = await new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = reject;
      element.src = url;
    });
    return { bitmap: image, cleanup: () => URL.revokeObjectURL(url) };
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
}

async function cropToBlob(source, x, y, width, height) {
  const scale = Math.min(1.5, 1600 / width);
  const outputWidth = Math.max(1, Math.round(width * scale));
  const outputHeight = Math.max(1, Math.round(height * scale));
  const canvas = typeof OffscreenCanvas === "function"
    ? new OffscreenCanvas(outputWidth, outputHeight)
    : Object.assign(document.createElement("canvas"), { width: outputWidth, height: outputHeight });
  const context = canvas.getContext("2d");
  if (!context) return null;

  context.filter = "grayscale(1) contrast(1.35)";
  context.drawImage(source, x, y, width, height, 0, 0, outputWidth, outputHeight);
  if (typeof OffscreenCanvas === "function" && canvas instanceof OffscreenCanvas) {
    return canvas.convertToBlob({ type: "image/jpeg", quality: 0.88 });
  }
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.88));
}

/** Yield one bounded crop at a time so encoded crops are not retained together. */
export async function* getAggressivePlateCrops(imageSource) {
  let loaded;
  let sourceCanvas;
  try {
    loaded = await loadImageBitmap(imageSource);
    const source = loaded.bitmap;
    const longest = Math.max(source.width, source.height);
    const scale = Math.min(1, 1600 / longest);
    const width = Math.max(1, Math.round(source.width * scale));
    const height = Math.max(1, Math.round(source.height * scale));

    sourceCanvas = typeof OffscreenCanvas === "function"
      ? new OffscreenCanvas(width, height)
      : Object.assign(document.createElement("canvas"), { width, height });
    const context = sourceCanvas.getContext("2d");
    if (!context) return;
    context.drawImage(source, 0, 0, width, height);
    if (loaded.bitmap.close) loaded.bitmap.close();
    loaded.cleanup();
    loaded.cleanup = () => {};

    const bandHeight = Math.max(1, Math.round(height * 0.48));
    const yPositions = [...new Set([
      Math.round(height * 0.05),
      Math.round((height - bandHeight) / 2),
      Math.round(height * 0.95 - bandHeight),
    ].map((y) => Math.max(0, Math.min(height - bandHeight, y))))];

    for (const y of yPositions) {
      const blob = await cropToBlob(sourceCanvas, 0, y, width, bandHeight);
      if (blob) yield blob;
      // Let the browser paint/input between OCR preparation steps.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  } catch (error) {
    console.debug("[plateService] Could not prepare plate crops:", error);
  } finally {
    if (sourceCanvas) {
      sourceCanvas.width = 0;
      sourceCanvas.height = 0;
    }
    if (loaded?.bitmap?.close) loaded.bitmap.close();
    loaded?.cleanup();
  }
}

export async function runAggressiveANPR(imageSource) {
  let best = null;
  try {
    for await (const crop of getAggressivePlateCrops(imageSource)) {
      try {
        const result = await recognizeWithTesseract(crop, { plate: true });
        const plate = parsePlateString(result?.data?.text || "");
        const confidence = Number(result?.data?.confidence) || 0;
        if (plate && confidence >= 35 && (!best || confidence > best.confidence)) {
          best = { plate, confidence };
          if (confidence >= 75) break;
        }
      } catch (error) {
        console.debug("[plateService] Plate OCR crop failed:", error);
      }
    }
  } catch (error) {
    console.debug("[plateService] Plate recognition failed:", error);
  }
  return best;
}
