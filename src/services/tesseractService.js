// src/services/tesseractService.js
import { createWorker } from "tesseract.js";
import { parsePlateString } from "./plateService.js";

let workerPromise = null;
let idleTimer = null;
let activeProgress = null;
let queue = Promise.resolve();
const idleTimeout = 180000; // Keep worker warm for 3 minutes during user session

function resetIdleTimer(worker) {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(async () => {
    if (workerPromise) {
      workerPromise = null;
      try {
        await worker.terminate();
      } catch {
        // The worker may already have stopped during page teardown.
      }
    }
  }, idleTimeout);
}

export async function getTesseractWorker() {
  if (!workerPromise) {
    workerPromise = createWorker("eng", 1, {
      logger: (message) => {
        if (message.status === "recognizing text" && activeProgress) {
          activeProgress(Math.round((message.progress || 0) * 100));
        }
      },
    }).catch((error) => {
      workerPromise = null;
      throw error;
    });
  }
  return workerPromise;
}

/** Pre-warm Tesseract worker without blocking or allocating excessive memory */
export function prewarmOCR(delay = 3200) {
  if (typeof window === "undefined") return;
  setTimeout(() => {
    getTesseractWorker().catch((err) => {
      console.warn("[OCR] Pre-warming skipped:", err);
    });
  }, delay);
}

/** Serialize requests because Tesseract's whitelist and page mode are worker-wide settings. */
export function recognizeWithTesseract(image, {
  plate = false,
  pageSegmentationMode = "6",
  onProgress,
} = {}) {
  const task = queue.then(async () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = null;
    const worker = await getTesseractWorker();
    activeProgress = onProgress;
    try {
      await worker.setParameters({
        tessedit_char_whitelist: plate ? "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n" : "",
        tessedit_pageseg_mode: String(pageSegmentationMode),
        preserve_interword_spaces: "1",
      });
      return await worker.recognize(image);
    } finally {
      activeProgress = null;
      resetIdleTimer(worker);
    }
  });
  queue = task.catch(() => {});
  return task;
}

/**
 * Fast client-side image variant generation (< 50ms):
 * - baseline: normalized to 1200px max dimension
 * - contrast: enhanced contrast for textured / low-light text
 * - crop: focused center-lower crop for handheld plates & documents
 */
async function prepareFastVariants(imageSource) {
  const blob = imageSource instanceof Blob ? imageSource : await (await fetch(imageSource)).blob();
  const bitmap = await createImageBitmap(blob);
  try {
    const maxDim = 1200;
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));

    const canvas = typeof OffscreenCanvas === "function"
      ? new OffscreenCanvas(width, height)
      : document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, width, height);

    const baselineBlob = canvas.convertToBlob
      ? await canvas.convertToBlob({ type: "image/jpeg", quality: 0.92 })
      : await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.92));

    // High-contrast variant for textured or low-light text (e.g. textured bike plates)
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      const contrast = Math.max(0, Math.min(255, (gray - 128) * 1.6 + 128));
      data[i] = contrast;
      data[i + 1] = contrast;
      data[i + 2] = contrast;
    }
    ctx.putImageData(imgData, 0, 0);

    const contrastBlob = canvas.convertToBlob
      ? await canvas.convertToBlob({ type: "image/png" })
      : await new Promise((r) => canvas.toBlob(r, "image/png"));

    // Center/lower crop where vehicle plates and documents naturally sit
    const cropX = Math.round(width * 0.08);
    const cropY = Math.round(height * 0.20);
    const cropW = Math.round(width * 0.84);
    const cropH = Math.round(height * 0.75);

    const cropCanvas = typeof OffscreenCanvas === "function"
      ? new OffscreenCanvas(cropW, cropH)
      : document.createElement("canvas");
    cropCanvas.width = cropW;
    cropCanvas.height = cropH;
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.drawImage(canvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

    const cropBlob = cropCanvas.convertToBlob
      ? await cropCanvas.convertToBlob({ type: "image/png" })
      : await new Promise((r) => cropCanvas.toBlob(r, "image/png"));

    return { baseline: baselineBlob, contrast: contrastBlob, crop: cropBlob };
  } finally {
    bitmap.close();
  }
}

/**
 * High-speed, high-accuracy OCR pipeline (< 1.5 seconds):
 * - Pass 1: Baseline image with PSM 6 (single text block)
 * - Pass 2: Contrast-enhanced image with PSM 11 (sparse text) if needed
 * - Pass 3: Center crop with PSM 6 for isolated plate / document regions
 */
export async function recognizeWithTesseractEnhancements(image, options = {}) {
  let variants;
  try {
    variants = await prepareFastVariants(image);
  } catch (err) {
    variants = { baseline: image, contrast: image, crop: image };
  }

  let bestResult = null;
  let bestScore = -1;

  const testPass = async (candidate, psm) => {
    try {
      const result = await recognizeWithTesseract(candidate, {
        ...options,
        pageSegmentationMode: psm,
      });
      const text = result?.data?.text?.trim() || "";
      const hasPlate = !!parsePlateString(text);
      const confidence = Number(result?.data?.confidence) || 0;
      const score = confidence + (hasPlate ? 200 : 0) + text.length;
      if (score > bestScore) {
        bestScore = score;
        bestResult = result;
      }
      return { result, text, hasPlate, score };
    } catch {
      return null;
    }
  };

  // PASS 1: Baseline image with PSM 6 (Single uniform block of text)
  const pass1 = await testPass(variants.baseline, "6");
  if (pass1?.hasPlate || (pass1?.score || 0) >= 120) {
    return pass1.result;
  }

  // PASS 2: Contrast-enhanced with PSM 11 (Sparse text)
  const pass2 = await testPass(variants.contrast, "11");
  if (pass2?.hasPlate || (pass2?.score || 0) >= 120) {
    return pass2.result;
  }

  // PASS 3: Focused center crop with PSM 6 (Catches 2-line motorcycle plates & centered cards)
  if (!pass1?.hasPlate && !pass2?.hasPlate) {
    const pass3 = await testPass(variants.crop, "6");
    if (pass3?.hasPlate) {
      return pass3.result;
    }
  }

  return bestResult || pass1?.result || { data: { text: "", confidence: 0 } };
}
