// src/services/tesseractService.js
import { createWorker } from "tesseract.js";
import { parsePlateString } from "./plateService.js";

let workerPromise = null;
let idleTimer = null;
let activeProgress = null;
let queue = Promise.resolve();
let ocrWarmupPromise = null;
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
      // Load eng.traineddata from the app's own bundle (public/eng.traineddata)
      // so it works fully offline — no CDN fetch needed after the first install.
      langPath: "/",
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
export function prewarmOCR(delay = 0) {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (!ocrWarmupPromise) {
    ocrWarmupPromise = new Promise((resolve) => setTimeout(resolve, delay))
      .then(() => getTesseractWorker())
      .catch((error) => {
        ocrWarmupPromise = null;
        console.warn("[OCR] Pre-warming skipped:", error);
        return null;
      });
  }
  return ocrWarmupPromise;
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

function detectPlateRegion(data, width, height) {
  let bestScore = -1;
  let bestRect = null;

  const winWidths = [
    Math.round(width * 0.25),
    Math.round(width * 0.38),
    Math.round(width * 0.50),
  ];

  for (const w of winWidths) {
    for (const aspect of [1.4, 2.0, 3.5]) {
      const h = Math.round(w / aspect);
      if (h < 25 || h > height * 0.45) continue;

      const yStep = Math.max(12, Math.round(height * 0.06));
      const xStep = Math.max(12, Math.round(width * 0.06));

      for (let y = Math.round(height * 0.15); y + h < height * 0.92; y += yStep) {
        for (let x = Math.round(width * 0.08); x + w < width * 0.92; x += xStep) {
          let sumLum = 0;
          let edgeSum = 0;
          let samples = 0;

          for (let sampleY = y; sampleY < y + h; sampleY += 10) {
            for (let sampleX = x; sampleX < x + w; sampleX += 10) {
              const idx = (sampleY * width + sampleX) * 4;
              const lum = (data[idx] * 299 + data[idx + 1] * 587 + data[idx + 2] * 114) / 1000;
              sumLum += lum;

              if (sampleX + 4 < width) {
                const nextIdx = (sampleY * width + (sampleX + 4)) * 4;
                const nextLum = (data[nextIdx] * 299 + data[nextIdx + 1] * 587 + data[nextIdx + 2] * 114) / 1000;
                edgeSum += Math.abs(lum - nextLum);
              }
              samples++;
            }
          }

          const meanLum = samples > 0 ? sumLum / samples : 0;
          const meanEdge = samples > 0 ? edgeSum / samples : 0;

          if (meanLum > 125 && meanEdge > 8) {
            const score = meanLum * 0.5 + meanEdge * 2.5;
            if (score > bestScore) {
              bestScore = score;
              bestRect = { x, y, width: w, height: h };
            }
          }
        }
      }
    }
  }

  return bestRect;
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
      const contrast = Math.max(0, Math.min(255, Math.round((gray - 128) * 1.25 + 128)));
      data[i] = contrast;
      data[i + 1] = contrast;
      data[i + 2] = contrast;
    }
    ctx.putImageData(imgData, 0, 0);

    const contrastBlob = canvas.convertToBlob
      ? await canvas.convertToBlob({ type: "image/png" })
      : await new Promise((r) => canvas.toBlob(r, "image/png"));

    // Tighter center/lower crop focused on vehicle & document centers (excludes outer gravel/background)
    const cropX = Math.round(width * 0.12);
    const cropY = Math.round(height * 0.22);
    const cropW = Math.round(width * 0.76);
    const cropH = Math.round(height * 0.68);

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

    // Localized high-contrast plate crop
    let plateCropBlob = null;
    const detectedPlate = detectPlateRegion(data, width, height);
    if (detectedPlate) {
      const padX = Math.round(detectedPlate.width * 0.08);
      const padY = Math.round(detectedPlate.height * 0.08);
      const pX = Math.max(0, detectedPlate.x - padX);
      const pY = Math.max(0, detectedPlate.y - padY);
      const pW = Math.min(width - pX, detectedPlate.width + padX * 2);
      const pH = Math.min(height - pY, detectedPlate.height + padY * 2);

      const pCanvas = typeof OffscreenCanvas === "function"
        ? new OffscreenCanvas(pW, pH)
        : document.createElement("canvas");
      pCanvas.width = pW;
      pCanvas.height = pH;
      const pCtx = pCanvas.getContext("2d");
      pCtx.drawImage(canvas, pX, pY, pW, pH, 0, 0, pW, pH);

      plateCropBlob = pCanvas.convertToBlob
        ? await pCanvas.convertToBlob({ type: "image/png" })
        : await new Promise((r) => pCanvas.toBlob(r, "image/png"));
    }

    return {
      baseline: baselineBlob,
      contrast: contrastBlob,
      crop: cropBlob,
      plateCrop: plateCropBlob,
    };
  } finally {
    bitmap.close();
  }
}

export function scoreCandidateOcr(text, confidence, hasPlate) {
  if (!text || !text.trim()) return -100;
  const clean = text.trim();
  const nonSpaceChars = clean.replace(/\s+/g, "");
  if (!nonSpaceChars.length) return -100;

  const alphanumCount = (nonSpaceChars.match(/[a-zA-Z0-9]/g) || []).length;
  const alphanumRatio = alphanumCount / nonSpaceChars.length;
  const symbolCount = nonSpaceChars.length - alphanumCount;

  // Heavily penalize high-symbol ratio and low confidence
  const symbolPenalty = (1 - alphanumRatio) * 80 + symbolCount * 2;
  const lowConfPenalty = confidence < 60 ? (60 - confidence) * 2 : 0;

  // Reward plate presence, high confidence, and clean alphanumeric ratio
  // Notice: "+ text.length" is completely removed so hallucinated long noise is not favored
  const baseScore = (confidence * 0.8) + (alphanumRatio * 20);
  const plateBonus = hasPlate ? 200 : 0;

  return baseScore + plateBonus - symbolPenalty - lowConfPenalty;
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
  } catch {
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
      const score = scoreCandidateOcr(text, confidence, hasPlate);
      if (score > bestScore) {
        bestScore = score;
        bestResult = result;
      }
      return { result, text, hasPlate, score };
    } catch {
      return null;
    }
  };

  // PASS 0: If targeted plate candidate crop was detected, test it with PSM 6
  if (variants.plateCrop) {
    const platePass = await testPass(variants.plateCrop, "6");
    if (platePass?.hasPlate || (options.plate && (platePass?.score || 0) >= 50)) {
      return platePass.result;
    }
  }

  // PASS 1: Focused center crop (catches vehicle plates and centered cards without background gravel)
  const pass1 = await testPass(variants.crop, "6");
  if (pass1?.hasPlate || (!options.plate && (pass1?.score || 0) >= 70)) {
    return pass1.result;
  }

  // PASS 2: Contrast-enhanced image (use PSM 6 for plates, PSM 11 for sparse documents)
  const pass2 = await testPass(variants.contrast, options.plate ? "6" : "11");
  if (pass2?.hasPlate || (!options.plate && (pass2?.score || 0) >= 70)) {
    return pass2.result;
  }

  // PASS 3: Baseline image
  const pass3 = await testPass(variants.baseline, "6");
  if (pass3?.hasPlate) {
    return pass3.result;
  }

  return bestResult || pass1?.result || { data: { text: "", confidence: 0 } };
}
