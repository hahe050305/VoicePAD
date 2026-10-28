import { createWorker } from "tesseract.js";
import { createOcrVariants } from "./ocrPreprocessService.js";
import { scoreOcrResult } from "./ocrRanking.js";
import { parsePlateString } from "./plateService.js";

let workerPromise = null;
let idleTimer = null;
let activeProgress = null;
let queue = Promise.resolve();
const idleTimeout = typeof navigator !== "undefined" &&
  Number.isFinite(navigator.deviceMemory) &&
  navigator.deviceMemory <= 4
  ? 10000
  : 20000;

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

async function getWorker() {
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

/** Serialize requests because Tesseract's whitelist and page mode are worker-wide settings. */
export function recognizeWithTesseract(image, {
  plate = false,
  pageSegmentationMode = plate ? "6" : "3",
  onProgress,
} = {}) {
  const task = queue.then(async () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = null;
    const worker = await getWorker();
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

export async function recognizeWithTesseractEnhancements(image, options = {}) {
  let processed = {
    baseline: null,
    variants: [],
    perspectiveVariants: [],
    focusedRegions: [],
    rotated: [],
  };
  try {
    processed = await createOcrVariants(image);
  } catch (error) {
    console.warn("[OCR] Could not prepare enhanced image variants; using the original:", error);
  }

  let bestResult = null;
  let bestScore = -1;
  let lastError = null;
  const recordCandidate = async (candidate, pageSegmentationMode) => {
    try {
      const result = await recognizeWithTesseract(candidate, {
        ...options,
        pageSegmentationMode,
      });
      const score = scoreOcrResult(result, options);
      if (score > bestScore) {
        bestResult = result;
        bestScore = score;
      }
    } catch (error) {
      lastError = error;
      console.warn("[OCR] Recognition pass failed; continuing with other image variants:", error);
    }
  };

  const baseCandidates = [
    processed.baseline || image,
    ...processed.perspectiveVariants,
    ...processed.variants,
  ];
  const baseModes = options.plate
    ? ["7", "7", "8", "6", "6"]
    : ["3", "6", "6", "11", "3"];
  for (const [index, candidate] of baseCandidates.entries()) {
    await recordCandidate(candidate, baseModes[index]);
  }

  const bestText = bestResult?.data?.text?.trim() || "";
  const needsOrientationPass = options.plate
    ? !parsePlateString(bestText)
    : bestScore < 55 || (bestText.match(/[A-Z0-9]/gi) || []).length < 8;
  if (needsOrientationPass) {
    for (const [index, focusedRegion] of processed.focusedRegions.entries()) {
      await recordCandidate(focusedRegion, options.plate
        ? (index % 2 === 0 ? "7" : "8")
        : (index % 2 === 0 ? "6" : "11"));
    }
  }
  if (needsOrientationPass && (options.plate ? !parsePlateString(bestResult?.data?.text) : bestScore < 30)) {
    for (const candidate of processed.rotated) {
      await recordCandidate(candidate, options.plate ? "7" : "6");
    }
  }
  if (!bestResult && lastError) throw lastError;
  return bestResult;
}
