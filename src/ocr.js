import { createWorker } from "tesseract.js";

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
    })();
  }
  return initPromise;
}

/**
 * Runs OCR on an image file or blob.
 * @param {Blob | File} file
 * @param {(progressPercent: number) => void} [onProgress]
 * @returns {Promise<string>} Recognized text
 */
export async function performOCR(file, onProgress) {
  const worker = await getWorker(onProgress);
  const ret = await worker.recognize(file);
  return ret?.data?.text?.trim() || "";
}
