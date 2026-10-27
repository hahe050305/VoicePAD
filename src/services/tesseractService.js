import { createWorker } from "tesseract.js";

let workerPromise = null;
let idleTimer = null;
let activeProgress = null;
let queue = Promise.resolve();

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
  }, 30000);
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
export function recognizeWithTesseract(image, { plate = false, onProgress } = {}) {
  const task = queue.then(async () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = null;
    const worker = await getWorker();
    activeProgress = onProgress;
    try {
      await worker.setParameters({
        tessedit_char_whitelist: plate ? "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n" : "",
        tessedit_pageseg_mode: plate ? "6" : "3",
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
