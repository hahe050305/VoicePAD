// src/services/deepOcrService.js
// Client-side manager for our dedicated off-thread Web Worker.
// Keeps the main UI completely fluid and responsive during inference.

let workerInstance = null;
let jobId = 0;
const pendingJobs = new Map();

function getWorker() {
  if (workerInstance) return workerInstance;

  // Instantiate standard Vite-supported ES module worker
  workerInstance = new Worker(
    new URL("../workers/ocrWorker.js", import.meta.url),
    { type: "module" }
  );

  workerInstance.onmessage = (e) => {
    const { id, type, rawText, plate, error } = e.data;
    if (type === "PREWARM_DONE") return;

    const resolver = pendingJobs.get(id);
    if (!resolver) return;

    pendingJobs.delete(id);
    if (type === "SUCCESS") {
      resolver.resolve({ rawText, plate });
    } else {
      resolver.reject(new Error(error || "Worker failed"));
    }
  };

  workerInstance.onerror = (err) => {
    console.error("[deepOcrService] Worker error:", err);
  };

  return workerInstance;
}

/**
 * Pre-warms the worker and model in the background during idle time.
 */
export function prewarmDeepLearningOCR() {
  try {
    const worker = getWorker();
    worker.postMessage({ type: "PREWARM", id: ++jobId });
  } catch (err) {
    console.debug("[deepOcrService] Pre-warm failed:", err);
  }
}

/**
 * Executes Deep Learning OCR completely off the main thread.
 * Returns { rawText, plate } without blocking UI renders or interactions.
 */
export async function runDeepOcrInWorker(imageSource, timeoutMs = 12000) {
  const worker = getWorker();
  const currentId = ++jobId;

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingJobs.delete(currentId);
      reject(new Error("OCR worker execution timed out"));
    }, timeoutMs);

    pendingJobs.set(currentId, {
      resolve: (data) => {
        clearTimeout(timer);
        resolve(data);
      },
      reject: (err) => {
        clearTimeout(timer);
        reject(err);
      },
    });

    worker.postMessage({
      id: currentId,
      type: "RECOGNIZE",
      imageSource,
    });
  });
}

/**
 * Normalizes vehicle registration plates (Indian & international standard alphanumeric)
 * e.g., "BR 01 C J 6440" -> "BR01CJ6440"
 */
export function normalizeRegistrationPlate(text) {
  if (!text) return null;
  const compact = text.toUpperCase().replace(/[^A-Z0-9]/g, "");
  const plateMatch = compact.match(/([A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4})/);
  return plateMatch ? plateMatch[1] : null;
}
