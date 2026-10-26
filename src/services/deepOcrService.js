// src/services/deepOcrService.js
// Client-side manager for our dedicated off-thread Web Worker.
// Includes auto-termination after idle timeout to drop RAM consumption.

let workerInstance = null;
let jobId = 0;
let idleTimer = null;
const pendingJobs = new Map();

function resetIdleTimer() {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    if (workerInstance && pendingJobs.size === 0) {
      workerInstance.terminate();
      workerInstance = null;
      console.log("[deepOcrService] Worker terminated after idle timeout to reclaim RAM.");
    }
  }, 25000);
}

function getWorker() {
  resetIdleTimer();
  if (workerInstance) return workerInstance;

  workerInstance = new Worker(
    new URL("../workers/ocrWorker.js", import.meta.url),
    { type: "module" }
  );

  workerInstance.onmessage = (e) => {
    resetIdleTimer();
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

export function prewarmDeepLearningOCR() {
  // Deliberately no-op on initial page load to save 120MB+ RAM until user uploads an image
}

export async function runDeepOcrInWorker(imageSource, timeoutMs = 10000) {
  const worker = getWorker();
  const currentId = ++jobId;

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pendingJobs.delete(currentId);
      resetIdleTimer();
      reject(new Error("OCR worker execution timed out"));
    }, timeoutMs);

    pendingJobs.set(currentId, {
      resolve: (data) => {
        clearTimeout(timer);
        resetIdleTimer();
        resolve(data);
      },
      reject: (err) => {
        clearTimeout(timer);
        resetIdleTimer();
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

export function normalizeRegistrationPlate(text) {
  if (!text) return null;
  const compact = text.toUpperCase().replace(/[^A-Z0-9]/g, "");
  const plateMatch = compact.match(/([A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4})/);
  return plateMatch ? plateMatch[1] : null;
}
