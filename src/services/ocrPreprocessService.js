let worker = null;
let nextRequestId = 0;
const requests = new Map();
let idleTimer = null;

function getWorker() {
  if (worker) return worker;

  worker = new Worker(new URL("../workers/ocrPreprocessWorker.js", import.meta.url), {
    type: "module",
  });
  worker.onmessage = ({ data }) => {
    const request = requests.get(data.id);
    if (!request) return;
    requests.delete(data.id);
    if (data.error) request.reject(new Error(data.error));
    else request.resolve({
      baseline: data.baseline,
      scanner: data.scanner || null,
      variants: data.variants,
      perspectiveVariants: data.perspectiveVariants || [],
      focusedRegions: data.focusedRegions || [],
      rotated: data.rotated || [],
    });
  };
  const rejectPending = (event) => {
    const error = new Error(event.message || "OCR image enhancement failed.");
    for (const request of requests.values()) request.reject(error);
    requests.clear();
    worker?.terminate();
    worker = null;
  };
  worker.onerror = rejectPending;
  worker.onmessageerror = rejectPending;
  return worker;
}

export function createOcrVariants(image) {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = null;
  const id = ++nextRequestId;
  const result = new Promise((resolve, reject) => {
    requests.set(id, { resolve, reject });
    try {
      getWorker().postMessage({ id, image });
    } catch (error) {
      requests.delete(id);
      reject(error);
    }
  });
  return result.finally(() => {
    idleTimer = setTimeout(() => {
      worker?.terminate();
      worker = null;
      idleTimer = null;
    }, 3000);
  });
}
