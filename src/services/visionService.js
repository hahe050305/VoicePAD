import { HYPERLOCAL_COCO_TAGS } from "./visionTags.js";
export { HYPERLOCAL_COCO_TAGS };

export async function getDetectedLabelsAndTags(imageSource, threshold = 0.40) {
  const task = visionQueue.then(async () => {
    if (visionIdleTimer) clearTimeout(visionIdleTimer);
    visionIdleTimer = null;
    const worker = getVisionWorker();
    const id = ++visionRequestId;
    const result = new Promise((resolve, reject) => {
      visionRequests.set(id, { resolve, reject });
      worker.postMessage({ id, imageSource, threshold });
    });
    try {
      return await result;
    } finally {
      visionRequests.delete(id);
      if (visionIdleTimer) clearTimeout(visionIdleTimer);
      visionIdleTimer = setTimeout(() => {
        visionWorker?.terminate();
        visionWorker = null;
        visionIdleTimer = null;
      }, 20000);
    }
  });
  visionQueue = task.catch(() => {});
  return task;
}

let visionWorker = null;
let visionIdleTimer = null;
let visionRequestId = 0;
let visionQueue = Promise.resolve();
const visionRequests = new Map();

function getVisionWorker() {
  if (visionWorker) return visionWorker;
  visionWorker = new Worker(new URL("../workers/visionWorker.js", import.meta.url), {
    type: "module",
  });
  visionWorker.onmessage = ({ data }) => {
    const request = visionRequests.get(data.id);
    if (!request) return;
    visionRequests.delete(data.id);
    if (data.error) request.reject(new Error(data.error));
    else request.resolve(data.result);
  };
  const rejectPending = (event) => {
    const error = new Error(event.message || "Vision worker failed.");
    for (const request of visionRequests.values()) request.reject(error);
    visionRequests.clear();
    visionWorker?.terminate();
    visionWorker = null;
  };
  visionWorker.onerror = rejectPending;
  visionWorker.onmessageerror = rejectPending;
  return visionWorker;
}
