// src/services/visionService.js
import { HYPERLOCAL_COCO_TAGS } from "./visionTags.js";
export { HYPERLOCAL_COCO_TAGS };

/** Run a task on the shared vision WebWorker. */
function runVisionTask(type, imageSource, threshold = 0.40) {
  const task = visionQueue.then(async () => {
    if (visionIdleTimer) clearTimeout(visionIdleTimer);
    visionIdleTimer = null;
    const worker = getVisionWorker();
    const id = ++visionRequestId;
    const result = new Promise((resolve, reject) => {
      visionRequests.set(id, { resolve, reject });
      worker.postMessage({ id, type, imageSource, threshold });
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
      }, visionIdleTimeout);
    }
  });
  visionQueue = task.catch(() => {});
  return task;
}

export function createImageThumbnail(imageSource) {
  return runVisionTask("thumbnail", imageSource);
}

/** Generic object detection – returns labels, tags and raw detections. */
export function getDetectedLabelsAndTags(imageSource, threshold = 0.40) {
  return runVisionTask("detect", imageSource, threshold);
}

/**
 * Detect the licence‑plate region using the vision worker.
 * Returns { box, score } where box = { x, y, width, height, angle }.
 */
export async function detectPlateRegion(imageSource, minScore = 0.6) {
  const result = await runVisionTask("detectPlate", imageSource, minScore);
  if (!result?.box || result.score < minScore) return null;
  const { box } = result;
  return {
    box: {
      x: (box.xmin ?? 0) / 100,
      y: (box.ymin ?? 0) / 100,
      width: ((box.xmax ?? 0) - (box.xmin ?? 0)) / 100,
      height: ((box.ymax ?? 0) - (box.ymin ?? 0)) / 100,
      angle: 0, // YOLO‑tiny does not provide rotation.
    },
    score: result.score,
  };
}

// ---------------------------------------------------------------------------
// Worker management
let visionWorker = null;
let visionIdleTimer = null;
let visionRequestId = 0;
let visionQueue = Promise.resolve();
const visionRequests = new Map();
const visionIdleTimeout = typeof navigator !== "undefined" &&
  Number.isFinite(navigator.deviceMemory) &&
  navigator.deviceMemory <= 4
  ? 8000
  : 15000;

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
