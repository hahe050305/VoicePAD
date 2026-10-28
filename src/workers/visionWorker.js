import { HYPERLOCAL_COCO_TAGS } from "../services/visionTags.js";

let detectorPromise = null;
const INCIDENTAL_PROPS = new Set([
  "chair", "dining table", "cup", "bottle", "bowl", "fork", "knife", "spoon",
  "mouse", "keyboard", "remote", "bench", "vase", "potted plant", "fire hydrant", "traffic light",
]);
const LOW_MEMORY_DEVICE = typeof navigator !== "undefined" &&
  Number.isFinite(navigator.deviceMemory) &&
  navigator.deviceMemory <= 4;
const MAX_INFERENCE_DIMENSION = LOW_MEMORY_DEVICE ? 320 : 384;

function publishModelStatus(status) {
  self.postMessage({ type: "MODEL_STATUS", ...status });
}

async function getDetector() {
  if (!detectorPromise) {
    publishModelStatus({ state: "loading", progress: null });
    detectorPromise = (async () => {
      const {
        AutoModelForObjectDetection,
        AutoProcessor,
        ObjectDetectionPipeline,
        env,
      } = await import("@huggingface/transformers");

      env.allowLocalModels = false;
      env.useBrowserCache = true;
      const modelId = "Xenova/yolos-tiny";
      const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
      const device = hasWebGPU ? "webgpu" : "wasm";
      const dtype = hasWebGPU ? "fp32" : "q8";
      const progress_callback = (progress) => {
        if (progress.status === "progress_total") {
          publishModelStatus({
            state: "loading",
            progress: Math.round(progress.progress),
          });
        }
      };
      let model;
      let processor;

      try {
        [model, processor] = await Promise.all([
          AutoModelForObjectDetection.from_pretrained(modelId, {
            device,
            dtype,
            progress_callback,
          }),
          AutoProcessor.from_pretrained(modelId),
        ]);
      } catch (error) {
        console.warn("[Vision] WebGPU initialization failed; falling back to quantized WASM:", error);
        [model, processor] = await Promise.all([
          AutoModelForObjectDetection.from_pretrained(modelId, {
            device: "wasm",
            dtype: "q8",
            progress_callback,
          }),
          AutoProcessor.from_pretrained(modelId),
        ]);
      }
      const pipeline = new ObjectDetectionPipeline({ task: "object-detection", model, processor });
      publishModelStatus({ state: "ready", progress: 100 });
      return pipeline;
    })().catch((error) => {
      detectorPromise = null;
      publishModelStatus({ state: "error", error: error?.message || String(error) });
      throw error;
    });
  }
  return detectorPromise;
}

async function makeInferenceUrl(imageSource) {
  const blob = await toBlob(imageSource);
  const bitmap = await createImageBitmap(blob);
  try {
    const scale = Math.min(1, MAX_INFERENCE_DIMENSION / Math.max(bitmap.width, bitmap.height));
    if (scale === 1) return URL.createObjectURL(blob);
    if (typeof OffscreenCanvas !== "function") return URL.createObjectURL(blob);

    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d");
    if (!context) return URL.createObjectURL(blob);
    context.drawImage(bitmap, 0, 0, width, height);
    const resized = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.88 });
    return URL.createObjectURL(resized);
  } finally {
    bitmap.close();
  }
}

async function toBlob(imageSource) {
  return imageSource instanceof Blob ? imageSource : (await fetch(imageSource)).blob();
}

async function createThumbnail(imageSource) {
  const bitmap = await createImageBitmap(await toBlob(imageSource));
  try {
    const maxDim = LOW_MEMORY_DEVICE ? 320 : 480;
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not prepare image preview.");
    context.drawImage(bitmap, 0, 0, width, height);
    return await canvas.convertToBlob({ type: "image/jpeg", quality: 0.78 });
  } finally {
    bitmap.close();
  }
}

async function detect(imageSource, threshold) {
  const detector = await getDetector();
  const url = await makeInferenceUrl(imageSource);
  try {
    const raw = await detector(url, { threshold, percentage: true });
    const detections = (raw || []).map((item) => ({
      label: item.label.toLowerCase(),
      score: Math.round(item.score * 100) / 100,
      box: item.box,
    }));
    const salient = detections.filter((item) => {
      const width = Math.max(0, (item.box?.xmax ?? 0) - (item.box?.xmin ?? 0));
      const height = Math.max(0, (item.box?.ymax ?? 0) - (item.box?.ymin ?? 0));
      const area = (width * height) / 10000;
      return INCIDENTAL_PROPS.has(item.label) ? area >= 0.015 : area >= 0.003;
    });
    const labels = [...new Set(salient.map((item) => item.label))];
    const tags = new Set();
    for (const label of labels) {
      const mapped = HYPERLOCAL_COCO_TAGS[label];
      if (mapped) mapped.forEach((tag) => tags.add(tag));
      else tags.add(`#${label.replace(/\s+/g, "_")}`);
    }
    return { labels, tags: [...tags], detections: salient };
  } finally {
    URL.revokeObjectURL(url);
  }
}

self.onmessage = async ({ data }) => {
  try {
    let result;
    if (data.type === "detectPlate") {
      const detections = await detect(data.imageSource, data.threshold);
      const plates = detections.detections.filter((d) => /plate/i.test(d.label));
      const best = plates.reduce((current, candidate) =>
        current && current.score > candidate.score ? current : candidate, null);
      result = best ? { box: best.box, score: best.score } : null;
    } else {
      result = data.type === "thumbnail"
        ? await createThumbnail(data.imageSource)
        : await detect(data.imageSource, data.threshold);
    }
    self.postMessage({ id: data.id, result });
  } catch (error) {
    self.postMessage({ id: data.id, error: error?.message || "Vision detection failed." });
  };
};
