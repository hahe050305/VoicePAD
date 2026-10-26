import { HYPERLOCAL_COCO_TAGS } from "../services/visionTags.js";

let detectorPromise = null;
const INCIDENTAL_PROPS = new Set([
  "chair", "dining table", "cup", "bottle", "bowl", "fork", "knife", "spoon",
  "mouse", "keyboard", "remote", "bench", "vase", "potted plant", "fire hydrant", "traffic light",
]);

async function getDetector() {
  if (!detectorPromise) {
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
      let model;
      let processor;

      try {
        [model, processor] = await Promise.all([
          AutoModelForObjectDetection.from_pretrained(modelId, { device, dtype }),
          AutoProcessor.from_pretrained(modelId),
        ]);
      } catch {
        [model, processor] = await Promise.all([
          AutoModelForObjectDetection.from_pretrained(modelId, { device: "wasm", dtype: "q8" }),
          AutoProcessor.from_pretrained(modelId),
        ]);
      }
      return new ObjectDetectionPipeline({ task: "object-detection", model, processor });
    })().catch((error) => {
      detectorPromise = null;
      throw error;
    });
  }
  return detectorPromise;
}

async function makeInferenceUrl(imageSource) {
  const blob = imageSource instanceof Blob
    ? imageSource
    : await (await fetch(imageSource)).blob();
  const bitmap = await createImageBitmap(blob);
  try {
    const maxDim = 640;
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
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
      const area = width * height;
      return INCIDENTAL_PROPS.has(item.label) ? area >= 0.15 : area >= 0.05;
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
    const result = await detect(data.imageSource, data.threshold);
    self.postMessage({ id: data.id, result });
  } catch (error) {
    self.postMessage({ id: data.id, error: error?.message || "Vision detection failed." });
  }
};
