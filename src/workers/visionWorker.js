// src/workers/visionWorker.js
// 100% Offline Vector Embedding Engine via CLIP (Xenova/clip-vit-base-patch32)
// Generates unified 512-dimensional multimodal embeddings for zero-tag visual search.

let clipPromise = null;

const LOW_MEMORY_DEVICE =
  typeof navigator !== "undefined" &&
  Number.isFinite(navigator.deviceMemory) &&
  navigator.deviceMemory <= 4;

function publishModelStatus(status) {
  self.postMessage({ type: "MODEL_STATUS", ...status });
}

function normalizeVector(data) {
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    sum += data[i] * data[i];
  }
  const norm = Math.sqrt(sum) || 1e-12;
  const normalized = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    normalized[i] = data[i] / norm;
  }
  return Array.from(normalized);
}

async function getClipModels() {
  if (!clipPromise) {
    publishModelStatus({ state: "loading", progress: null });
    clipPromise = (async () => {
      const {
        AutoTokenizer,
        AutoProcessor,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
        env,
      } = await import("@huggingface/transformers");

      let isCacheAvailable = false;
      try {
        isCacheAvailable = typeof caches !== "undefined" && typeof caches.open === "function";
      } catch {
        isCacheAvailable = false;
      }
      env.useBrowserCache = isCacheAvailable;
      env.useWasmCache = isCacheAvailable;
      env.allowLocalModels = false;
      env.allowRemoteModels = true;

      const modelId = "Xenova/clip-vit-base-patch32";
      const progress_callback = (progress) => {
        if (progress.status === "progress_total") {
          publishModelStatus({
            state: "loading",
            progress: Math.round(progress.progress),
          });
        }
      };

      const [tokenizer, processor, textModel, visionModel] = await Promise.all([
        AutoTokenizer.from_pretrained(modelId),
        AutoProcessor.from_pretrained(modelId),
        CLIPTextModelWithProjection.from_pretrained(modelId, {
          device: "wasm",
          dtype: "q8",
          progress_callback,
        }),
        CLIPVisionModelWithProjection.from_pretrained(modelId, {
          device: "wasm",
          dtype: "q8",
        }),
      ]);

      publishModelStatus({ state: "ready", progress: 100 });
      return { tokenizer, processor, textModel, visionModel };
    })().catch((error) => {
      clipPromise = null;
      publishModelStatus({ state: "error", error: error?.message || String(error) });
      throw error;
    });
  }
  return clipPromise;
}

async function toBlob(imageSource) {
  return imageSource instanceof Blob ? imageSource : (await fetch(imageSource)).blob();
}

async function embedImage(imageSource) {
  const { processor, visionModel } = await getClipModels();
  const { load_image } = await import("@huggingface/transformers");
  const blob = await toBlob(imageSource);
  const url = URL.createObjectURL(blob);
  try {
    const img = await load_image(url);
    const inputs = await processor(img);
    const { image_embeds } = await visionModel(inputs);
    return normalizeVector(image_embeds.data);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function embedText(text) {
  const clean = (text || "").trim();
  if (!clean) return null;
  const { tokenizer, textModel } = await getClipModels();
  const inputs = tokenizer([clean], { padding: true, truncation: true });
  const { text_embeds } = await textModel(inputs);
  return normalizeVector(text_embeds.data);
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

self.onmessage = async ({ data }) => {
  try {
    let result;
    if (data.type === "warmup") {
      await getClipModels();
      result = { warmed: true };
    } else if (data.type === "embedImage") {
      result = await embedImage(data.imageSource);
    } else if (data.type === "embedText") {
      result = await embedText(data.text);
    } else if (data.type === "thumbnail") {
      result = await createThumbnail(data.imageSource);
    } else {
      result = null;
    }
    self.postMessage({ id: data.id, result });
  } catch (error) {
    self.postMessage({ id: data.id, error: error?.message || "Vision task failed." });
  }
};
