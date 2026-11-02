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

// Curated dictionary of everyday real-world visual concepts for automatic offline tag extraction
const VISUAL_CATEGORIES = [
  // Drinks & Food
  { prompt: "a photo of a cup of coffee, tea, or mug", tags: ["#cup", "#tea", "#coffee"] },
  { prompt: "a photo of a water bottle or drink", tags: ["#bottle", "#drink"] },
  { prompt: "a photo of food, meal, or cooked dish on a plate", tags: ["#food", "#meal"] },
  { prompt: "a photo of snacks, cake, or bakery dessert", tags: ["#food", "#snack"] },
  { prompt: "a photo of fresh fruits or vegetables", tags: ["#fruit", "#food"] },

  // Tech & Desk
  { prompt: "a photo of a computer mouse or trackpad", tags: ["#mouse", "#tech"] },
  { prompt: "a photo of a laptop computer", tags: ["#laptop", "#tech"] },
  { prompt: "a photo of a computer keyboard", tags: ["#keyboard", "#tech"] },
  { prompt: "a photo of a smartphone or mobile phone", tags: ["#phone", "#mobile"] },
  { prompt: "a photo of a computer monitor or television screen", tags: ["#screen", "#tech"] },
  { prompt: "a photo of headphones or earphones", tags: ["#headphones", "#audio"] },
  { prompt: "a photo of a wristwatch or smartwatch", tags: ["#watch"] },

  // Vehicles
  { prompt: "a photo of a motorcycle, scooter, or motorbike", tags: ["#motorcycle", "#vehicle"] },
  { prompt: "a photo of a car or automobile", tags: ["#car", "#vehicle"] },
  { prompt: "a photo of a vehicle license plate", tags: ["#vehicle", "#licenseplate"] },
  { prompt: "a photo of a bicycle", tags: ["#bicycle"] },

  // Personal items & apparel
  { prompt: "a photo of shoes or sneakers", tags: ["#shoes"] },
  { prompt: "a photo of a backpack, shoulder bag, or handbag", tags: ["#bag"] },
  { prompt: "a photo of eyeglasses or sunglasses", tags: ["#glasses"] },
  { prompt: "a photo of house keys or car keys", tags: ["#keys"] },
  { prompt: "a photo of a wallet or purse", tags: ["#wallet"] },
  { prompt: "a photo of clothes, shirt, jacket, or apparel", tags: ["#clothing"] },
  { prompt: "a photo of a book, notebook, or diary", tags: ["#book", "#notes"] },

  // Paper & Documents
  { prompt: "a photo of a cash receipt or store invoice bill", tags: ["#receipt", "#bill"] },
  { prompt: "a photo of an official printed paper document or certificate", tags: ["#document"] },

  // People & Nature
  { prompt: "a photo of a human hand holding an item", tags: ["#hand"] },
  { prompt: "a photo of a portrait of a person", tags: ["#person"] },
  { prompt: "a photo of a pet dog or puppy", tags: ["#dog", "#pet"] },
  { prompt: "a photo of a pet cat or kitten", tags: ["#cat", "#pet"] },
  { prompt: "a photo of green plants or flowers", tags: ["#plants", "#nature"] },
  { prompt: "a photo of outdoor landscape, trees, or scenery", tags: ["#scenery", "#nature"] },
];

let candidatePromptEmbeddings = null;

async function getCandidateEmbeddings(tokenizer, textModel) {
  if (candidatePromptEmbeddings) return candidatePromptEmbeddings;
  const prompts = VISUAL_CATEGORIES.map((c) => c.prompt);
  const inputs = tokenizer(prompts, { padding: true, truncation: true });
  const { text_embeds } = await textModel(inputs);
  const dim = 512;
  const data = text_embeds.data;
  const rows = [];
  for (let i = 0; i < VISUAL_CATEGORIES.length; i++) {
    const offset = i * dim;
    const row = new Float32Array(dim);
    let sum = 0;
    for (let j = 0; j < dim; j++) {
      const v = data[offset + j];
      row[j] = v;
      sum += v * v;
    }
    const norm = Math.sqrt(sum) || 1e-12;
    for (let j = 0; j < dim; j++) {
      row[j] /= norm;
    }
    rows.push(row);
  }
  candidatePromptEmbeddings = rows;
  return candidatePromptEmbeddings;
}

function extractVisualTags(imageEmbedding, candidateEmbeddings, threshold = 0.22) {
  if (!imageEmbedding || !candidateEmbeddings || candidateEmbeddings.length === 0) return [];
  const scored = [];
  for (let i = 0; i < VISUAL_CATEGORIES.length; i++) {
    const catEmbed = candidateEmbeddings[i];
    let dot = 0;
    for (let j = 0; j < 512; j++) {
      dot += imageEmbedding[j] * catEmbed[j];
    }
    scored.push({ index: i, tags: VISUAL_CATEGORIES[i].tags, score: dot });
  }

  scored.sort((a, b) => b.score - a.score);

  if (scored.length === 0 || scored[0].score < threshold) {
    return [];
  }

  const topScore = scored[0].score;
  const tags = new Set();

  for (const item of scored) {
    if (item.score >= threshold && item.score >= topScore - 0.05) {
      for (const t of item.tags) {
        if (tags.size < 4) tags.add(t);
      }
    }
    if (tags.size >= 3) break;
  }

  return Array.from(tags);
}

async function toBlob(imageSource) {
  return imageSource instanceof Blob ? imageSource : (await fetch(imageSource)).blob();
}

async function embedImage(imageSource) {
  const { processor, visionModel, tokenizer, textModel } = await getClipModels();
  const { load_image } = await import("@huggingface/transformers");
  const blob = await toBlob(imageSource);
  const url = URL.createObjectURL(blob);
  try {
    const img = await load_image(url);
    const inputs = await processor(img);
    const { image_embeds } = await visionModel(inputs);
    const embedding = normalizeVector(image_embeds.data);
    let visualTags = [];
    try {
      const candidates = await getCandidateEmbeddings(tokenizer, textModel);
      visualTags = extractVisualTags(embedding, candidates);
    } catch (tagErr) {
      console.warn("[VisionWorker] Visual tag extraction skipped:", tagErr);
    }
    return { embedding, visualTags };
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
      const { tokenizer, textModel } = await getClipModels();
      await getCandidateEmbeddings(tokenizer, textModel);
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
