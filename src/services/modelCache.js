// src/services/modelCache.js
// Utility to pre‑download and cache model assets for Moonshine (STT) and CLIP (vision).
// Uses the standard Cache Storage API matching the exact repository file paths.

/** Helper to fetch a URL and store the response in the named cache. */
async function cacheUrl(cacheName, url) {
  if (typeof caches === "undefined") return;
  try {
    const cache = await caches.open(cacheName);
    const match = await cache.match(url);
    if (match) return; // already cached
    const resp = await fetch(url, { cache: "force-cache" });
    if (!resp.ok) {
      console.warn(`[modelCache] Optional asset skip (${resp.status}): ${url}`);
      return;
    }
    await cache.put(url, resp.clone());
  } catch (err) {
    console.warn(`[modelCache] Asset cache warning:`, err?.message || err);
  }
}

/** Ensure Moonshine model files are cached locally.
 *  onnx-community/moonshine-tiny-ONNX stores quantized encoder and merged decoder models under onnx/.
 */
export async function ensureMoonshineModelCached() {
  if (typeof window === "undefined" || typeof caches === "undefined") return;
  const CACHE = "moonshine-model";
  const base = "https://huggingface.co/onnx-community/moonshine-tiny-ONNX/resolve/main";
  const files = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "onnx/encoder_model_quantized.onnx",
    "onnx/decoder_model_merged_quantized.onnx",
  ];
  await Promise.all(files.map((f) => cacheUrl(CACHE, `${base}/${f}`)));
}

/** Ensure CLIP model files are cached locally.
 *  Xenova/clip-vit-base-patch32 stores separate text and vision projection models under onnx/.
 */
export async function ensureClipModelCached() {
  if (typeof window === "undefined" || typeof caches === "undefined") return;
  const CACHE = "clip-model";
  const base = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main";
  const files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "onnx/text_model_quantized.onnx",
    "onnx/vision_model_quantized.onnx",
  ];
  await Promise.all(files.map((f) => cacheUrl(CACHE, `${base}/${f}`)));
}
