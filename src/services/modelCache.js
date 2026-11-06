// src/services/modelCache.js
// Utility to pre‑download and cache model assets for Moonshine (STT) and CLIP (vision)
// Uses the standard Cache Storage API. Files are stored under distinct cache names.

/** Helper to fetch a URL and store the response in the named cache. */
async function cacheUrl(cacheName, url) {
  const cache = await caches.open(cacheName);
  const match = await cache.match(url);
  if (match) return; // already cached
  const resp = await fetch(url, { cache: "force-cache" });
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  await cache.put(url, resp.clone());
}

/** Ensure Moonshine model files are cached locally.
 *  Returns a promise that resolves when all files are cached.
 */
export async function ensureMoonshineModelCached() {
  const CACHE = "moonshine-model";
  const base = "https://huggingface.co/onnx-community/moonshine-tiny-ONNX/resolve/main";
  const files = [
    "model.onnx",
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
  ];
  await Promise.all(files.map((f) => cacheUrl(CACHE, `${base}/${f}`)));
}

/** Ensure CLIP model files are cached locally.
 *  The Xenova/clip-vit-base-patch32 repo contains several assets.
 */
export async function ensureClipModelCached() {
  const CACHE = "clip-model";
  const base = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main";
  const files = [
    "model.onnx",
    "config.json",
    "tokenizer.json",
    "vocab.txt",
    "merges.txt",
    "preprocessor_config.json",
  ];
  await Promise.all(files.map((f) => cacheUrl(CACHE, `${base}/${f}`)));
}
