import { pipeline, env } from "@huggingface/transformers";

// Configure Transformers.js for browser offline cache
env.allowLocalModels = false;
env.useBrowserCache = true;

let ocrPipeline = null;
let initPromise = null;
let isModelLoading = false;

/**
 * Initializes and caches the deep-learning Vision Transformer OCR model.
 * Model: Xenova/trocr-small-printed (~42MB quantized)
 * Robust against hatched/textured security fonts, scene text, noise, and tilted graphics.
 */
export async function getDeepLearningOCR(onProgress) {
  if (ocrPipeline) return ocrPipeline;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    isModelLoading = true;
    try {
      const progress_callback = (p) => {
        if (typeof onProgress === "function") {
          onProgress(p);
        }
      };

      const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
      const device = hasWebGPU ? "webgpu" : "wasm";
      const dtype = hasWebGPU ? "fp32" : "q8";

      try {
        ocrPipeline = await pipeline("image-to-text", "Xenova/trocr-small-printed", {
          device,
          dtype,
          progress_callback,
        });
      } catch (err) {
        if (device === "webgpu") {
          console.warn("[DeepLearningOCR] WebGPU failed, falling back to WASM q8:", err);
          ocrPipeline = await pipeline("image-to-text", "Xenova/trocr-small-printed", {
            device: "wasm",
            dtype: "q8",
            progress_callback,
          });
        } else {
          throw err;
        }
      }

      return ocrPipeline;
    } catch (err) {
      console.error("[DeepLearningOCR] Pipeline initialization failed:", err);
      initPromise = null;
      throw err;
    } finally {
      isModelLoading = false;
    }
  })();

  return initPromise;
}

/**
 * Normalizes vehicle registration plates (Indian & international standard alphanumeric)
 * e.g., "BR 01 C J 6440" -> "BR01C J6440" or "BR01CJ6440"
 */
export function normalizeRegistrationPlate(text) {
  if (!text) return null;
  const compact = text.toUpperCase().replace(/[^A-Z0-9]/g, "");
  
  // Standard format: 2 State letters + 1-2 District digits + 0-3 Series letters + 3-4 Digits
  // e.g., BR01CJ6440, DL3CAF1234, MH12DE1433
  const plateMatch = compact.match(/([A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4})/);
  if (plateMatch) {
    return plateMatch[1];
  }
  return null;
}

/**
 * Pre-warms the Deep Learning OCR model on app idle.
 */
export async function prewarmDeepLearningOCR() {
  try {
    await getDeepLearningOCR();
    console.log("[DeepLearningOCR] Model pre-warmed successfully.");
  } catch (err) {
    console.debug("[DeepLearningOCR] Pre-warm idle fail:", err);
  }
}
