// src/services/visionService.js
// In-browser YOLOv8-Nano object detection pipeline via Transformers.js
import { pipeline, env } from "@huggingface/transformers";

// Use browser cache (Cache API / IndexedDB) and avoid local file system attempts
env.allowLocalModels = false;
env.useBrowserCache = true;

let detector = null;
let isLoading = false;
let initPromise = null;

// COCO category mapping to app canonical tags for semantic retrieval
const COCO_TO_APP_TAGS = {
  // Vehicles
  car: ["#vehicle", "#car"],
  motorcycle: ["#vehicle", "#bike"],
  bicycle: ["#vehicle", "#cycle"],
  bus: ["#vehicle", "#bus"],
  truck: ["#vehicle", "#truck"],
  airplane: ["#travel", "#flight"],
  boat: ["#travel", "#boat"],

  // People & Accessories
  person: ["#photo", "#person"],
  backpack: ["#bag", "#accessory"],
  handbag: ["#bag", "#pouch"],
  suitcase: ["#travel", "#bag"],

  // Common Objects & Household
  bottle: ["#item"],
  cup: ["#item"],
  "potted plant": ["#plant", "#nature"],
  plant: ["#plant", "#nature"],
  chair: ["#furniture"],
  couch: ["#furniture"],
  bed: ["#furniture"],
  laptop: ["#electronics", "#tech"],
  "cell phone": ["#electronics", "#phone"],
  book: ["#document", "#book"],
  clock: ["#time"],
  cat: ["#pet", "#animal"],
  dog: ["#pet", "#animal"],
};

/**
 * Initializes and caches the YOLOv8-Nano object detection pipeline.
 * Xenova/yolov8n runs in browser via ONNX Runtime Web (WASM / WebGPU).
 */
export async function getDetector(onProgress) {
  if (detector) return detector;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    isLoading = true;
    try {
      detector = await pipeline("object-detection", "Xenova/yolov8n", {
        device: "wasm",
        progress_callback: (p) => {
          if (typeof onProgress === "function") {
            onProgress(p);
          }
        },
      });
      return detector;
    } catch (err) {
      console.error("Failed to load YOLOv8-nano pipeline:", err);
      initPromise = null;
      throw err;
    } finally {
      isLoading = false;
    }
  })();

  return initPromise;
}

/**
 * Detects objects in an image URL, base64 data URL, or Blob.
 * 
 * @param {string|Blob} imageSource - Image base64 data URL or Blob
 * @param {number} threshold - Minimum confidence score (0.0 to 1.0, default: 0.35)
 * @returns {Promise<Array<{ label: string, score: number, box: object }>>}
 */
export async function detectObjects(imageSource, threshold = 0.35) {
  try {
    const pipe = await getDetector();
    let src = imageSource;
    let tempUrl = null;

    if (imageSource instanceof Blob) {
      tempUrl = URL.createObjectURL(imageSource);
      src = tempUrl;
    }

    const rawDetections = await pipe(src, {
      threshold,
      percentage: true, // returns normalized 0..1 bounding box coordinates
    });

    if (tempUrl) {
      URL.revokeObjectURL(tempUrl);
    }

    return (rawDetections || []).map((det) => ({
      label: det.label.toLowerCase(),
      score: Math.round(det.score * 100) / 100,
      box: det.box,
    }));
  } catch (error) {
    console.error("YOLO detection error:", error);
    return [];
  }
}

/**
 * Detects objects and maps them to app tags for indexing and search.
 * 
 * @param {string|Blob} imageSource
 * @param {number} threshold
 * @returns {Promise<{ labels: string[], tags: string[], detections: Array }>}
 */
export async function getDetectedLabelsAndTags(imageSource, threshold = 0.35) {
  const detections = await detectObjects(imageSource, threshold);
  const uniqueLabels = [...new Set(detections.map((d) => d.label))];
  
  const derivedTags = new Set();
  uniqueLabels.forEach((label) => {
    // Add raw label as tag
    derivedTags.add(`#${label.replace(/\s+/g, "_")}`);
    // Add semantic mapped tags
    const mapped = COCO_TO_APP_TAGS[label];
    if (mapped) {
      mapped.forEach((tag) => derivedTags.add(tag));
    }
  });

  return {
    labels: uniqueLabels,
    tags: Array.from(derivedTags),
    detections,
  };
}
