// src/services/visionService.js
// In-browser Vision Object Detection pipeline via Transformers.js
let detector = null;
let initPromise = null;
let detectorIdleTimer = null;

function resetDetectorTimer() {
  if (detectorIdleTimer) clearTimeout(detectorIdleTimer);
  detectorIdleTimer = setTimeout(() => {
    if (detector) {
      detector = null;
      initPromise = null;
      console.log("[visionService] YOLOS detector released after idle timeout to reclaim RAM.");
    }
  }, 30000);
}

/**
 * HYPERLOCAL & REAL-WORLD COCO-80 MAPPING
 */
export const HYPERLOCAL_COCO_TAGS = {
  // ── 1. Vehicles & Mobility ──
  car: ["#vehicle", "#car", "#automobile", "#fourwheeler"],
  truck: ["#vehicle", "#car", "#truck", "#suv", "#fourwheeler"],
  bus: ["#vehicle", "#bus", "#transport"],
  motorcycle: ["#vehicle", "#bike", "#motorcycle", "#scooty", "#scooter", "#twowheeler"],
  bicycle: ["#vehicle", "#cycle", "#bicycle", "#pedalcycle", "#twowheeler"],
  airplane: ["#travel", "#flight", "#airplane", "#plane"],
  boat: ["#travel", "#boat", "#ship"],
  train: ["#travel", "#train", "#railway"],

  // ── 2. Personal Accessories & Bags ──
  backpack: ["#bag", "#backpack", "#schoolbag", "#officebag"],
  handbag: ["#bag", "#pouch", "#purse", "#handbag", "#clutch", "#wallet"],
  suitcase: ["#bag", "#travel", "#luggage", "#suitcase", "#trolley", "#briefcase"],
  umbrella: ["#accessory", "#umbrella", "#rain"],
  tie: ["#accessory", "#clothes", "#formal"],

  // ── 3. Gadgets & Tech ──
  laptop: ["#electronics", "#tech", "#laptop", "#computer", "#office", "#work"],
  "cell phone": ["#electronics", "#tech", "#phone", "#mobile", "#smartphone", "#cellphone"],
  tv: ["#electronics", "#tv", "#television", "#screen", "#monitor"],
  keyboard: ["#electronics", "#tech", "#keyboard", "#computer", "#typing"],
  mouse: ["#electronics", "#tech", "#mouse", "#computer"],
  remote: ["#electronics", "#remote", "#controller"],
  clock: ["#time", "#clock", "#watch"],

  // ── 4. Nature & Plants ──
  "potted plant": ["#plant", "#nature", "#flower", "#pot", "#garden", "#indoorplant"],
  plant: ["#plant", "#nature", "#garden", "#flower"],

  // ── 5. Home & Furniture ──
  chair: ["#furniture", "#chair", "#seat"],
  couch: ["#furniture", "#sofa", "#couch", "#livingroom"],
  bed: ["#furniture", "#bed", "#bedroom"],
  "dining table": ["#table"],
  toilet: ["#washroom", "#bathroom"],
  bottle: ["#bottle"],
  "wine glass": ["#kitchen", "#glass", "#drink"],
  cup: ["#cup", "#mug"],

  // ── 6. Documents & Books ──
  book: ["#document", "#book", "#reading", "#notes", "#diary", "#paper"],
  scissors: ["#stationery", "#tool", "#scissors"],

  // ── 7. People & Pets ──
  person: ["#photo", "#person", "#selfie", "#people"],
  cat: ["#pet", "#animal", "#cat", "#kitten"],
  dog: ["#pet", "#animal", "#dog", "#puppy"],
  bird: ["#animal", "#bird", "#nature"],
  horse: ["#animal", "#horse"],
  cow: ["#animal", "#cow"],
  elephant: ["#animal", "#elephant"],

  // ── 8. Sports & Recreation ──
  "sports ball": ["#sports", "#ball", "#cricket", "#football"],
  skateboard: ["#sports", "#skateboard"],
  "tennis racket": ["#sports", "#racket", "#badminton", "#tennis"],

  // ── 9. Food & Groceries ──
  banana: ["#food", "#fruit", "#grocery", "#banana"],
  apple: ["#food", "#fruit", "#grocery", "#apple"],
  pizza: ["#food", "#meal", "#pizza", "#dinner"],
  cake: ["#cake"],

  // ── 10. Posters & Graphics ──
  poster: ["#poster", "#graphic"],
  banner: ["#poster", "#banner"],
};

export async function getDetector(onProgress) {
  resetDetectorTimer();
  if (detector) return detector;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      const {
        AutoModelForObjectDetection,
        AutoProcessor,
        ObjectDetectionPipeline,
        env,
      } = await import("@huggingface/transformers");

      env.allowLocalModels = false;
      env.useBrowserCache = true;

      const modelId = "Xenova/yolos-tiny";
      const progress_callback = (p) => {
        if (typeof onProgress === "function") onProgress(p);
      };

      const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
      const device = hasWebGPU ? "webgpu" : "wasm";
      const dtype = hasWebGPU ? "fp32" : "q8";

      let model, processor;
      try {
        [model, processor] = await Promise.all([
          AutoModelForObjectDetection.from_pretrained(modelId, {
            device,
            dtype,
            progress_callback,
          }),
          AutoProcessor.from_pretrained(modelId, { progress_callback }),
        ]);
      } catch {
        [model, processor] = await Promise.all([
          AutoModelForObjectDetection.from_pretrained(modelId, {
            device: "wasm",
            dtype: "q8",
            progress_callback,
          }),
          AutoProcessor.from_pretrained(modelId, { progress_callback }),
        ]);
      }
      detector = new ObjectDetectionPipeline({ task: "object-detection", model, processor });
      return detector;
    } catch (err) {
      console.error("Failed to load YOLO pipeline:", err);
      initPromise = null;
      throw err;
    }
  })();
  return initPromise;
}

export function prewarmDetector() {
  // No-op on mount to save 120MB+ RAM until user uploads an image
}

export async function detectObjects(imageSource, threshold = 0.40) {
  try {
    const pipe = await getDetector();
    resetDetectorTimer();
    const optimizedSource = await resizeImageForInference(imageSource, 640);
    let src = optimizedSource;
    let tempUrl = null;
    try {
      if (optimizedSource instanceof Blob) {
        tempUrl = URL.createObjectURL(optimizedSource);
        src = tempUrl;
      }
      const rawDetections = await pipe(src, {
        threshold,
        percentage: true,
      });
      return (rawDetections || []).map((det) => ({
        label: det.label.toLowerCase(),
        score: Math.round(det.score * 100) / 100,
        box: det.box,
      }));
    } finally {
      if (tempUrl) URL.revokeObjectURL(tempUrl);
    }
  } catch (error) {
    console.error("Vision detection error:", error);
    return [];
  }
}

export async function resizeImageForInference(imageSource, maxDim = 640) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    let objectUrl = null;

    img.onload = () => {
      try {
        const { naturalWidth: width, naturalHeight: height } = img;
        if (width <= maxDim && height <= maxDim) {
          resolve(imageSource);
          return;
        }
        const scale = maxDim / Math.max(width, height);
        const newWidth = Math.max(1, Math.round(width * scale));
        const newHeight = Math.max(1, Math.round(height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = newWidth;
        canvas.height = newHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(imageSource);
          return;
        }
        ctx.drawImage(img, 0, 0, newWidth, newHeight);
        resolve(canvas.toDataURL("image/jpeg", 0.85));
        canvas.width = 0;
        canvas.height = 0;
      } catch {
        resolve(imageSource);
      } finally {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
      }
    };

    img.onerror = () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      resolve(imageSource);
    };

    if (imageSource instanceof Blob) {
      objectUrl = URL.createObjectURL(imageSource);
      img.src = objectUrl;
    } else {
      img.src = imageSource;
    }
  });
}

const INCIDENTAL_PROPS = new Set([
  "chair", "dining table", "cup", "bottle", "bowl", "fork", "knife", "spoon",
  "mouse", "keyboard", "remote", "bench", "vase", "potted plant", "fire hydrant", "traffic light"
]);

export async function getDetectedLabelsAndTags(imageSource, threshold = 0.40) {
  const detections = await detectObjects(imageSource, threshold);

  const salientDetections = detections.filter((d) => {
    const width = Math.max(0, (d.box?.xmax ?? 0) - (d.box?.xmin ?? 0));
    const height = Math.max(0, (d.box?.ymax ?? 0) - (d.box?.ymin ?? 0));
    const area = width * height;

    if (INCIDENTAL_PROPS.has(d.label)) {
      return area >= 0.15;
    }
    return area >= 0.05;
  });

  const uniqueLabels = [...new Set(salientDetections.map((d) => d.label))];
  const derivedTags = new Set();
  uniqueLabels.forEach((label) => {
    const mapped = HYPERLOCAL_COCO_TAGS[label];
    if (mapped) {
      mapped.forEach((tag) => derivedTags.add(tag));
    } else {
      derivedTags.add(`#${label.replace(/\s+/g, "_")}`);
    }
  });

  return {
    labels: uniqueLabels,
    tags: Array.from(derivedTags),
    detections: salientDetections,
  };
}
