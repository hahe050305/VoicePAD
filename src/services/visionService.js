// src/services/visionService.js
// In-browser Vision Object Detection pipeline via Transformers.js
let detector = null;
let initPromise = null;

/**
 * HYPERLOCAL & REAL-WORLD COCO-80 MAPPING
 * Bridges the gap between rigid COCO model outputs (e.g. "truck", "handbag", "potted plant")
 * and how everyday users think, speak, and search (hyperlocal, synonyms, colloquial terms).
 */
export const HYPERLOCAL_COCO_TAGS = {
  // ── 1. Vehicles & Mobility (Handles SUVs, Crossovers, Scooters, Autos, Cycles) ──
  car: ["#vehicle", "#car", "#automobile", "#fourwheeler"],
  truck: ["#vehicle", "#car", "#truck", "#suv", "#fourwheeler"], // Crossovers/SUVs often detect as truck
  bus: ["#vehicle", "#bus", "#transport"],
  motorcycle: ["#vehicle", "#bike", "#motorcycle", "#scooty", "#scooter", "#twowheeler"],
  bicycle: ["#vehicle", "#cycle", "#bicycle", "#pedalcycle", "#twowheeler"],
  airplane: ["#travel", "#flight", "#airplane", "#plane"],
  boat: ["#travel", "#boat", "#ship"],
  train: ["#travel", "#train", "#railway"],

  // ── 2. Personal Accessories & Bags (Pouches, Purses, Wallets, Luggage) ──────────
  backpack: ["#bag", "#backpack", "#schoolbag", "#officebag"],
  handbag: ["#bag", "#pouch", "#purse", "#handbag", "#clutch", "#wallet"],
  suitcase: ["#bag", "#travel", "#luggage", "#suitcase", "#trolley", "#briefcase"],
  umbrella: ["#accessory", "#umbrella", "#rain"],
  tie: ["#accessory", "#clothes", "#formal"],

  // ── 3. Gadgets & Office / Study Tech ───────────────────────────────────────────
  laptop: ["#electronics", "#tech", "#laptop", "#computer", "#office", "#work"],
  "cell phone": ["#electronics", "#tech", "#phone", "#mobile", "#smartphone", "#cellphone"],
  tv: ["#electronics", "#tv", "#television", "#screen", "#monitor"],
  keyboard: ["#electronics", "#tech", "#keyboard", "#computer", "#typing"],
  mouse: ["#electronics", "#tech", "#mouse", "#computer"],
  remote: ["#electronics", "#remote", "#controller"],
  clock: ["#time", "#clock", "#watch"],

  // ── 4. Nature, Houseplants & Garden ───────────────────────────────────────────
  "potted plant": ["#plant", "#nature", "#flower", "#pot", "#garden", "#indoorplant", "#potted_plant"],
  plant: ["#plant", "#nature", "#garden", "#flower"],

  // ── 5. Home, Furniture & Kitchen Essentials ────────────────────────────────────
  chair: ["#furniture", "#chair", "#seat"],
  couch: ["#furniture", "#sofa", "#couch", "#livingroom"],
  bed: ["#furniture", "#bed", "#bedroom"],
  "dining table": ["#table"],
  toilet: ["#washroom", "#bathroom"],
  bottle: ["#bottle"],
  "wine glass": ["#kitchen", "#glass", "#drink"],
  cup: ["#cup", "#mug"],
  fork: ["#kitchen", "#cutlery", "#fork"],
  knife: ["#kitchen", "#knife", "#cutlery"],
  spoon: ["#kitchen", "#spoon", "#cutlery"],
  bowl: ["#bowl"],
  sink: ["#kitchen", "#sink", "#washbasin"],
  refrigerator: ["#kitchen", "#fridge", "#refrigerator", "#appliance"],
  microwave: ["#kitchen", "#microwave", "#oven", "#appliance"],
  oven: ["#kitchen", "#oven", "#baking", "#appliance"],
  toaster: ["#kitchen", "#toaster", "#appliance"],

  // ── 6. Documents, Books & Stationeries ────────────────────────────────────────
  book: ["#document", "#book", "#reading", "#notes", "#diary", "#paper"],
  scissors: ["#stationery", "#tool", "#scissors"],

  // ── 7. People & Pets ──────────────────────────────────────────────────────────
  // A person in frame does not imply a selfie. The image already receives a
  // generic photo tag at upload; keep detection labels specific to the object.
  person: ["#person", "#people"],
  cat: ["#pet", "#animal", "#cat", "#kitten"],
  dog: ["#pet", "#animal", "#dog", "#puppy"],
  bird: ["#animal", "#bird", "#nature"],
  horse: ["#animal", "#horse"],
  sheep: ["#animal", "#sheep"],
  cow: ["#animal", "#cow"],
  elephant: ["#animal", "#elephant"],
  bear: ["#animal", "#bear"],
  zebra: ["#animal", "#zebra"],
  giraffe: ["#animal", "#giraffe"],

  // ── 8. Sports & Recreation ────────────────────────────────────────────────────
  frisbee: ["#sports", "#game"],
  skis: ["#sports", "#travel"],
  snowboard: ["#sports", "#travel"],
  "sports ball": ["#sports", "#ball", "#cricket", "#football", "#basketball"],
  kite: ["#sports", "#kite"],
  "baseball bat": ["#sports", "#bat"],
  "baseball glove": ["#sports", "#glove"],
  skateboard: ["#sports", "#skateboard"],
  surfboard: ["#sports", "#surfboard"],
  "tennis racket": ["#sports", "#racket", "#badminton", "#tennis"],

  // ── 9. Food & Groceries ───────────────────────────────────────────────────────
  banana: ["#food", "#fruit", "#grocery", "#banana"],
  apple: ["#food", "#fruit", "#grocery", "#apple"],
  sandwich: ["#food", "#snack", "#breakfast"],
  orange: ["#food", "#fruit", "#orange"],
  broccoli: ["#food", "#vegetable", "#grocery"],
  carrot: ["#food", "#vegetable", "#grocery"],
  "hot dog": ["#food", "#snack"],
  pizza: ["#food", "#meal", "#pizza", "#dinner"],
  donut: ["#food", "#sweet", "#snack"],
  cake: ["#cake"],

  // ── 10. Street & Urban ────────────────────────────────────────────────────────
  "traffic light": ["#traffic", "#signal", "#street"],
  "fire hydrant": ["#street", "#safety"],
  "stop sign": ["#traffic", "#sign"],
  "parking meter": ["#parking", "#vehicle"],
  bench: ["#park", "#outdoor", "#bench"],
  vase: ["#decor", "#vase", "#home"],
  "teddy bear": ["#toy", "#gift"],
  "hair drier": ["#grooming", "#appliance"],
  toothbrush: ["#grooming", "#hygiene", "#toothbrush"],

    // Education & Science Posters
  science: ["#science", "#education"],
  experiment: ["#science", "#diy"],
  experiments: ["#science", "#diy"],
  diy: ["#diy", "#creative"],
  stem: ["#science", "#education", "#stem"],
  poster: ["#poster", "#graphic"],
  banner: ["#poster", "#banner"],
  kids: ["#kids", "#education"],
};

/**
 * Initializes and caches the YOLOS-Tiny object detection pipeline.
 * Loads the model and image processor directly to avoid probing for tokenizer files.
 */
export async function getDetector(onProgress) {
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
      // 1. Prefer WebGPU for 10x hardware acceleration; fallback to quantized WASM (q8)
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
        // Fallback if browser/device WebGPU initialisation fails
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

/**
 * Detects objects in an image URL, base64 data URL, or Blob.
 * 
 * @param {string|Blob} imageSource - Image base64 data URL or Blob
 * @param {number} threshold - Minimum confidence score (0.0 to 1.0, default: 0.28 for balanced precision/recall)
 * @returns {Promise<Array<{ label: string, score: number, box: object }>>}
 */

export async function prewarmDetector() {
  try {
    await getDetector();
    console.log("[Vision] Pre-warmed YOLO pipeline.");
  } catch (err) {
    console.debug("[Vision] Pre-warm fail:", err);
  }
}

export async function detectObjects(imageSource, threshold = 0.40) {
  try {
    const pipe = await getDetector();
    // 2. Pre-scale image down to max 640px before inference
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
/**
 * Downscales image so Transformers.js doesn't freeze processing massive camera images.
 */
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

/**
 * Detects objects and maps them to hyperlocal, real-world everyday tags.
 * 
 * @param {string|Blob} imageSource
 * @param {number} threshold
 * @returns {Promise<{ labels: string[], tags: string[], detections: Array }>}
 */

const INCIDENTAL_PROPS = new Set([
  "chair", "dining table", "cup", "bottle", "bowl", "fork", "knife", "spoon",
  "mouse", "keyboard", "remote", "bench", "vase", "potted plant"
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
    // High-priority subjects (cars, bikes, persons, dogs, laptops) need at least 5% coverage
    return area >= 0.05;
  });

  // Keep labels and tags aligned with the size-filtered detections. Using the
  // raw detections here reintroduced tiny background guesses as full tags.
  const uniqueLabels = [...new Set(salientDetections.map((d) => d.label))];
  
  const derivedTags = new Set();
  uniqueLabels.forEach((label) => {
    // Add raw label as tag    
    // Add hyperlocal synonyms and category tags
    const mapped = HYPERLOCAL_COCO_TAGS[label];
    if (mapped) {
      mapped.forEach((tag) => derivedTags.add(tag));
    }

    else
    {
        derivedTags.add(`#${label.replace(/\s+/g, "_")}`);

    }
  });

  return {
    labels: uniqueLabels,
    tags: Array.from(derivedTags),
    detections: salientDetections,
  };
}
