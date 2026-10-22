// src/services/visionService.js
// In-browser Vision Object Detection pipeline via Transformers.js
let detector = null;
let isLoading = false;
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
  person: ["#photo", "#person", "#selfie", "#people"],
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
    isLoading = true;
    try {
      // Keep the large Transformers.js bundle out of the app's initial load.
      const {
        AutoModelForObjectDetection,
        AutoProcessor,
        ObjectDetectionPipeline,
        env,
      } = await import("@huggingface/transformers");

      // Use browser cache (Cache API / IndexedDB) and avoid local file system attempts.
      env.allowLocalModels = false;
      env.useBrowserCache = true;

      const modelId = "Xenova/yolos-tiny";
      const progress_callback = (p) => {
        if (typeof onProgress === "function") {
          onProgress(p);
        }
      };
      const [model, processor] = await Promise.all([
        AutoModelForObjectDetection.from_pretrained(modelId, {
          device: "wasm",
          progress_callback,
        }),
        AutoProcessor.from_pretrained(modelId, { progress_callback }),
      ]);
      detector = new ObjectDetectionPipeline({ task: "object-detection", model, processor });
      return detector;
    } catch (err) {
      console.error("Failed to load YOLO pipeline:", err);
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
 * @param {number} threshold - Minimum confidence score (0.0 to 1.0, default: 0.28 for balanced precision/recall)
 * @returns {Promise<Array<{ label: string, score: number, box: object }>>}
 */
export async function detectObjects(imageSource, threshold = 0.40) {
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
    console.error("Vision detection error:", error);
    return [];
  }
}

/**
 * Detects objects and maps them to hyperlocal, real-world everyday tags.
 * 
 * @param {string|Blob} imageSource
 * @param {number} threshold
 * @returns {Promise<{ labels: string[], tags: string[], detections: Array }>}
 */
export async function getDetectedLabelsAndTags(imageSource, threshold = 0.40) {
  const detections = await detectObjects(imageSource, threshold);
  const uniqueLabels = [...new Set(detections.map((d) => d.label))];
  
  const derivedTags = new Set();
  uniqueLabels.forEach((label) => {
    // Add raw label as tag
    derivedTags.add(`#${label.replace(/\s+/g, "_")}`);
    
    // Add hyperlocal synonyms and category tags
    const mapped = HYPERLOCAL_COCO_TAGS[label];
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
