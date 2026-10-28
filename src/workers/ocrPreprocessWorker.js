import { detectTextRegions } from "../services/ocrRegionDetection.js";

let openCvLoader;
async function getOpenCvVariants(imageData) {
  if (!openCvLoader) {
    openCvLoader = import("../services/openCvScanner.js")
      .then(({ createOpenCvScanVariants }) => createOpenCvScanVariants(imageData));
  } else {
    const { createOpenCvScanVariants } = await import("../services/openCvScanner.js");
    return createOpenCvScanVariants(imageData);
  }
  try {
    return await openCvLoader;
  } finally {
    openCvLoader = null;
  }
}

async function enhance(image) {
  const lowMemoryDevice = Number.isFinite(navigator.deviceMemory) && navigator.deviceMemory <= 4;
  const maxDimension = lowMemoryDevice ? 1100 : 1600;
  let bitmap;
  try {
    bitmap = await createImageBitmap(image, {
      resizeWidth: maxDimension,
      resizeQuality: "high",
    });
  } catch {
    bitmap = await createImageBitmap(image);
  }
  try {
    const sourceMax = Math.max(bitmap.width, bitmap.height);
    const scale = Math.min(1, maxDimension / sourceMax);
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Could not create OCR enhancement canvas.");
    context.drawImage(bitmap, 0, 0, width, height);
    const baseline = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.94 });

    const pixels = context.getImageData(0, 0, width, height);
    let openCvVariants = { variants: [], perspectiveVariants: [] };
    let openCvAvailable = false;
    let openCvError = null;
    try {
      openCvVariants = await getOpenCvVariants(pixels);
      openCvAvailable = true;
    } catch (error) {
      openCvError = error?.message || "OpenCV could not be initialized.";
      console.warn("[OCR] OpenCV scan correction unavailable; using lightweight enhancement:", error);
    }
    const luminance = new Uint8Array(width * height);
    for (let pixel = 0; pixel < luminance.length; pixel++) {
      const offset = pixel * 4;
      const gray = Math.round(
        pixels.data[offset] * 0.299 +
        pixels.data[offset + 1] * 0.587 +
        pixels.data[offset + 2] * 0.114
      );
      luminance[pixel] = gray;
    }
    const tileSize = 64;
    const regions = detectTextRegions(luminance, width, height);
    const localMeans = calculateLocalMeans(luminance, width, height, tileSize);
    const columns = Math.ceil(width / tileSize);
    const rows = Math.ceil(height / tileSize);

    const enhanced = context.createImageData(width, height);
    const binary = context.createImageData(width, height);
    for (let pixel = 0; pixel < luminance.length; pixel++) {
      const offset = pixel * 4;
      const value = luminance[pixel];
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      const localMean = interpolateLocalMean(
        x, y, tileSize, localMeans, columns, rows
      );
      // Increase local legibility while retaining grayscale anti-aliasing.
      const contrast = Math.max(0, Math.min(255, Math.round((value - 128) * 1.35 + 128)));
      enhanced.data[offset] = contrast;
      enhanced.data[offset + 1] = contrast;
      enhanced.data[offset + 2] = contrast;
      enhanced.data[offset + 3] = 255;

      const binaryValue = value < localMean - 12 ? 0 : 255;
      binary.data[offset] = binaryValue;
      binary.data[offset + 1] = binaryValue;
      binary.data[offset + 2] = binaryValue;
      binary.data[offset + 3] = 255;
    }

    context.putImageData(enhanced, 0, 0);
    const contrastBlob = await canvas.convertToBlob({ type: "image/png" });
    const binaryCanvas = new OffscreenCanvas(width, height);
    const binaryContext = binaryCanvas.getContext("2d");
    if (!binaryContext) throw new Error("Could not create OCR threshold canvas.");
    binaryContext.putImageData(binary, 0, 0);
    const binaryBlob = await binaryCanvas.convertToBlob({ type: "image/png" });

    const rotated = [];
    for (const angle of [90, 270]) {
      const rotatedCanvas = new OffscreenCanvas(height, width);
      const rotatedContext = rotatedCanvas.getContext("2d");
      if (!rotatedContext) continue;
      rotatedContext.translate(height / 2, width / 2);
      rotatedContext.rotate((angle * Math.PI) / 180);
      rotatedContext.drawImage(bitmap, -width / 2, -height / 2, width, height);
      rotated.push(await rotatedCanvas.convertToBlob({ type: "image/jpeg", quality: 0.92 }));
    }

    const focusedRegions = [];
    for (const region of regions) {
      for (const sourceCanvas of [canvas, binaryCanvas]) {
        const crop = new OffscreenCanvas(region.width, region.height);
        const cropContext = crop.getContext("2d");
        if (!cropContext) continue;
        cropContext.drawImage(
          sourceCanvas,
          region.x,
          region.y,
          region.width,
          region.height,
          0,
          0,
          region.width,
          region.height
        );
        focusedRegions.push(await crop.convertToBlob({ type: "image/png" }));
      }
    }

    const scannerFoundRegions = openCvVariants.perspectiveVariants.length > 0;
    return {
      baseline,
      scanner: {
        openCvAvailable,
        perspectiveCount: openCvVariants.perspectiveVariants.length,
        error: openCvError,
      },
      variants: openCvVariants.variants.length > 0
        ? openCvVariants.variants
        : [contrastBlob, binaryBlob],
      perspectiveVariants: openCvVariants.perspectiveVariants.slice(0, 3),
      focusedRegions: scannerFoundRegions ? [] : focusedRegions.slice(0, 2),
      rotated,
    };
  } finally {
    bitmap.close();
  }
}

function calculateLocalMeans(luminance, width, height, tileSize) {
  const columns = Math.ceil(width / tileSize);
  const rows = Math.ceil(height / tileSize);
  const means = new Float32Array(columns * rows);
  for (let tileY = 0; tileY < rows; tileY++) {
    for (let tileX = 0; tileX < columns; tileX++) {
      const xEnd = Math.min(width, (tileX + 1) * tileSize);
      const yEnd = Math.min(height, (tileY + 1) * tileSize);
      let sum = 0;
      let count = 0;
      for (let y = tileY * tileSize; y < yEnd; y++) {
        for (let x = tileX * tileSize; x < xEnd; x++) {
          sum += luminance[y * width + x];
          count++;
        }
      }
      means[tileY * columns + tileX] = sum / count;
    }
  }
  return means;
}

function interpolateLocalMean(x, y, tileSize, means, columns, rows) {
  const gridX = Math.max(0, Math.min(columns - 1, x / tileSize - 0.5));
  const gridY = Math.max(0, Math.min(rows - 1, y / tileSize - 0.5));
  const left = Math.floor(gridX);
  const top = Math.floor(gridY);
  const right = Math.min(columns - 1, left + 1);
  const bottom = Math.min(rows - 1, top + 1);
  const horizontal = gridX - left;
  const vertical = gridY - top;
  const topMean = means[top * columns + left] * (1 - horizontal) +
    means[top * columns + right] * horizontal;
  const bottomMean = means[bottom * columns + left] * (1 - horizontal) +
    means[bottom * columns + right] * horizontal;
  return topMean * (1 - vertical) + bottomMean * vertical;
}

self.onmessage = async ({ data }) => {
  try {
    const result = await enhance(data.image);
    self.postMessage({ id: data.id, ...result });
  } catch (error) {
    self.postMessage({
      id: data.id,
      error: error?.message || "OCR image enhancement failed.",
    });
  }
};
