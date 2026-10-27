// src/workers/cropWorker.js
// Receives an ImageBitmap and a plate bounding box, returns a JPEG Blob of a
// perspective‑corrected, up‑scaled plate image. All heavy canvas work stays in
// this worker, keeping the UI thread free.
self.onmessage = async ({ data }) => {
  const { bitmap, box } = data; // box: { x, y, width, height, angle }
  const { x, y, width, height, angle } = box;

  // 1. Create a temporary canvas large enough to hold the rotated rectangle.
  const maxDim = Math.max(width, height);
  const tempCanvas = new OffscreenCanvas(maxDim, maxDim);
  const ctx = tempCanvas.getContext("2d");
  if (!ctx) {
    self.postMessage({ error: "No 2d context" });
    return;
  }

  // Center the source, rotate so the plate becomes axis‑aligned.
  ctx.translate(maxDim / 2, maxDim / 2);
  ctx.rotate(-angle); // negative to undo the angle from detection
  ctx.drawImage(
    bitmap,
    -width / 2 - x,
    -height / 2 - y,
    bitmap.width,
    bitmap.height
  );

  // 2. Crop the axis‑aligned rectangle.
  const cropCanvas = new OffscreenCanvas(width, height);
  const cctx = cropCanvas.getContext("2d");
  cctx.drawImage(
    tempCanvas,
    (maxDim - width) / 2,
    (maxDim - height) / 2,
    width,
    height,
    0,
    0,
    width,
    height
  );

  // 3. Upscale – a larger image gives Tesseract more pixels to work with.
  // Limit upscale to keep final dimensions reasonable for Tesseract (≈1200px).
  const upscale = Math.min(1200 / Math.max(width, height), 2);
  const finalW = Math.round(width * upscale);
  const finalH = Math.round(height * upscale);
  const finalCanvas = new OffscreenCanvas(finalW, finalH);
  finalCanvas.getContext("2d").drawImage(cropCanvas, 0, 0, finalW, finalH);

  // 4. Convert to JPEG Blob (high quality).
  const blob = await finalCanvas.convertToBlob({ type: "image/jpeg", quality: 0.92 });
  self.postMessage({ blob }, [blob]);
};