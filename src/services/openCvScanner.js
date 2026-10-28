let openCvPromise;

async function getOpenCv() {
  if (!openCvPromise) {
    openCvPromise = (async () => {
      const imported = await import("@techstark/opencv-js");
      const module = imported.default || imported;
      if (module instanceof Promise) return module;
      if (module.Mat) return module;
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("OpenCV initialization timed out.")), 30000);
        module.onRuntimeInitialized = () => {
          clearTimeout(timeout);
          resolve();
        };
        module.onAbort = (reason) => {
          clearTimeout(timeout);
          reject(new Error(`OpenCV initialization failed: ${reason || "unknown error"}`));
        };
      });
      return module;
    })().catch((error) => {
      openCvPromise = null;
      throw error;
    });
  }
  return openCvPromise;
}

function orderCorners(points) {
  const sums = points.map((point) => point.x + point.y);
  const differences = points.map((point) => point.x - point.y);
  return [
    points[sums.indexOf(Math.min(...sums))],
    points[differences.indexOf(Math.max(...differences))],
    points[sums.indexOf(Math.max(...sums))],
    points[differences.indexOf(Math.min(...differences))],
  ];
}

function distance(first, second) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function findQuadrilaterals(cv, source, edges) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const candidates = [];
  try {
    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    for (let index = 0; index < contours.size(); index++) {
      const contour = contours.get(index);
      const approximation = new cv.Mat();
      try {
        const perimeter = cv.arcLength(contour, true);
        for (const epsilonScale of [0.018, 0.028, 0.04]) {
          cv.approxPolyDP(contour, approximation, epsilonScale * perimeter, true);
          if (approximation.rows !== 4 || !cv.isContourConvex(approximation)) continue;
          const points = [];
          for (let pointIndex = 0; pointIndex < 4; pointIndex++) {
            const offset = pointIndex * 2;
            points.push({
              x: approximation.data32S[offset],
              y: approximation.data32S[offset + 1],
            });
          }
          const corners = orderCorners(points);
          const area = Math.abs(cv.contourArea(approximation));
          const areaRatio = area / (source.cols * source.rows);
          if (areaRatio < 0.002 || areaRatio > 0.98) continue;

          const top = distance(corners[0], corners[1]);
          const bottom = distance(corners[3], corners[2]);
          const left = distance(corners[0], corners[3]);
          const right = distance(corners[1], corners[2]);
          const width = Math.max(top, bottom);
          const height = Math.max(left, right);
          if (width < 80 || height < 30) continue;
          const aspect = width / height;
          if (aspect < 0.35 || aspect > 8) continue;

          const boundsArea = width * height;
          const rectangularity = Math.min(1, area / boundsArea);
          if (rectangularity < 0.55) continue;
          const pageLike = areaRatio >= 0.1 && aspect >= 0.42 && aspect <= 2.25;
          const plateLike = areaRatio >= 0.002 && aspect >= 1.35 && aspect <= 7.5;
          if (!pageLike && !plateLike) continue;
          candidates.push({
            corners,
            areaRatio,
            aspect,
            score: areaRatio * rectangularity * (pageLike ? 1.1 : 1),
          });
          break;
        }
      } finally {
        approximation.delete();
        contour.delete();
      }
    }
  } finally {
    contours.delete();
    hierarchy.delete();
  }
  candidates.sort((first, second) => second.score - first.score);
  const selected = [];
  for (const candidate of candidates) {
    const overlaps = selected.some((existing) =>
      Math.abs(existing.areaRatio - candidate.areaRatio) < 0.01 &&
      Math.abs(existing.aspect - candidate.aspect) < 0.12
    );
    if (!overlaps) selected.push(candidate);
    if (selected.length >= 2) break;
  }
  return selected;
}

function matToBlob(cv, source, mimeType = "image/png") {
  const rgba = new cv.Mat();
  try {
    if (source.channels() === 4) {
      source.copyTo(rgba);
    } else {
      cv.cvtColor(source, rgba, cv.COLOR_GRAY2RGBA);
    }
    const pixels = new Uint8ClampedArray(rgba.data);
    const imageData = new ImageData(pixels, rgba.cols, rgba.rows);
    const canvas = new OffscreenCanvas(rgba.cols, rgba.rows);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Could not encode OpenCV scan output.");
    context.putImageData(imageData, 0, 0);
    return canvas.convertToBlob({ type: mimeType });
  } finally {
    rgba.delete();
  }
}

function createRectifiedVariants(cv, source, corners) {
  const [topLeft, topRight, bottomRight, bottomLeft] = corners;
  const width = Math.round(Math.max(
    distance(topLeft, topRight),
    distance(bottomLeft, bottomRight)
  ));
  const height = Math.round(Math.max(
    distance(topLeft, bottomLeft),
    distance(topRight, bottomRight)
  ));
  if (width < 80 || height < 30) return [];

  const srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flatMap(({ x, y }) => [x, y]));
  const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0, width - 1, 0, width - 1, height - 1, 0, height - 1,
  ]);
  const transform = cv.getPerspectiveTransform(srcPoints, dstPoints);
  const warped = new cv.Mat();
  const gray = new cv.Mat();
  const denoised = new cv.Mat();
  const claheOutput = new cv.Mat();
  const thresholded = new cv.Mat();
  const clahe = new cv.CLAHE(2.2, new cv.Size(8, 8));
  try {
    cv.warpPerspective(
      source,
      warped,
      transform,
      new cv.Size(width, height),
      cv.INTER_CUBIC,
      cv.BORDER_REPLICATE
    );
    cv.cvtColor(warped, gray, cv.COLOR_RGBA2GRAY);
    cv.medianBlur(gray, denoised, 3);
    clahe.apply(denoised, claheOutput);
    cv.adaptiveThreshold(
      claheOutput,
      thresholded,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY,
      31,
      11
    );
    return Promise.all([
      matToBlob(cv, warped),
      matToBlob(cv, claheOutput),
      matToBlob(cv, thresholded),
    ]);
  } finally {
    srcPoints.delete();
    dstPoints.delete();
    transform.delete();
    warped.delete();
    gray.delete();
    denoised.delete();
    claheOutput.delete();
    thresholded.delete();
    clahe.delete();
  }
}

export async function createOpenCvScanVariants(imageData) {
  const cv = await getOpenCv();
  const source = cv.matFromImageData(imageData);
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
  const closed = new cv.Mat();
  try {
    cv.cvtColor(source, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, 55, 165);
    cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel);
    const candidates = findQuadrilaterals(cv, source, closed);
    const perspectiveVariants = [];
    for (const candidate of candidates) {
      perspectiveVariants.push(...await createRectifiedVariants(cv, source, candidate.corners));
    }

    const denoised = new cv.Mat();
    const claheOutput = new cv.Mat();
    const thresholded = new cv.Mat();
    const clahe = new cv.CLAHE(2.2, new cv.Size(8, 8));
    try {
      cv.medianBlur(gray, denoised, 3);
      clahe.apply(denoised, claheOutput);
      cv.adaptiveThreshold(
        claheOutput,
        thresholded,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        31,
        11
      );
      const globalVariants = await Promise.all([
        matToBlob(cv, claheOutput),
        matToBlob(cv, thresholded),
      ]);
      return { variants: globalVariants, perspectiveVariants };
    } finally {
      denoised.delete();
      claheOutput.delete();
      thresholded.delete();
      clahe.delete();
    }
  } finally {
    source.delete();
    gray.delete();
    blurred.delete();
    edges.delete();
    kernel.delete();
    closed.delete();
  }
}
