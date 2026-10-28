function buildLocalMeans(luminance, width, height, tileSize) {
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
  return { means, columns, rows };
}

function localMeanAt(x, y, tileSize, means, columns, rows) {
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

export function detectTextRegions(luminance, width, height, {
  tileSize = 64,
  maxRegions = 2,
} = {}) {
  if (!width || !height || luminance.length !== width * height) return [];
  const { means, columns, rows } = buildLocalMeans(luminance, width, height, tileSize);
  const rowInk = new Uint32Array(height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (luminance[y * width + x] < localMeanAt(x, y, tileSize, means, columns, rows) - 12) {
        rowInk[y]++;
      }
    }
  }

  const minRowInk = Math.max(3, Math.round(width * 0.0025));
  const runs = [];
  let runStart = -1;
  for (let y = 0; y <= height; y++) {
    const active = y < height && rowInk[y] >= minRowInk;
    if (active && runStart < 0) runStart = y;
    if (!active && runStart >= 0) {
      runs.push({ top: runStart, bottom: y });
      runStart = -1;
    }
  }
  if (runs.length === 0) return [];

  const lineHeights = runs.map((run) => run.bottom - run.top);
  const medianLineHeight = [...lineHeights].sort((a, b) => a - b)[
    Math.floor(lineHeights.length / 2)
  ];
  const maxLineGap = Math.max(8, Math.min(42, medianLineHeight * 2.5));
  const groups = [];
  for (const run of runs) {
    const current = groups[groups.length - 1];
    if (current && run.top - current.bottom <= maxLineGap) {
      current.bottom = run.bottom;
    } else {
      groups.push({ top: run.top, bottom: run.bottom });
    }
  }

  const regions = [];
  for (const group of groups) {
    const groupHeight = group.bottom - group.top;
    if (groupHeight < 6) continue;
    const columnInk = new Uint32Array(width);
    for (let y = group.top; y < group.bottom; y++) {
      for (let x = 0; x < width; x++) {
        if (luminance[y * width + x] <
          localMeanAt(x, y, tileSize, means, columns, rows) - 12) {
          columnInk[x]++;
        }
      }
    }
    const minColumnInk = Math.max(2, Math.round(groupHeight * 0.025));
    let left = 0;
    while (left < width && columnInk[left] < minColumnInk) left++;
    let right = width - 1;
    while (right > left && columnInk[right] < minColumnInk) right--;
    if (right <= left) continue;

    const horizontalPadding = Math.max(8, Math.round((right - left) * 0.035));
    const verticalPadding = Math.max(6, Math.round(groupHeight * 0.2));
    const region = {
      x: Math.max(0, left - horizontalPadding),
      y: Math.max(0, group.top - verticalPadding),
      width: Math.min(width, right + horizontalPadding + 1) -
        Math.max(0, left - horizontalPadding),
      height: Math.min(height, group.bottom + verticalPadding) -
        Math.max(0, group.top - verticalPadding),
    };
    const areaRatio = (region.width * region.height) / (width * height);
    if (region.width < 40 || region.height < 12 || areaRatio < 0.002 || areaRatio > 0.82) continue;
    regions.push({ ...region, ink: Array.from(columnInk).reduce((sum, count) => sum + count, 0) });
  }

  regions.sort((a, b) => b.ink - a.ink);
  const selected = [];
  for (const region of regions) {
    const overlaps = selected.some((existing) => {
      const intersectionWidth = Math.max(
        0,
        Math.min(existing.x + existing.width, region.x + region.width) -
          Math.max(existing.x, region.x)
      );
      const intersectionHeight = Math.max(
        0,
        Math.min(existing.y + existing.height, region.y + region.height) -
          Math.max(existing.y, region.y)
      );
      return intersectionWidth * intersectionHeight / (region.width * region.height) > 0.7;
    });
    if (!overlaps) selected.push(region);
    if (selected.length >= maxRegions) break;
  }
  return selected.map(({ x, y, width: regionWidth, height: regionHeight }) => ({
    x,
    y,
    width: regionWidth,
    height: regionHeight,
  }));
}
