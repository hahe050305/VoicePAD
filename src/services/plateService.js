// src/services/plateService.js

const VALID_INDIAN_STATES = new Set([
  "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "GA", "GJ", "HP", "HR",
  "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL",
  "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
]);

const STATE_PREFIX = [...VALID_INDIAN_STATES].join("|");
const PLATE_PATTERN = new RegExp(`(?:${STATE_PREFIX})(?:[0-9]{1,2})(?:[A-Z]{0,3})(?:[0-9]{3,4})`);
const BH_PLATE_PATTERN = /\d{2}BH\d{4}[A-Z]{1,2}/;

export function formatPlate(value) {
  if (!value) return "";
  const upper = value.toUpperCase().replace(/[^A-Z0-9]/g, "");

  // Bharat Series: e.g. 22BH6517A -> 22 BH 6517 A
  const bhMatch = upper.match(BH_PLATE_PATTERN);
  if (bhMatch) {
    const val = bhMatch[0];
    return `${val.slice(0, 4)} ${val.slice(4, 8)} ${val.slice(8)}`;
  }

  // Standard Indian Plate: e.g. TN59AX2332 -> TN 59 AX 2332
  const state = upper.slice(0, 2);
  const districtMatch = upper.slice(2).match(/^\d{1,2}/);
  const district = districtMatch ? districtMatch[0] : "";
  const seriesStart = 2 + district.length;
  const seriesMatch = upper.slice(seriesStart).match(/^[A-Z]{0,3}/);
  const series = seriesMatch ? seriesMatch[0] : "";
  const serial = upper.slice(seriesStart + series.length);

  return [state, district, series, serial].filter(Boolean).join(" ").trim();
}

/**
 * Robustly parses Indian vehicle registration plates from raw OCR output.
 * Seamlessly handles single-line and two-row motorcycle plates (e.g., 'TN 59' on top, 'AX 2332' on bottom).
 */
export function parsePlateString(raw) {
  if (!raw) return null;

  const rawStr = String(raw).toUpperCase();

  // 1. Direct compact match across entire text (handles multiline bike plates & arbitrary whitespace)
  const compact = rawStr.replace(/[^A-Z0-9]/g, "");

  const bhMatch = compact.match(BH_PLATE_PATTERN);
  if (bhMatch) return formatPlate(bhMatch[0]);

  const plateMatch = compact.match(PLATE_PATTERN);
  if (plateMatch && VALID_INDIAN_STATES.has(plateMatch[0].slice(0, 2))) {
    return formatPlate(plateMatch[0]);
  }

  // 2. Check line-by-line and joined adjacent lines
  const lines = rawStr.split(/[\r\n]+/).map(l => l.replace(/[^A-Z0-9]/g, "")).filter(Boolean);
  const candidates = [...lines];
  for (let i = 0; i < lines.length - 1; i++) {
    candidates.push(lines[i] + lines[i + 1]);
  }

  for (const cand of candidates) {
    const cMatch = cand.match(PLATE_PATTERN);
    if (cMatch && VALID_INDIAN_STATES.has(cMatch[0].slice(0, 2))) {
      return formatPlate(cMatch[0]);
    }
  }

  return null;
}
