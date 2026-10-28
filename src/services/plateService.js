// src/services/plateService.js

const VALID_INDIAN_STATES = new Set([
  "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "GA", "GJ", "HP", "HR",
  "JH", "JK", "KA", "KL", "LA", "LD", "DL", "MH", "ML", "MN", "MP", "MZ", "NL",
  "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
]);

const BH_PLATE_PATTERN = /\d{2}BH\d{4}[A-Z]{1,2}/;
const STATE_OCR_ALTERNATIVES = {
  "0": ["O"],
  "1": ["I", "L"],
  "2": ["Z"],
  "5": ["S"],
  "6": ["G"],
  "8": ["B"],
};
const OCR_TO_DIGIT = { O: "0", Q: "0", D: "0", I: "1", L: "1", Z: "2", S: "5", G: "6", B: "8" };
const OCR_TO_LETTER = { "0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B" };

export function formatPlate(value) {
  if (!value) return "";
  const upper = value.toUpperCase().replace(/[^A-Z0-9]/g, "");

  // Bharat Series: e.g. 22BH6517A -> 22 BH 6517 A
  const bhMatch = upper.match(BH_PLATE_PATTERN);
  if (bhMatch) {
    const val = bhMatch[0];
    return `${val.slice(0, 2)} BH ${val.slice(4, 8)} ${val.slice(8)}`;
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

function findState(value) {
  const alternatives = [...value].map((character) => STATE_OCR_ALTERNATIVES[character] || [character]);
  for (const first of alternatives[0]) {
    for (const second of alternatives[1]) {
      const state = `${first}${second}`;
      if (VALID_INDIAN_STATES.has(state)) return state;
    }
  }
  return null;
}

function normalizeDigits(value) {
  return [...value].map((character) => OCR_TO_DIGIT[character] || character).join("");
}

function normalizeLetters(value) {
  return [...value].map((character) => OCR_TO_LETTER[character] || character).join("");
}

function findStandardPlate(compact) {
  const candidates = new Map();
  for (let start = 0; start <= compact.length - 7; start++) {
    const state = findState(compact.slice(start, start + 2));
    if (!state) continue;

    for (const districtLength of [2, 1]) {
      const districtStart = start + 2;
      const district = normalizeDigits(compact.slice(districtStart, districtStart + districtLength));
      if (district.length !== districtLength || !/^\d{1,2}$/.test(district)) continue;

      const suffixStart = districtStart + districtLength;
      for (const serialLength of [4, 3]) {
        for (let seriesLength = 3; seriesLength >= 0; seriesLength--) {
          const series = normalizeLetters(compact.slice(suffixStart, suffixStart + seriesLength));
          const serialStart = suffixStart + seriesLength;
          const serial = normalizeDigits(compact.slice(serialStart, serialStart + serialLength));
          if (serial.length !== serialLength || !/^\d{3,4}$/.test(serial)) continue;
          if (series.length !== seriesLength || !/^[A-Z]{0,3}$/.test(series)) continue;
          if (serialStart + serialLength > compact.length) continue;

          const value = `${state}${district}${series}${serial}`;
          const seriesScore = seriesLength === 2 ? 10 : seriesLength === 1 ? 6 : 2;
          const districtScore = districtLength === 2 ? 5 : 2;
          const serialScore = serialLength === 4 ? 4 : 1;
          const score = seriesScore + districtScore + serialScore;
          candidates.set(value, Math.max(candidates.get(value) || 0, score));
        }
      }
    }
  }
  const best = [...candidates.entries()].sort((first, second) => second[1] - first[1])[0];
  return best ? formatPlate(best[0]) : null;
}

/**
 * Robustly parses Indian vehicle registration plates from raw OCR output.
 * Seamlessly handles single-line and two-row motorcycle plates (e.g., 'TN 59' on top, 'AX 2332' on bottom).
 */
export function parsePlateString(raw) {
  if (!raw) return null;

  const rawStr = String(raw).toUpperCase();
  const compact = rawStr.replace(/[^A-Z0-9]/g, "");

  const bhMatch = compact.match(BH_PLATE_PATTERN);
  if (bhMatch) return formatPlate(bhMatch[0]);

  return findStandardPlate(compact);
}
