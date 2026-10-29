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

// In Indian High Security Registration Plates (HSRP), the German DIN 1451-based font has
// distinctive character geometry that OCR engines frequently misread:
// - '6' has a rounded top loop that OCR easily mistakes for 'B' or '8'
// - '4' has an open or angled stroke that OCR frequently misreads as 'L' or 'A'
// - '0' is misread as 'D', 'O', or 'Q'
// - '1' is misread as 'I' or 'L'
const DIGIT_ALTERNATIVES = {
  B: ["6", "8"],
  L: ["4", "1"],
  D: ["0"],
  O: ["0"],
  Q: ["0"],
  I: ["1"],
  Z: ["2"],
  S: ["5"],
  G: ["6"],
  A: ["4"],
};

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

function normalizeLetters(value) {
  return [...value].map((character) => OCR_TO_LETTER[character] || character).join("");
}

function getDigitCandidates(rawChar) {
  if (/^\d$/.test(rawChar)) return [rawChar];
  return DIGIT_ALTERNATIVES[rawChar] || [OCR_TO_DIGIT[rawChar] || rawChar];
}

function expandDigitSequences(rawString) {
  let results = [""];
  for (const ch of rawString) {
    const alts = getDigitCandidates(ch);
    const next = [];
    for (const prefix of results) {
      for (const alt of alts) {
        next.push(prefix + alt);
      }
    }
    results = next;
  }
  return results.filter((s) => /^\d+$/.test(s));
}

function findStandardPlate(compact, rawStr = "") {
  // If the raw text is very short or doesn't have at least some digit-like characters, reject to avoid hallucinations on noise
  const digitCount = (compact.match(/\d/g) || []).length;
  const letterCount = (compact.match(/[A-Z]/g) || []).length;
  if (compact.length < 8 || (digitCount < 2 && letterCount < 4)) {
    return null;
  }

  const candidates = new Map();
  for (let start = 0; start <= compact.length - 7; start++) {
    const state = findState(compact.slice(start, start + 2));
    if (!state) continue;

    for (const districtLength of [2, 1]) {
      const districtStart = start + 2;
      const districtSlice = compact.slice(districtStart, districtStart + districtLength);
      const districtCandidates = expandDigitSequences(districtSlice);
      if (districtCandidates.length === 0) continue;

      const suffixStart = districtStart + districtLength;
      for (const serialLength of [4, 3]) {
        for (let seriesLength = 3; seriesLength >= 0; seriesLength--) {
          const rawSeries = compact.slice(suffixStart, suffixStart + seriesLength);
          const series = normalizeLetters(rawSeries);
          const serialStart = suffixStart + seriesLength;
          if (serialStart + serialLength > compact.length) continue;
          if (series.length !== seriesLength || !/^[A-Z]{0,3}$/.test(series)) continue;

          const rawSerial = compact.slice(serialStart, serialStart + serialLength);
          const serialCandidates = expandDigitSequences(rawSerial);
          if (serialCandidates.length === 0) continue;

          for (const district of districtCandidates) {
            for (const serial of serialCandidates) {
              const value = `${state}${district}${series}${serial}`;
              let score = 0;
              // Length bonuses
              score += seriesLength === 2 ? 10 : seriesLength === 1 ? 6 : 2;
              score += districtLength === 2 ? 5 : 2;
              score += serialLength === 4 ? 6 : 2;

              // Penalty if rawSerial characters needed alternative substitutions
              for (let i = 0; i < rawSerial.length; i++) {
                if (rawSerial[i] === serial[i]) score += 3;
                else if (rawSerial[i] === 'L' && serial[i] === '4') score += 2.5; // HSRP L->4 is very high fidelity
                else if (rawSerial[i] === 'B' && serial[i] === '6') score += 2.5; // HSRP B->6 is very high fidelity
                else score += 1;
              }

              // Require a minimum score threshold so weak random combinations are rejected
              if (score >= 18) {
                candidates.set(value, Math.max(candidates.get(value) || 0, score));
              }
            }
          }
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

  const rawStr = String(raw).toUpperCase().trim();
  const compact = rawStr.replace(/[^A-Z0-9]/g, "");

  // Require minimum structure before testing
  if (compact.length < 7) return null;

  const bhMatch = compact.match(BH_PLATE_PATTERN);
  if (bhMatch) return formatPlate(bhMatch[0]);

  return findStandardPlate(compact, rawStr);
}
