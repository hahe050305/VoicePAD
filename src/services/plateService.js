// src/services/plateService.js

const VALID_INDIAN_STATES = new Set([
  "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "GA", "GJ", "HP", "HR",
  "JH", "JK", "KA", "KL", "LA", "LD", "DL", "MH", "ML", "MN", "MP", "MZ", "NL",
  "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
]);

// Negative keywords that frequently appear in movie tickets, bills, documents, or apps
// that should never be parsed as vehicle plates
const NON_VEHICLE_KEYWORDS = [
  "BOOKMYSHOW", "CINEMA", "TICKET", "SCREEN", "SEAT", "AUDI", "SHOW",
  "INVOICE", "ORDER", "PAYMENT", "TRANSACTION", "RECEIPT", "BILL", "REFUND",
  "FLIPKART", "AMAZON", "SWIGGY", "ZOMATO", "COUPON", "VOUCHER", "VALID", "EXPIRES"
];

const BH_PLATE_PATTERN = /\b\d{2}\s*BH\s*\d{4}\s*[A-Z]{1,2}\b/;
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

  const bhMatch = upper.match(/\d{2}BH\d{4}[A-Z]{1,2}/);
  if (bhMatch) {
    const val = bhMatch[0];
    return `${val.slice(0, 2)} BH ${val.slice(4, 8)} ${val.slice(8)}`;
  }

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

function findStandardPlate(rawStr) {
  const upper = rawStr.toUpperCase();

  // If text is obviously a ticket, booking, invoice, or receipt, do not extract a plate
  for (const kw of NON_VEHICLE_KEYWORDS) {
    if (upper.includes(kw)) return null;
  }

  // Find candidate words or compact tokens that actually match a plate structure
  // Indian plate tokens: state (2 letters) + district (1-2 digits) + series (0-3 letters) + serial (3-4 digits)
  // Look for tokens that are separated by whitespace or line boundaries, or isolated tokens
  const tokens = upper.match(/[A-Z0-9]{6,12}/g) || [];
  const candidates = new Map();

  for (const token of tokens) {
    const compact = token.replace(/[^A-Z0-9]/g, "");
    if (compact.length < 8 || compact.length > 11) continue;

    const state = findState(compact.slice(0, 2));
    if (!state) continue;

    for (const districtLength of [2, 1]) {
      const districtStart = 2;
      const districtSlice = compact.slice(districtStart, districtStart + districtLength);
      const districtCandidates = expandDigitSequences(districtSlice);
      if (districtCandidates.length === 0) continue;

      const suffixStart = districtStart + districtLength;
      for (const serialLength of [4, 3]) {
        for (let seriesLength = 3; seriesLength >= 0; seriesLength--) {
          const rawSeries = compact.slice(suffixStart, suffixStart + seriesLength);
          const series = normalizeLetters(rawSeries);
          const serialStart = suffixStart + seriesLength;
          if (serialStart + serialLength !== compact.length) continue;
          if (series.length !== seriesLength || !/^[A-Z]{0,3}$/.test(series)) continue;

          const rawSerial = compact.slice(serialStart, serialStart + serialLength);
          const serialCandidates = expandDigitSequences(rawSerial);
          if (serialCandidates.length === 0) continue;

          for (const district of districtCandidates) {
            for (const serial of serialCandidates) {
              const value = `${state}${district}${series}${serial}`;
              let score = 0;
              score += seriesLength === 2 ? 10 : seriesLength === 1 ? 6 : 2;
              score += districtLength === 2 ? 5 : 2;
              score += serialLength === 4 ? 6 : 2;

              for (let i = 0; i < rawSerial.length; i++) {
                if (rawSerial[i] === serial[i]) score += 3;
                else if (rawSerial[i] === 'L' && serial[i] === '4') score += 2.5;
                else if (rawSerial[i] === 'B' && serial[i] === '6') score += 2.5;
                else score += 1;
              }

              // Plates must score >= 20 and be a distinct token
              if (score >= 20) {
                candidates.set(value, Math.max(candidates.get(value) || 0, score));
              }
            }
          }
        }
      }
    }
  }

  // Also support 2-row plates where state+district is on top line and series+serial on next line
  const lines = upper.split(/[\r\n]+/).map((l) => l.replace(/[^A-Z0-9]/g, "").trim()).filter(Boolean);
  for (let i = 0; i < lines.length - 1; i++) {
    const top = lines[i];
    const bottom = lines[i + 1];
    if (top.length >= 3 && top.length <= 5 && bottom.length >= 4 && bottom.length <= 7) {
      const combined = `${top}${bottom}`;
      const state = findState(combined.slice(0, 2));
      if (state) {
        const res = formatPlate(combined);
        if (res && res.split(" ").length >= 3) {
          candidates.set(combined, 25);
        }
      }
    }
  }

  const best = [...candidates.entries()].sort((first, second) => second[1] - first[1])[0];
  return best ? formatPlate(best[0]) : null;
}

export function parsePlateString(raw) {
  if (!raw) return null;

  const rawStr = String(raw).toUpperCase().trim();
  for (const kw of NON_VEHICLE_KEYWORDS) {
    if (rawStr.includes(kw)) return null;
  }

  const bhMatch = rawStr.match(BH_PLATE_PATTERN);
  if (bhMatch) return formatPlate(bhMatch[0]);

  return findStandardPlate(rawStr);
}
