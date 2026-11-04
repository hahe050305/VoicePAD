// src/services/plateService.js

const VALID_INDIAN_STATES = new Set([
  "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "GA", "GJ", "HP", "HR",
  "JH", "JK", "KA", "KL", "LA", "LD", "DL", "MH", "ML", "MN", "MP", "MZ", "NL",
  "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
]);

// Negative keywords that frequently appear in movie tickets, bills, documents, badges, or apps
// that should never be parsed as vehicle plates
const NON_VEHICLE_KEYWORDS = [
  "BOOKMYSHOW", "CINEMA", "TICKET", "SCREEN", "SEAT", "AUDI", "SHOW",
  "INVOICE", "ORDER", "PAYMENT", "TRANSACTION", "RECEIPT", "BILL", "REFUND",
  "FLIPKART", "AMAZON", "SWIGGY", "ZOMATO", "COUPON", "VOUCHER", "VALID", "EXPIRES",
  "HACKERRANK", "ORCHESTRATE", "ATTENDING", "PARTICIPANT", "ATTENDEE", "CONFERENCE",
  "HACKATHON", "EDITION", "SPEAKER", "DELEGATE", "ORGANIZER", "BADGE", "INVITATION",
  "WELCOME", "DOT COM", "INFOWAY", "TECHNOLOGY", "TECHNOLOGIES", "SOLUTIONS", "COMPANY",
  "PVT", "LTD", "LIMITED", "CAFE", "COFFEE", "RESTAURANT", "MENU", "HOTEL", "RESORT",
  "HOMESTAY", "LOVE", "PEACE", "HAPPY", "BIRTHDAY", "FESTIVAL", "CELEBRATION"
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

  const lines = upper
    .split(/[\r\n]+/)
    .map((l) => l.trim())
    .filter(Boolean);

  const tokens = [];

  // 1. Single standalone compact tokens: e.g. KA01AB1234, TN-59-AX-2332
  const singleTokens = upper.match(/[A-Z0-9]{8,12}/g) || [];
  for (const st of singleTokens) {
    tokens.push(st);
    if (st.startsWith("IND") && st.length >= 11) {
      tokens.push(st.slice(3));
    }
  }

  // 2. Intra-line word combinations: only combine adjacent words on the SAME line
  for (const line of lines) {
    const lineWords = line.replace(/[^A-Z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
    // Real plates on one line have 2 to 4 compact chunks (e.g., ["TN", "59", "AX", "2332"])
    if (lineWords.length >= 2 && lineWords.length <= 5) {
      for (let i = 0; i < lineWords.length; i++) {
        let combo = "";
        for (let j = i; j < Math.min(lineWords.length, i + 4); j++) {
          combo += lineWords[j];
          if (combo.length >= 8 && combo.length <= 11) {
            tokens.push(combo);
            if (combo.startsWith("IND") && combo.length >= 11) {
              tokens.push(combo.slice(3));
            }
          }
        }
      }
    }
  }

  // 3. Two-line plates: e.g. Line 1: "BR 01 C", Line 2: "J 6440" or Line 1: "TN 59", Line 2: "AX 2332"
  for (let i = 0; i < lines.length - 1; i++) {
    const top = lines[i].replace(/[^A-Z0-9]/g, "");
    const bottom = lines[i + 1].replace(/[^A-Z0-9]/g, "");
    if (top.length >= 3 && top.length <= 5 && bottom.length >= 4 && bottom.length <= 7) {
      const combined = `${top}${bottom}`;
      if (combined.length >= 8 && combined.length <= 11) {
        tokens.push(combined);
      }
    }
  }

  const candidates = new Map();

  for (const token of tokens) {
    const compact = token.replace(/[^A-Z0-9]/g, "");
    if (compact.length < 8 || compact.length > 11) continue;

    // Genuine Indian plates contain at least 3 actual numeric digits in the raw text
    const rawDigits = (compact.match(/\d/g) || []).length;
    if (rawDigits < 3) continue;

    const state = findState(compact.slice(0, 2));
    if (!state) continue;

    const districtLengths = state === "DL" ? [2, 1] : [2];

    for (const districtLength of districtLengths) {
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
          // If serial is 3 digits, all 3 characters must be genuine digits
          if (serialLength === 3 && !/^\d{3}$/.test(rawSerial)) continue;

          const serialCandidates = expandDigitSequences(rawSerial);
          if (serialCandidates.length === 0) continue;

          // Must have at least 2 actual digits in the raw serial
          const rawSerialDigits = (rawSerial.match(/\d/g) || []).length;
          if (rawSerialDigits < 2) continue;

          for (const district of districtCandidates) {
            for (const serial of serialCandidates) {
              const value = `${state}${district}${series}${serial}`;
              let score = 0;
              score += seriesLength === 2 ? 12 : seriesLength === 1 ? 8 : 4;
              score += districtLength === 2 ? 6 : 3;
              score += serialLength === 4 ? 8 : 4;

              for (let idx = 0; idx < rawSerial.length; idx++) {
                if (rawSerial[idx] === serial[idx]) score += 3;
                else if (rawSerial[idx] === 'L' && serial[idx] === '4') score += 2;
                else if (rawSerial[idx] === 'B' && serial[idx] === '6') score += 2;
                else score += 1;
              }

              if (score >= 22) {
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
