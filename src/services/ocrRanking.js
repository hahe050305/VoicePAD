import { parsePlateString } from "./plateService.js";

export function scoreOcrText(text) {
  const cleanText = String(text || "").trim();
  if (!cleanText) return 0;
  const alphanumericCount = (cleanText.match(/[A-Z0-9]/gi) || []).length;
  const words = cleanText.split(/\s+/).filter((word) => /[A-Z0-9]{2,}/i.test(word));
  const structuredIdCount = [
    /[A-Z]{5}\d{4}[A-Z]/i,
    /\d{12}/,
    /[A-Z]{2}\d{13,15}/i,
  ].filter((pattern) => pattern.test(cleanText)).length;
  const plateBonus = parsePlateString(cleanText) ? 120 : 0;
  return alphanumericCount * 0.25 + Math.min(words.length, 20) * 2 +
    structuredIdCount * 80 + plateBonus;
}

export function scoreOcrResult(result, { plate = false } = {}) {
  const text = result?.data?.text?.trim() || "";
  const confidence = Number(result?.data?.confidence) || 0;
  const contentScore = scoreOcrText(text);
  const plateScore = plate && parsePlateString(text) ? 100 : 0;
  return confidence + contentScore + plateScore;
}
