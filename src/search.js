// src/search.js
import { db } from "./db.js";
import aliasMap from "./aliasMap.json";
import { deriveTags } from "./tagger.js";
import { parseIntentWithLLM } from "./llmService.js";

/**
 * Normalizes a raw query string into lowercase tokens,
 * expanding known words to their canonical tags via aliasMap + semantic tagger.
 */
export function normalizeQuery(query) {
  if (!query) return [];

  // Get semantic tags from the full query
  const semanticTags = deriveTags(query);

  // Also tokenize individual words for substring matching
  const wordTokens = query
    .toLowerCase()
    .replace(/[^\w\s#]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((t) => {
      if (t.startsWith("#")) return t;
      const mapped = aliasMap[t];
      return mapped ? mapped : t;
    });

  // Merge: semantic tags + raw word tokens (deduped)
  const merged = new Set([...semanticTags, ...wordTokens]);
  return Array.from(merged);
}

let cachedEntries = [];
let cacheLoaded = false;

export async function refreshCache() {
  const notes = await db.notes.toArray();
  const images = await db.images.toArray();
  cachedEntries = [...notes, ...images];
  cacheLoaded = true;
  return cachedEntries;
}

export function getCachedEntries() {
  return cachedEntries;
}

/**
 * Scores a single entry against a list of tokens and intent hints.
 * Returns a numeric relevance score (0 = no match).
 */
function scoreEntry(entry, tokens, rawLower, llmIntent = null) {
  let score = 0;
  const text = (
    entry.type === "text" ? entry.content || "" : entry.ocr || ""
  ).toLowerCase();

  // Exact full-phrase match (highest signal)
  if (text.includes(rawLower)) score += 15;

  // Per-token match against content/ocr
  tokens.forEach((t) => {
    if (!t.startsWith("#") && text.includes(t)) score += 3;
  });

  // Tag matches
  const tagTokens = tokens.filter((t) => t.startsWith("#"));
  tagTokens.forEach((t) => {
    if (entry.tags && entry.tags.includes(t)) score += 10;
  });

  // LLM Intent Boost
  if (llmIntent) {
    if (llmIntent.target && text.includes(llmIntent.target.toLowerCase())) {
      score += 12;
    }
    if (llmIntent.categories && Array.isArray(llmIntent.categories)) {
      llmIntent.categories.forEach((cat) => {
        if (entry.tags && entry.tags.includes(cat)) {
          score += 8;
        }
      });
    }
  }

  // Null / Blurred OCR fallback:
  // If entry is an image with minimal or unreadable OCR text,
  // score it favorably when the query asks for visual items, documents, cards, or photos
  if (entry.type === "image") {
    const ocrLen = (entry.ocr || "").trim().length;
    const isBlurredOrEmpty = ocrLen < 15; // no text or only fragmented noise

    if (isBlurredOrEmpty) {
      const visualHints = [
        "#photo",
        "#image",
        "#document",
        "#vehicle",
        "#bill",
        "#parking",
        "#finance",
        "#contact"
      ];
      const matchesVisualHint = tokens.some((t) => visualHints.includes(t));
      if (matchesVisualHint) {
        // Generous boost so tight situations retrieve the photo immediately
        score += 8;
      }
    }
  }

  return score;
}

/**
 * Returns ALL scored and ranked entries for live feed filtering.
 * If rawQuery is empty/blank, returns all entries sorted by recency.
 */
export async function rankedSearch(rawQuery, llmIntent = null) {
  if (!cacheLoaded || cachedEntries.length === 0) {
    await refreshCache();
  }

  if (!rawQuery || !rawQuery.trim()) {
    return [...cachedEntries].sort((a, b) => b.timestamp - a.timestamp);
  }

  const tokens = normalizeQuery(rawQuery);
  if (tokens.length === 0) {
    return [...cachedEntries].sort((a, b) => b.timestamp - a.timestamp);
  }

  const rawLower = rawQuery.toLowerCase().trim();

  return cachedEntries
    .map((entry) => ({
      entry,
      score: scoreEntry(entry, tokens, rawLower, llmIntent),
    }))
    .filter((item) => item.score > 0)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return b.entry.timestamp - a.entry.timestamp;
    })
    .map((item) => item.entry);
}

/**
 * Hybrid fast search used for voice lookup:
 * 1. Instant local response (<2ms) via aliasMap & rule-matching.
 * 2. If high-confidence match found immediately, returns right away for speed.
 * 3. Otherwise races LLM intent parsing with a strict 400ms budget to prevent lag.
 */
export async function fastSearch(rawQuery, onProgressHit = null) {
  // 1. Instant local response (<2ms)
  const initialRanked = await rankedSearch(rawQuery);
  let bestHit = initialRanked.length > 0 ? initialRanked[0] : null;

  // If initial hit is already available, notify caller immediately for zero perceived lag
  if (bestHit && typeof onProgressHit === "function") {
    onProgressHit(bestHit);
  }

  // 2. If we already have a strong match (e.g., direct phrase, high-signal tags),
  // return immediately to maintain lightning-fast voice response
  if (bestHit && initialRanked.length === 1) {
    return bestHit;
  }

  // 3. Concurrently refine with local LLM with a 450ms deadline
  try {
    const intentPromise = parseIntentWithLLM(rawQuery);
    const timeoutPromise = new Promise((resolve) => setTimeout(() => resolve(null), 450));
    const intent = await Promise.race([intentPromise, timeoutPromise]);

    if (intent && (intent.target || (intent.categories && intent.categories.length > 0))) {
      const refinedRanked = await rankedSearch(rawQuery, intent);
      if (refinedRanked.length > 0) {
        bestHit = refinedRanked[0];
      }
    }
  } catch (err) {
    console.debug("LLM refinement skipped for speed:", err);
  }

  return bestHit;
}
