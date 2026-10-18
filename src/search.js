// src/search.js
import { db } from "./db.js";
import aliasMap from "./aliasMap.json";
import { deriveTags } from "./tagger.js";
import { parseIntentWithLLM } from "./llmService.js";
import { isPureImageText } from "./ocr.js";

/**
 * Normalizes a raw query string into lowercase tokens,
 * expanding known words to their canonical tags via aliasMap + semantic tagger.
 */
export function normalizeQuery(query) {
  if (!query) return [];

  // Get semantic tags from the full query
  const semanticTags = deriveTags(query);

  // Also tokenize individual words for substring matching (preserve BOTH raw word AND mapped tag)
  const rawWordTokens = [];
  const aliasTags = [];

  query
    .toLowerCase()
    .replace(/[^\w\s#]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .forEach((t) => {
      rawWordTokens.push(t);
      if (aliasMap[t]) {
        aliasTags.push(aliasMap[t]);
      }
    });

  // Merge: semantic tags + alias tags + raw word tokens (deduped)
  const merged = new Set([...semanticTags, ...aliasTags, ...rawWordTokens]);
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
  const labelText = (entry.label || "").trim();
  const mainText = (
    entry.type === "text" ? entry.content || "" : entry.ocr || ""
  ).trim();
  const fullText = `${labelText} ${mainText}`.toLowerCase();

  const isPure =
    entry.type === "image" &&
    (entry.isPureImage === true || (!entry.ocr && !entry.label) || isPureImageText(entry.ocr));

  // Determine effective tags for the entry:
  // If entry has readable text and is not a pure image, strip #photo tags so text docs don't masquerade as photos
  const rawTags = entry.tags || [];
  const effectiveTags = new Set(
    rawTags.filter((t) => {
      if (!isPure && (t === "#photo" || t === "#selfie") && !fullText.includes("photo") && !fullText.includes("selfie")) {
        return false;
      }
      return true;
    })
  );

  if (isPure) {
    effectiveTags.add("#photo");
    effectiveTags.add("#image");
    effectiveTags.add("#picture");
    // Note: #selfie is NOT force-added — it stays only if the entry was
    // originally tagged as selfie, so "selfie" queries still match all photos
    // (they are treated as a photo-family query below) while preserving the
    // distinction for future per-entry selfie labeling.
  }

  // 1. Exact full-phrase match against text/label (highest signal)
  if (rawLower && fullText.includes(rawLower)) {
    score += 35;
  }

  // 2. Direct manual label match
  if (labelText && (rawLower.includes(labelText.toLowerCase()) || labelText.toLowerCase().includes(rawLower))) {
    score += 40;
  }

  // 3. Per-token match against content/ocr/label (word matches)
  tokens.forEach((t) => {
    if (!t.startsWith("#")) {
      if (labelText && labelText.toLowerCase().includes(t)) {
        score += 25; // High confidence for user-labeled images
      } else if (mainText.toLowerCase().includes(t)) {
        score += 15;
      }
    }
  });

  // 4. Tag matches against sanitized effective tags
  const tagTokens = tokens.filter((t) => t.startsWith("#"));
  tagTokens.forEach((t) => {
    if (effectiveTags.has(t)) {
      if ((t === "#photo" || t === "#selfie") && !isPure && !fullText.includes("photo") && !fullText.includes("selfie")) {
        return; // Skip false photo tags on text notes
      }
      score += 12;
    }
  });

  // 5. Manual label tag match: e.g. token 'aadhaar' matching '#label_aadhaar'
  tokens.forEach((t) => {
    const cleanT = t.replace(/^#/, "");
    if (effectiveTags.has(`#label_${cleanT}`) || effectiveTags.has(`#${cleanT}`)) {
      score += 30;
    }
  });

  // 6. Photo & Selfie Intent Match
  const isPhotoQuery = tokens.some((t) =>
    [
      "#photo", "#selfie", "#image", "#picture",
      "selfie", "selfies", "photo", "photos",
      "picture", "pictures", "pic", "pics",
      "portrait", "portraits", "snap", "snaps",
      "image", "images", "gallery", "camera"
    ].includes(t)
  );

  if (isPhotoQuery) {
    if (entry.type === "image") {
      // Any uploaded image is a match for "photo"
      score += isPure ? 35 : 25;
    } else {
      // Non-image note: DISQUALIFY from matching photo/selfie queries
      // unless the text explicitly mentions photo or selfie
      const hasExplicit = fullText.includes("selfie") || fullText.includes("photo") || fullText.includes("portrait") || fullText.includes("picture");
      if (!hasExplicit) {
        return 0; // Immediate 0 score
      }
    }
  }

  // 7. General text query on an unlabeled pure image:
  // If user is searching for something specific like "harish", "insurance", "bike", "wifi", "bill":
  // A pure photo that has no text and no matching label must NOT match!
  if (isPure && !isPhotoQuery && !labelText) {
    return 0;
  }

  // 8. LLM Intent Boost
  if (llmIntent) {
    if (llmIntent.target && fullText.includes(llmIntent.target.toLowerCase())) {
      score += 12;
    }
    if (llmIntent.categories && Array.isArray(llmIntent.categories)) {
      llmIntent.categories.forEach((cat) => {
        if (effectiveTags.has(cat)) {
          score += 8;
        }
      });
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
    .map((item) => ({ ...item.entry, _score: item.score }));
}

/**
 * Fast search used for voice lookup:
 * - For photo-family queries (photo, selfie, picture, snap, etc.),
 *   returns ALL matching photos so the user sees their full gallery.
 * - For other queries, returns ONLY the single highest-confidence match.
 * 1. Instant local response (<2ms) via aliasMap & rule-matching.
 * 2. Background LLM refinement within a strict deadline.
 */
const PHOTO_FAMILY_WORDS = new Set([
  "photo", "photos", "selfie", "selfies", "picture", "pictures",
  "pic", "pics", "portrait", "portraits", "snap", "snaps", "image", "images",
  "gallery", "camera", "#photo", "#selfie", "#image", "#picture"
]);

export async function fastSearch(rawQuery, onProgressHit = null) {
  const initialRanked = await rankedSearch(rawQuery);
  if (initialRanked.length === 0) return null;

  // Detect if this is a photo-family query (check words, tokens, and raw query)
  const queryLower = (rawQuery || "").toLowerCase();
  const queryWords = queryLower.replace(/[^\w\s#]/g, " ").split(/\s+/).filter(Boolean);
  const normalizedTokens = normalizeQuery(rawQuery);
  const isPhotoFamilyQuery =
    queryWords.some((w) => PHOTO_FAMILY_WORDS.has(w)) ||
    normalizedTokens.some((t) => PHOTO_FAMILY_WORDS.has(t)) ||
    /\b(photo|photos|selfie|selfies|picture|pictures|pic|pics|portrait|snap|snaps|image|images)\b/i.test(queryLower);

  if (isPhotoFamilyQuery) {
    // Return ALL matching photos (rankedSearch already filtered to score > 0)
    if (typeof onProgressHit === "function") {
      onProgressHit(initialRanked); // send array of all matches
    }
    return initialRanked;
  }

  // Non-photo query: single best hit
  let bestHit = initialRanked[0];

  // If initial hit is available, notify caller immediately
  if (typeof onProgressHit === "function") {
    onProgressHit(bestHit);
  }

  // Refine with local LLM if intent helps narrow down
  try {
    const intentPromise = parseIntentWithLLM(rawQuery);
    const timeoutPromise = new Promise((resolve) => setTimeout(() => resolve(null), 350));
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

