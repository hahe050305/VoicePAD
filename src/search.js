// src/search.js
import { db } from "./db.js";
import aliasMap from "./aliasMap.json";
import { deriveTags } from "./tagger.js";
import { parseIntentWithLLM } from "./llmService.js";
import { isPureImageText } from "./ocr.js";

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "at", "did", "do", "does", "for", "from", "give",
  "have", "help", "i", "in", "is", "it", "me", "my", "of", "on", "please",
  "show", "that", "the", "this", "to", "want", "was", "what", "where", "which",
  "who", "with", "you", "your", "find", "search", "look", "get", "tell", "about",
  "save", "saved", "store", "stored", "memory", "remember", "can", "could", "would",
  "should", "how", "when", "why", "any", "all", "upload", "uploaded", "put", "leave", "left",
  "keep", "kept", "take", "taken", "add", "added",
]);

const CATEGORY_TAGS = new Set([
  "#vehicle", "#document", "#contact", "#travel", "#bill", "#health", "#finance",
  "#shopping", "#security", "#reminder", "#photo", "#image", "#picture",
]);
const PHOTO_INTENT_TERMS = new Set([
  "photo", "selfie", "picture", "pic", "portrait", "snap", "image", "gallery", "camera", "roll", "face", "capture", "captured",
]);

const TERM_SYNONYMS = {
  bike: ["bike", "motorcycle", "bicycle"],
  bicycle: ["bicycle", "bike"],
  cycle: ["bicycle", "cycle"],
  motorcycle: ["motorcycle", "bike"],
  scooter: ["scooter", "motorcycle"],
  car: ["car", "automobile"],
  automobile: ["automobile", "car"],
  truck: ["truck"],
  vehicle: ["vehicle", "car", "truck", "motorcycle", "bicycle", "bus"],
  phone: ["phone", "cellphone", "cell phone", "mobile"],
  cell: ["cell", "cellphone", "cell phone", "phone"],
  cellphone: ["cellphone", "cell phone", "phone", "mobile"],
  mobile: ["mobile", "phone", "cellphone", "cell phone"],
  bag: ["bag", "backpack", "handbag", "suitcase"],
  plant: ["plant", "potted plant"],
  book: ["book"],
  bottle: ["bottle"],
  laptop: ["laptop"],
  scooty: ["scooty", "motorcycle", "bicycle"],
  dog: ["dog", "puppy"],
  puppy: ["puppy", "dog"],
  cat: ["cat", "kitten"],
  kitten: ["kitten", "cat"],
  photo: ["photo", "picture", "pic", "image", "selfie", "portrait", "snap", "face"],
  selfie: ["selfie", "photo", "picture", "portrait"],
};

function normalizeWord(word) {
  if (["news", "series", "species"].includes(word)) return word;
  const irregularPlurals = {
    photos: "photo",
    pictures: "picture",
    selfies: "selfie",
    images: "image",
    buses: "bus",
    boxes: "box",
  };
  if (irregularPlurals[word]) return irregularPlurals[word];
  if (word.length > 4 && word.endsWith("ies")) return `${word.slice(0, -3)}y`;
  if (word.length > 3 && word.endsWith("s") && !/(ss|us|is)$/.test(word)) return word.slice(0, -1);
  return word;
}

function wordsIn(text) {
  return (text.toLowerCase().match(/[a-z0-9]+/g) || []).map(normalizeWord);
}

// Bounded edit distance for short query tokens. This catches common typos and
// transpositions without turning broad substring matches into false positives.
function editDistanceAtMost(a, b, limit) {
  if (Math.abs(a.length - b.length) > limit) return limit + 1;
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const current = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      current[j] = Math.min(
        current[j - 1] + 1,
        previous[j] + 1,
        previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
      rowMin = Math.min(rowMin, current[j]);
    }
    if (rowMin > limit) return limit + 1;
    previous = current;
  }
  return previous[b.length];
}

function fuzzyWordMatch(queryWord, candidateWord) {
  if (queryWord === candidateWord) return 2;
  if (queryWord.length < 4 || candidateWord.length < 4) return 0;
  const limit = Math.min(queryWord.length, candidateWord.length) >= 8 ? 2 : 1;
  return editDistanceAtMost(queryWord, candidateWord, limit) <= limit ? 1 : 0;
}

function wordMatch(queryWord, candidateWords) {
  if (candidateWords.has(queryWord)) return 2;
  let best = 0;
  for (const candidate of candidateWords) best = Math.max(best, fuzzyWordMatch(queryWord, candidate));
  return best;
}

/**
 * Normalizes a raw query string into lowercase tokens,
 * expanding known words to their canonical tags via aliasMap + semantic tagger.
 */
export function normalizeQuery(query) {
  if (!query) return [];
  const rawWords = wordsIn(query);
  const terms = [...new Set(rawWords.filter((word) => (word.length > 1 || /^\d+$/.test(word)) && !STOP_WORDS.has(word)))];
  const explicitTags = (query.toLowerCase().match(/#[a-z0-9_]+/g) || []);
  const semanticTags = deriveTags(query).filter((tag) => tag !== "#note" || terms.includes("note"));
  const aliasTags = terms.map((term) => aliasMap[term]).filter(Boolean);
  return Array.from(new Set([...terms, ...explicitTags, ...semanticTags, ...aliasTags]));
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
function scoreEntry(entry, tokens, llmIntent = null) {
  let score = 0;
  const matchedTerms = new Set();
  const queryTerms = tokens.filter((token) => !token.startsWith("#"));
  const queryTags = tokens.filter((token) => token.startsWith("#"));
  const labelText = (entry.label || "").trim();
  const mainText = (
    entry.type === "text" ? entry.content || "" : entry.ocr || ""
  ).trim();
  const fullText = `${labelText} ${mainText}`.toLowerCase();
  const labelWords = new Set(wordsIn(labelText));
  const mainWords = new Set(wordsIn(mainText));
  const allWords = wordsIn(fullText);
  const compactText = allWords.join("");

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

  const isPhotoQuery = queryTerms.some((term) =>
    PHOTO_INTENT_TERMS.has(term)
  ) || queryTags.some((tag) => ["#photo", "#selfie", "#image", "#picture"].includes(tag));

  for (const term of queryTerms) {
    const variants = TERM_SYNONYMS[term] || [term];
    const labelMatch = variants.some((variant) => {
      const variantWords = wordsIn(variant);
      return variantWords.length === 1
        ? wordMatch(variantWords[0], labelWords) > 0
        : ` ${wordsIn(labelText).join(" ")} `.includes(` ${variantWords.join(" ")} `);
    });
    const textMatch = variants.some((variant) => {
      const variantWords = wordsIn(variant);
      return variantWords.length === 1
        ? wordMatch(variantWords[0], mainWords) > 0
        : ` ${wordsIn(mainText).join(" ")} `.includes(` ${variantWords.join(" ")} `);
    });

    if (labelMatch) {
      matchedTerms.add(term);
      score += variants.some((variant) => labelWords.has(wordsIn(variant)[0])) ? 80 : 55;
    } else if (textMatch) {
      matchedTerms.add(term);
      score += variants.some((variant) => mainWords.has(wordsIn(variant)[0])) ? 45 : 30;
    }

    const tagsToCheck = new Set([
      `#${term}`,
      `#label_${term}`,
      ...(aliasMap[term] ? [aliasMap[term]] : []),
      ...variants.map((variant) => `#${variant.replace(/\s+/g, "_")}`),
    ]);
    for (const tag of tagsToCheck) {
      if (!effectiveTags.has(tag)) continue;
      if (CATEGORY_TAGS.has(tag) && !["vehicle", "document", "contact", "travel", "bill", "health", "finance", "shopping", "security", "reminder", "photo", "image", "picture"].includes(term)) {
        continue;
      }
      matchedTerms.add(term);
      score += tag === `#label_${term}` ? 65 : 35;
      break;
    }

    if (term === "note" && entry.type === "text") {
      matchedTerms.add(term);
      score += 25;
    }

    if (entry.type === "image" && Array.isArray(entry.detectedObjects)) {
      const objects = entry.detectedObjects.map((object) => wordsIn(object).join(" "));
      if (variants.some((variant) => objects.includes(wordsIn(variant).join(" ")))) {
        matchedTerms.add(term);
        score += 75;
      }
    }
  }

  // Explicit multiword phrases receive a bonus only on whole-word boundaries.
  const phrase = queryTerms.join(" ");
  if (queryTerms.length > 1 && phrase && ` ${allWords.join(" ")} `.includes(` ${phrase} `)) {
    score += 35;
    queryTerms.forEach((term) => matchedTerms.add(term));
  }

  // Let a query split by spaces still match a single stored word ("iph one"
  // vs "iPhone"), and allow small typos in multiword queries as a phrase.
  const compactQuery = queryTerms.join("");
  if (queryTerms.length > 1 && compactQuery.length >= 4) {
    const compactDistance = editDistanceAtMost(compactQuery, compactText, 1);
    if (compactText.includes(compactQuery) || compactDistance <= 1) {
      score += 24;
      queryTerms.forEach((term) => matchedTerms.add(term));
    }
  }

  if (isPhotoQuery) {
    if (entry.type === "image") {
      score += 25;
      queryTerms.filter((term) => PHOTO_INTENT_TERMS.has(term)).forEach((term) => matchedTerms.add(term));
    } else if (!/\b(photo|selfie|picture|portrait|image)\b/i.test(fullText)) {
      return 0;
    }
  }

  // Category tags may support explicit category searches, but never turn an
  // unrelated one-word alias into a match on their own.
  for (const tag of queryTags) {
    if (effectiveTags.has(tag) && !CATEGORY_TAGS.has(tag)) {
      score += 20;
      const associatedTerm = queryTerms.find((term) =>
        aliasMap[term] === tag || `#${term}` === tag ||
        (tag === "#parking" && ["park", "parked", "parking"].includes(term))
      );
      if (associatedTerm) matchedTerms.add(associatedTerm);
    }
  }

  if (llmIntent && score > 0) {
    if (llmIntent.target && wordsIn(fullText).includes(normalizeWord(llmIntent.target.toLowerCase()))) score += 8;
    if (Array.isArray(llmIntent.categories)) {
      llmIntent.categories.forEach((category) => {
        if (effectiveTags.has(category)) score += 4;
      });
    }
  }

  const requiredMatches = Math.max(1, Math.ceil(queryTerms.length * 0.66));
  if (matchedTerms.size < requiredMatches) return 0;
  if (isPure && !isPhotoQuery && !labelText && score === 0) return 0;
  return score;
}

/**
 * Returns up to ten relevant entries, ranked by match strength.
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
    return [];
  }

  return cachedEntries
    .map((entry) => ({
      entry,
      score: scoreEntry(entry, tokens, llmIntent),
    }))
    .filter((item) => item.score > 0)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return b.entry.timestamp - a.entry.timestamp;
    })
    .slice(0, 10)
    .map((item) => ({ ...item.entry, _score: item.score }));
}

/**
 * Fast search used for voice lookup:
 * - For photo-family queries (photo, selfie, picture, snap, etc.),
 *   returns the top matching photos.
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
    // Keep the strongest photo matches ranked first.
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

