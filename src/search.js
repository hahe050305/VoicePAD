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
  bike: ["bike", "motorcycle", "bicycle", "scooty", "scooter", "twowheeler"],
  bicycle: ["bicycle", "bike", "cycle"],
  cycle: ["bicycle", "cycle"],
  motorcycle: ["motorcycle", "bike", "scooter", "scooty", "twowheeler"],
  scooter: ["scooter", "motorcycle", "scooty", "bike"],
  scooty: ["scooty", "motorcycle", "scooter", "bike", "twowheeler"],
  car: ["car", "automobile", "fourwheeler", "suv"],
  automobile: ["automobile", "car"],
  truck: ["truck"],
  vehicle: ["vehicle", "car", "truck", "motorcycle", "bicycle", "bike", "scooty", "scooter", "bus", "plate", "numberplate"],
  plate: ["plate", "numberplate", "license", "registration", "vehicle"],
  numberplate: ["numberplate", "plate", "license", "registration", "vehicle", "bike", "car"],
  phone: ["phone", "cellphone", "cell phone", "mobile"],
  cell: ["cell", "cellphone", "cell phone", "phone"],
  cellphone: ["cellphone", "cell phone", "phone", "mobile"],
  mobile: ["mobile", "phone", "cellphone", "cell phone"],
  bag: ["bag", "backpack", "handbag", "suitcase"],
  plant: ["plant", "potted plant"],
  book: ["book"],
  bottle: ["bottle"],
  laptop: ["laptop"],
  dog: ["dog", "puppy"],
  puppy: ["puppy", "dog"],
  cat: ["cat", "kitten"],
  kitten: ["kitten", "cat"],
  photo: ["photo", "picture", "pic", "image", "selfie", "portrait", "snap", "face"],
  selfie: ["selfie", "photo", "picture", "portrait"],
  diwali: ["diwali", "deepavali", "festival", "festive"],
  festival: ["festival", "diwali", "celebration"],
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
    bikes: "bike",
    cars: "car",
    plates: "plate",
  };
  if (irregularPlurals[word]) return irregularPlurals[word];
  if (word.length > 4 && word.endsWith("ies")) return `${word.slice(0, -3)}y`;
  if (word.length > 3 && word.endsWith("s") && !/(ss|us|is)$/.test(word)) return word.slice(0, -1);
  return word;
}

function wordsIn(text) {
  return (text.toLowerCase().match(/[a-z0-9]+/g) || []).map(normalizeWord);
}

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
  if (queryWord.length >= 3 && candidateWord.includes(queryWord)) return 1.5;
  if (candidateWord.length >= 3 && queryWord.includes(candidateWord)) return 1.5;

  const maxLen = Math.max(queryWord.length, candidateWord.length);
  const minLen = Math.min(queryWord.length, candidateWord.length);
  if (minLen >= 3) {
    const limit = maxLen >= 7 ? 2 : 1;
    if (editDistanceAtMost(queryWord, candidateWord, limit) <= limit) return 1;
  }
  return 0;
}

function wordMatch(queryWord, candidateWords) {
  if (candidateWords.has(queryWord)) return 2;
  let best = 0;
  for (const candidate of candidateWords) {
    best = Math.max(best, fuzzyWordMatch(queryWord, candidate));
  }
  return best;
}

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
 * Scores an entry against query tokens with Priority Boosting for real images on visual queries.
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
  }

  // Detect visual/entity query intent (e.g. numberplate, bike, car, plate)
  const isPlateOrVehicleQuery = queryTerms.some((term) =>
    ["numberplate", "plate", "license", "registration", "vehicle", "bike", "car", "scooty"].includes(term)
  );

  for (const term of queryTerms) {
    const variants = TERM_SYNONYMS[term] || [term];

    const labelScore = Math.max(...variants.map(v => wordMatch(wordsIn(v)[0] || v, labelWords)), 0);
    const textScore = Math.max(...variants.map(v => wordMatch(wordsIn(v)[0] || v, mainWords)), 0);

    if (labelScore > 0) {
      matchedTerms.add(term);
      score += labelScore >= 2 ? 80 : 55;
    } else if (textScore > 0) {
      matchedTerms.add(term);
      score += textScore >= 2 ? 50 : 35;
    }

    const tagsToCheck = new Set([
      `#${term}`,
      `#label_${term}`,
      ...(aliasMap[term] ? [aliasMap[term]] : []),
      ...variants.map((variant) => `#${variant.replace(/\s+/g, "_")}`),
    ]);
    for (const tag of tagsToCheck) {
      if (!effectiveTags.has(tag)) continue;
      matchedTerms.add(term);
      score += tag === `#label_${term}` ? 65 : 40;
      break;
    }

    if (term.length >= 3 && compactText.includes(term)) {
      matchedTerms.add(term);
      score += 45;
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

  const compactQuery = queryTerms.join("");
  if (queryTerms.length > 1 && compactQuery.length >= 3) {
    if (compactText.includes(compactQuery)) {
      score += 50;
      queryTerms.forEach((term) => matchedTerms.add(term));
    } else if (editDistanceAtMost(compactQuery, compactText, 1) <= 1) {
      score += 35;
      queryTerms.forEach((term) => matchedTerms.add(term));
    }
  }

  for (const tag of queryTags) {
    if (effectiveTags.has(tag) && !CATEGORY_TAGS.has(tag)) {
      score += 25;
      matchedTerms.add(tag.replace("#", ""));
    }
  }

  // KEY BOOST: If the user searches for a plate/vehicle and this is a vehicle image with a plate,
  // BOOST IT OVER PLAIN TEXT NOTES (+150 points) so it always ranks #1!
  if (entry.type === "image" && isPlateOrVehicleQuery) {
    const hasPlate = effectiveTags.has("#vehicle") || /vehicle plate|br01|dl\d|mh\d/i.test(fullText);
    if (hasPlate) {
      score += 150;
      queryTerms.forEach((t) => matchedTerms.add(t));
    }
  }

  const requiredMatches = Math.max(1, Math.ceil(queryTerms.length * 0.5));
  if (matchedTerms.size < requiredMatches) return 0;
  return score;
}

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
    .slice(0, 15)
    .map((item) => ({ ...item.entry, _score: item.score }));
}

export async function fastSearch(rawQuery, onProgressHit = null) {
  const initialRanked = await rankedSearch(rawQuery);
  if (initialRanked.length === 0) return null;

  if (typeof onProgressHit === "function") {
    onProgressHit(initialRanked);
  }
  return initialRanked;
}
