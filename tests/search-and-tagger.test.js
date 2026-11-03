// tests/search-and-tagger.test.js
import test from "node:test";
import assert from "node:assert/strict";

import { deriveTags } from "../src/tagger.js";
import { normalizeQuery, filterBestOrSimilarMatches } from "../src/search.js";

test("deriveTags detects sports and temporal reminder tags for cricket match tmrw", () => {
  const tags = deriveTags("cricket match tmrw");
  assert.ok(tags.includes("#sports"), "Should detect #sports");
  assert.ok(tags.includes("#reminder"), "Should detect #reminder");
  assert.ok(tags.includes("#event"), "Should detect #event");
});

test("deriveTags detects signature tags for sign and signature", () => {
  const tags1 = deriveTags("sign");
  assert.ok(tags1.includes("#signature"), "Should detect #signature for sign");

  const tags2 = deriveTags("signature on paper");
  assert.ok(tags2.includes("#signature"), "Should detect #signature for signature on paper");
});

test("normalizeQuery includes synonyms and tag mappings for sign", () => {
  const queryTokens = normalizeQuery("sign");
  assert.ok(queryTokens.includes("sign"), "Should include sign token");
  assert.ok(queryTokens.includes("#signature"), "Should include #signature tag");
});

test("filterBestOrSimilarMatches returns ONLY the best match when score gap is large", () => {
  const candidates = [
    { id: "1", content: "Handwritten signature", _score: 120 },
    { id: "2", content: "Motorcycle license plate", _score: 40 },
    { id: "3", content: "cricket match tmrw", _score: 15 },
  ];

  const filtered = filterBestOrSimilarMatches(candidates);
  assert.equal(filtered.length, 1, "Only the single best match should be returned");
  assert.equal(filtered[0].id, "1");
  assert.equal(filtered[0]._score, 120);
});

test("filterBestOrSimilarMatches returns all similar confidence matches ordered high-to-low", () => {
  const candidates = [
    { id: "1", content: "Match A", _score: 95 },
    { id: "2", content: "Match B", _score: 85 },
    { id: "3", content: "Match C", _score: 82 },
    { id: "4", content: "Match D (low confidence)", _score: 30 },
  ];

  const filtered = filterBestOrSimilarMatches(candidates);
  assert.equal(filtered.length, 3, "All 3 similar confidence matches should be returned");
  assert.equal(filtered[0].id, "1");
  assert.equal(filtered[1].id, "2");
  assert.equal(filtered[2].id, "3");
  assert.ok(filtered[0]._score >= filtered[1]._score);
  assert.ok(filtered[1]._score >= filtered[2]._score);
});
