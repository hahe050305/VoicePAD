import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer } from "vite";
import { formatPlate, parsePlateString } from "../src/services/plateService.js";
import { scoreOcrResult } from "../src/services/ocrRanking.js";
import { detectTextRegions } from "../src/services/ocrRegionDetection.js";
import { HYPERLOCAL_COCO_TAGS } from "../src/services/visionTags.js";

let server;
let deriveTags;

before(async () => {
  server = await createServer({
    configFile: false,
    plugins: [],
    server: { middlewareMode: true },
    appType: "custom",
    logLevel: "silent",
  });
  ({ deriveTags } = await server.ssrLoadModule("/src/tagger.js"));
});

after(async () => {
  await server?.close();
});

test("parses standard and two-line Indian registration plates", () => {
  assert.equal(parsePlateString("TN 59 AX 2332"), "TN 59 AX 2332");
  assert.equal(parsePlateString("TN 59\nAX 2332"), "TN 59 AX 2332");
  assert.equal(parsePlateString("KA01AB1234"), "KA 01 AB 1234");
});

test("corrects common OCR confusions and formats Bharat-series plates", () => {
  assert.equal(parsePlateString("D1 3CAF1234"), "DL 3 CAF 1234");
  assert.equal(formatPlate("22BH6517A"), "22 BH 6517 A");
  assert.equal(parsePlateString("22 BH 6517 A"), "22 BH 6517 A");
});

test("does not accept an invalid state code as a registration plate", () => {
  assert.equal(parsePlateString("ZZ 59 AX 2332"), null);
});

test("prefers the two-letter series reading when OCR includes conflicting plate candidates", () => {
  assert.equal(
    parsePlateString("Vehicle Plate: TN 59 AXZ 3321\nTN 59 AX 2332"),
    "TN 59 AX 2332"
  );
});

test("ranks a valid plate candidate above a higher-confidence invalid reading", () => {
  const valid = scoreOcrResult(
    { data: { text: "TN 59 AX 2332", confidence: 45 } },
    { plate: true }
  );
  const invalid = scoreOcrResult(
    { data: { text: "ZZ 59 AXX Z332", confidence: 92 } },
    { plate: true }
  );
  assert.ok(valid > invalid);
});

test("isolates a text block from a blank page image", () => {
  const width = 240;
  const height = 120;
  const pixels = new Uint8Array(width * height).fill(255);
  for (const yStart of [35, 44, 53]) {
    for (let y = yStart; y < yStart + 4; y++) {
      for (let x = 50; x < 190; x++) pixels[y * width + x] = 20;
    }
  }
  const [region] = detectTextRegions(pixels, width, height);
  assert.ok(region);
  assert.ok(region.x <= 50);
  assert.ok(region.x + region.width >= 190);
  assert.ok(region.y <= 35);
  assert.ok(region.y + region.height >= 57);
  assert.deepEqual(detectTextRegions(new Uint8Array(width * height).fill(255), width, height), []);
});

test("uses contextual tags rather than ambiguous one-word aliases", () => {
  assert.deepEqual(deriveTags("phone"), []);
  assert.deepEqual(deriveTags("phone number"), ["#contact"]);
  assert.deepEqual(deriveTags("gas bill"), ["#bill"]);
  assert.deepEqual(deriveTags("pass the station"), []);
  assert.deepEqual(deriveTags("stock photo"), ["#photo"]);
  assert.deepEqual(deriveTags("wifi\npassword"), ["#security"]);
  assert.deepEqual(deriveTags("Vehicle Plate TN 59 AX 2332"), ["#vehicle"]);
});

test("does not classify every picture as a selfie", () => {
  assert.deepEqual(deriveTags("picture"), ["#photo"]);
  assert.deepEqual(deriveTags("my selfie"), ["#photo", "#selfie"]);
});

test("maps detections to specific COCO labels without invented synonyms", () => {
  assert.deepEqual(HYPERLOCAL_COCO_TAGS.truck, ["#vehicle", "#truck"]);
  assert.deepEqual(HYPERLOCAL_COCO_TAGS.person, ["#person", "#people"]);
  assert.equal(HYPERLOCAL_COCO_TAGS.person.includes("#selfie"), false);
  assert.deepEqual(HYPERLOCAL_COCO_TAGS["sports ball"], ["#sports", "#ball"]);
});
