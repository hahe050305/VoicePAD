// tests/moonshine.test.js
import test from "node:test";
import assert from "node:assert/strict";

import { resampleTo16k, preparePcmForMoonshine } from "../src/services/moonshineService.js";

test("resampleTo16k preserves 16kHz buffer and downsamples 48kHz to 16kHz", () => {
  const input16k = new Float32Array([0.1, -0.2, 0.3]);
  assert.strictEqual(resampleTo16k(input16k, 16000), input16k);

  const input48k = new Float32Array(48000);
  for (let i = 0; i < input48k.length; i++) {
    input48k[i] = Math.sin((2 * Math.PI * 440 * i) / 48000) * 0.4;
  }
  const resampled = resampleTo16k(input48k, 48000);
  assert.equal(resampled.length, 16000);
});

test("preparePcmForMoonshine rejects pure silence and too-short clips", () => {
  const silence = new Float32Array(16000);
  assert.equal(preparePcmForMoonshine(silence), null);

  const tooShort = new Float32Array(1600);
  tooShort.fill(0.5);
  assert.equal(preparePcmForMoonshine(tooShort), null);
});

test("preparePcmForMoonshine removes DC offset, trims silence, and boosts quiet speech", () => {
  // 2 seconds total: 0.7s silence (with DC offset), 0.6s quiet speech, 0.7s silence
  const pcm = new Float32Array(32000);
  const dcOffset = 0.05;
  pcm.fill(dcOffset);

  for (let i = 11200; i < 20800; i++) {
    pcm[i] = dcOffset + Math.sin((2 * Math.PI * 300 * i) / 16000) * 0.04;
  }

  const prepared = preparePcmForMoonshine(pcm);
  assert.ok(prepared instanceof Float32Array, "Should return Float32Array");
  // Should be trimmed shorter than the original 32000 samples (with ~150ms pad on each side)
  assert.ok(prepared.length < 18000, `Expected trimmed length < 18000, got ${prepared.length}`);
  assert.ok(prepared.length >= 9600, `Expected speech preserved >= 9600, got ${prepared.length}`);

  let peak = 0;
  for (let i = 0; i < prepared.length; i++) {
    const abs = Math.abs(prepared[i]);
    if (abs > peak) peak = abs;
  }
  // Quiet 0.04 peak should be boosted (capped at 8x => 0.32)
  assert.ok(peak >= 0.25 && peak <= 0.6, `Expected normalized peak in [0.25, 0.6], got ${peak}`);
});
