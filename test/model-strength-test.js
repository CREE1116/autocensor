'use strict';

const assert = require('assert');
const path = require('path');
const sharp = require('sharp');
const detector = require('../electron/detector');
const { defaultLabelConfig } = require('../electron/labels');

const W = 320;
const H = 320;

(async () => {
  console.log('--- testing model strength feature ---');

  // Create a synthetic image buffer
  const buf = Buffer.alloc(W * H * 3, 200);
  // Add a dark spot in the middle
  for (let y = 140; y < 180; y++) {
    for (let x = 140; x < 180; x++) {
      const idx = (y * W + x) * 3;
      buf[idx] = 160;
      buf[idx + 1] = 90;
      buf[idx + 2] = 80;
    }
  }
  const imgPng = await sharp(buf, { raw: { width: W, height: H, channels: 3 } }).png().toBuffer();

  const cfg = defaultLabelConfig();
  const os = require('os');
  detector.setUserModelsDir(path.join(os.homedir(), 'Library', 'Application Support', 'autocensor', 'models'));

  // Run with default strength (1.0)
  const resDefault = await detector.detect(imgPng, {
    models: ['anime-xl'],
    modelConfigs: { 'anime-xl': { strength: 1.0 } },
    labelConfig: cfg,
    tiling: 'never',
  });
  assert.ok(resDefault);
  assert.strictEqual(resDefault.width, W);
  assert.strictEqual(resDefault.height, H);
  console.log('PASS detect with strength 1.0');

  // Run with high strength (2.0x -> lower threshold, more sensitive)
  const resHigh = await detector.detect(imgPng, {
    models: ['anime-xl'],
    modelConfigs: { 'anime-xl': { strength: 2.0 } },
    labelConfig: cfg,
    tiling: 'never',
  });
  assert.ok(resHigh);
  console.log('PASS detect with strength 2.0 (sensitive)');

  // Run with low strength (0.5x -> higher threshold, stricter)
  const resLow = await detector.detect(imgPng, {
    models: ['anime-xl'],
    modelConfigs: { 'anime-xl': { strength: 0.5 } },
    labelConfig: cfg,
    tiling: 'never',
  });
  assert.ok(resLow);
  console.log('PASS detect with strength 0.5 (strict)');

  console.log('ALL MODEL STRENGTH TESTS PASSED');
})().catch((err) => {
  console.error('FAIL model-strength-test:', err);
  process.exit(1);
});
