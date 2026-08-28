'use strict';
const assert = require('assert');
const sharp = require('sharp');
const { censor } = require('../electron/censor');

(async () => {
  const W = 1000, H = 1000;
  const img = await sharp({
    create: { width: W, height: H, channels: 3, background: { r: 0, g: 0, b: 0 } },
  }).png().toBuffer();

  // A 200x200 solid square in the middle of the mask.
  const mask = new Uint8Array(W * H);
  for (let y = 400; y < 600; y++) for (let x = 400; x < 600; x++) mask[y * W + x] = 255;

  const out = await censor(img, { width: W, height: H, detections: [], mask }, {
    mode: 'white', shape: 'contour', dilateRadius: 4, featherRadius: 10, edgeGamma: 0.7,
  });
  const { data } = await sharp(out.buffer).raw().toBuffer({ resolveWithObject: true });
  const at = (x, y) => data[(y * W + x) * 3];

  // Centre fully white, far outside untouched, and a monotone ramp across the edge.
  assert.strictEqual(at(500, 500), 255, 'core must be fully opaque');
  assert.strictEqual(at(500, 100), 0, 'far outside must be untouched');

  const profile = [];
  for (let x = 570; x <= 650; x += 2) profile.push(at(x, 500));
  for (let i = 1; i < profile.length; i++) {
    assert.ok(profile[i] <= profile[i - 1] + 1, `ramp not monotone at ${i}: ${profile}`);
  }
  const mid = profile.filter((v) => v > 20 && v < 235).length;
  assert.ok(mid >= 5, `expected a gradual ramp, got ${mid} intermediate samples: ${profile}`);

  // Soft edge must reach beyond the original hard boundary (x=600).
  assert.ok(at(605, 500) > 40, `edge should extend outward, got ${at(605, 500)}`);
  console.log('PASS soft edge profile x=570..650:', profile.join(','));
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
