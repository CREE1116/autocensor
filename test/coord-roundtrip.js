'use strict';
// Feed the detector a fabricated model output describing a box at a known
// position in the ORIGINAL image, then assert the produced mask lands there.
const assert = require('assert');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

const SIZE = 640;
const NC = 7, NCOEF = 32, NBOX = 8400, PW = 160, PH = 160;

// Target rectangle in original-image coordinates.
const TARGET = { x0: 300, y0: 100, x1: 460, y1: 260 };
const W = 1000, H = 700;

let scaleUsed, padXUsed, padYUsed;

ort.InferenceSession.create = async () => ({
  inputNames: ['images'],
  outputNames: ['output0', 'output1'],
  run: async () => {
    const scale = Math.min(SIZE / W, SIZE / H);
    const nw = Math.round(W * scale), nh = Math.round(H * scale);
    const padX = Math.floor((SIZE - nw) / 2), padY = Math.floor((SIZE - nh) / 2);
    scaleUsed = scale; padXUsed = padX; padYUsed = padY;

    const ix0 = TARGET.x0 * scale + padX, iy0 = TARGET.y0 * scale + padY;
    const ix1 = TARGET.x1 * scale + padX, iy1 = TARGET.y1 * scale + padY;

    const pred = new Float32Array((4 + NC + NCOEF) * NBOX);
    const put = (ch, v) => { pred[ch * NBOX + 0] = v; };
    put(0, (ix0 + ix1) / 2);
    put(1, (iy0 + iy1) / 2);
    put(2, ix1 - ix0);
    put(3, iy1 - iy0);
    put(4 + 2, 0.9);            // class 2 = penis
    put(4 + NC + 0, 1.0);       // single active mask coefficient

    // Proto 0 is +10 inside the target region (sigmoid ~ 1) and -10 outside.
    const proto = new Float32Array(NCOEF * PH * PW).fill(-10);
    const gx = PW / SIZE, gy = PH / SIZE;
    for (let py = 0; py < PH; py++) {
      for (let px = 0; px < PW; px++) {
        const xi = px / gx, yi = py / gy;
        if (xi >= ix0 && xi <= ix1 && yi >= iy0 && yi <= iy1) proto[py * PW + px] = 10;
      }
    }
    return {
      output0: new ort.Tensor('float32', pred, [1, 4 + NC + NCOEF, NBOX]),
      output1: new ort.Tensor('float32', proto, [1, NCOEF, PH, PW]),
    };
  },
});

(async () => {
  const { detect } = require('../electron/detector');
  const { defaultLabelConfig } = require('../electron/labels');
  const cfg = defaultLabelConfig();
  cfg.penis.expand = 1.0;

  const img = await sharp({
    create: { width: W, height: H, channels: 3, background: { r: 40, g: 60, b: 90 } },
  }).png().toBuffer();

  const det = await detect(img, { model: 'anime-nano', labelConfig: cfg, tiling: 'never' });

  assert.strictEqual(det.detections.length, 1, 'one detection expected');
  const d = det.detections[0];
  assert.strictEqual(d.label, 'penis');
  for (const [k, want] of [['0', TARGET.x0], ['1', TARGET.y0], ['2', TARGET.x1], ['3', TARGET.y1]]) {
    assert.ok(Math.abs(d.box[k] - want) <= 2, `box[${k}]=${d.box[k]} want ~${want}`);
  }

  // Mask must cover the interior and stay out of a margin around the outside.
  const at = (x, y) => det.mask[y * W + x];
  const inside = [[320, 120], [380, 180], [440, 240]];
  const outside = [[280, 180], [480, 180], [380, 80], [380, 280], [10, 10]];
  for (const [x, y] of inside) assert.strictEqual(at(x, y), 255, `expected mask at ${x},${y}`);
  for (const [x, y] of outside) assert.strictEqual(at(x, y), 0, `expected no mask at ${x},${y}`);

  let covered = 0;
  for (let i = 0; i < det.mask.length; i++) if (det.mask[i]) covered++;
  const area = (TARGET.x1 - TARGET.x0) * (TARGET.y1 - TARGET.y0);
  assert.ok(Math.abs(covered - area) / area < 0.05, `covered=${covered} vs area=${area}`);

  // The censored output must actually be white inside and untouched outside.
  const { censor } = require('../electron/censor');
  const out = await censor(img, det, { mode: 'white', shape: 'contour', dilateRadius: 0, featherRadius: 0 });
  const { data } = await sharp(out.buffer).raw().toBuffer({ resolveWithObject: true });
  const px = (x, y) => [data[(y * W + x) * 3], data[(y * W + x) * 3 + 1], data[(y * W + x) * 3 + 2]];
  assert.deepStrictEqual(px(380, 180), [255, 255, 255], 'inside should be white');
  assert.deepStrictEqual(px(50, 50), [40, 60, 90], 'outside should be untouched');

  console.log(`PASS scale=${scaleUsed.toFixed(4)} pad=(${padXUsed},${padYUsed}) box=${d.box} covered=${covered}/${area}`);
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
