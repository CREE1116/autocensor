'use strict';
// The prototype mask is a circle; `expand` must scale that circle about its
// centre, not merely widen the bounding box.
const assert = require('assert');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

const SIZE = 640, NC = 7, NCOEF = 32, NBOX = 8400, PW = 160, PH = 160;
const W = 1000, H = 700;
const CX = 400, CY = 300, R = 60; // circle in original-image coordinates

ort.InferenceSession.create = async () => ({
  inputNames: ['images'],
  outputNames: ['output0', 'output1'],
  run: async () => {
    const scale = Math.min(SIZE / W, SIZE / H);
    const nw = Math.round(W * scale), nh = Math.round(H * scale);
    const padX = Math.floor((SIZE - nw) / 2), padY = Math.floor((SIZE - nh) / 2);
    const toIX = (x) => x * scale + padX, toIY = (y) => y * scale + padY;

    const pred = new Float32Array((4 + NC + NCOEF) * NBOX);
    const put = (ch, v) => { pred[ch * NBOX] = v; };
    put(0, toIX(CX)); put(1, toIY(CY));
    put(2, 2 * R * scale); put(3, 2 * R * scale);
    put(4 + 1, 0.9);          // class 1 = nipple
    put(4 + NC + 0, 1.0);

    const proto = new Float32Array(NCOEF * PH * PW).fill(-10);
    const gx = PW / SIZE, gy = PH / SIZE;
    for (let py = 0; py < PH; py++) for (let px = 0; px < PW; px++) {
      const xi = px / gx, yi = py / gy;
      const dx = xi - toIX(CX), dy = yi - toIY(CY);
      if (Math.hypot(dx, dy) <= R * scale) proto[py * PW + px] = 10;
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
  const img = await sharp({
    create: { width: W, height: H, channels: 3, background: { r: 30, g: 30, b: 30 } },
  }).png().toBuffer();

  const areaFor = async (expand) => {
    const cfg = defaultLabelConfig();
    cfg.nipple.expand = expand;
    cfg.nipple.areola.method = "off"; // measuring the contour scale alone
    const det = await detect(img, { model: 'anime-nano', labelConfig: cfg, tiling: 'never' });
    let n = 0;
    for (let i = 0; i < det.mask.length; i++) if (det.mask[i]) n++;
    return { n, det };
  };

  const base = await areaFor(1.0);
  const grown = await areaFor(2.5);

  const circle = Math.PI * R * R;
  assert.ok(Math.abs(base.n - circle) / circle < 0.05, `expand=1 area ${base.n} vs ${circle | 0}`);

  // Area scales with the square of the factor.
  const ratio = grown.n / base.n;
  assert.ok(Math.abs(ratio - 6.25) / 6.25 < 0.05, `expand=2.5 area ratio ${ratio.toFixed(2)}, want 6.25`);

  // And it is still a circle about the same centre, not a filled box.
  const m = grown.det.mask;
  const on = (x, y) => m[y * W + x] === 255;
  assert.ok(on(CX + 140, CY), 'should cover 2.5x radius horizontally');
  assert.ok(!on(CX + 160, CY), 'should stop past 2.5x radius');
  assert.ok(!on(CX + 105, CY - 105), 'corner of the box must stay empty');
  console.log(`PASS expand=1 area=${base.n} (circle ${circle | 0}), expand=2.5 ratio=${ratio.toFixed(2)}`);
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
