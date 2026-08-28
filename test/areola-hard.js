'use strict';
// The cases that break a flood fill: an areola seen at an angle, lighting that
// varies across the body, a hair strand running outward, a specular highlight.
const assert = require('assert');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

const SIZE = 640, NC = 7, NCOEF = 32, NBOX = 8400, PW = 160, PH = 160;
const W = 800, H = 600, CX = 400, CY = 300, NIPPLE_R = 16;
const AX = 62, AY = 34, PHI = (30 * Math.PI) / 180; // areola ellipse, rotated

const NIPPLE = [176, 108, 104];
const AREOLA = [214, 154, 146];
const SKIN = [246, 216, 200];
const HAIR = [58, 44, 52];
const SPEC = [252, 240, 232];

ort.InferenceSession.create = async () => ({
  inputNames: ['images'],
  outputNames: ['output0', 'output1'],
  run: async () => {
    const scale = Math.min(SIZE / W, SIZE / H);
    const nw = Math.round(W * scale), nh = Math.round(H * scale);
    const padX = Math.floor((SIZE - nw) / 2), padY = Math.floor((SIZE - nh) / 2);
    const toIX = (x) => x * scale + padX, toIY = (y) => y * scale + padY;
    const pred = new Float32Array((4 + NC + NCOEF) * NBOX);
    const put = (c, v) => { pred[c * NBOX] = v; };
    put(0, toIX(CX)); put(1, toIY(CY));
    put(2, 2 * NIPPLE_R * scale); put(3, 2 * NIPPLE_R * scale);
    put(4 + 1, 0.9); put(4 + NC + 0, 1.0);   // class 1 = nipple
    const proto = new Float32Array(NCOEF * PH * PW).fill(-10);
    const gx = PW / SIZE, gy = PH / SIZE;
    for (let py = 0; py < PH; py++) for (let px = 0; px < PW; px++) {
      if (Math.hypot(px / gx - toIX(CX), py / gy - toIY(CY)) <= NIPPLE_R * scale) proto[py * PW + px] = 10;
    }
    return {
      output0: new ort.Tensor('float32', pred, [1, 4 + NC + NCOEF, NBOX]),
      output1: new ort.Tensor('float32', proto, [1, NCOEF, PH, PW]),
    };
  },
});

const cosP = Math.cos(-PHI), sinP = Math.sin(-PHI);
function inAreola(x, y) {
  const dx = x - CX, dy = y - CY;
  const u = dx * cosP - dy * sinP;
  const v = dx * sinP + dy * cosP;
  return (u / AX) ** 2 + (v / AY) ** 2 <= 1;
}

// A hair strand leaving the areola at 200 degrees, reaching well past it.
const HAIR_ANG = (200 * Math.PI) / 180;
function onHair(x, y) {
  const dx = x - CX, dy = y - CY;
  const along = dx * Math.cos(HAIR_ANG) + dy * Math.sin(HAIR_ANG);
  const across = -dx * Math.sin(HAIR_ANG) + dy * Math.cos(HAIR_ANG);
  return along > 10 && along < 78 && Math.abs(across) < 3.5;
}

function makeImage() {
  const buf = Buffer.alloc(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    // Lighting ramp across the whole image: a global skin reference would drift.
    const shade = ((x - CX) / W) * 60 - ((y - CY) / H) * 30;
    let c;
    if (Math.hypot(x - CX, y - CY) <= NIPPLE_R) c = NIPPLE;
    else if (onHair(x, y)) c = HAIR;
    else if (Math.hypot(x - (CX + 30), y - (CY - 14)) <= 7) c = SPEC; // highlight on the areola
    else if (inAreola(x, y)) c = AREOLA;
    else c = SKIN;
    const i = (y * W + x) * 3;
    for (let k = 0; k < 3; k++) buf[i + k] = Math.max(0, Math.min(255, c[k] + shade));
  }
  return sharp(buf, { raw: { width: W, height: H, channels: 3 } }).png().toBuffer();
}

(async () => {
  const { detect } = require('../electron/detector');
  const { defaultLabelConfig } = require('../electron/labels');
  const img = await makeImage();

  const run = async (method) => {
    const cfg = defaultLabelConfig();
    cfg.nipple.areola.method = method;
    const det = await detect(img, { model: 'anime-nano', labelConfig: cfg, tiling: 'never' });
    let n = 0;
    for (let i = 0; i < det.mask.length; i++) if (det.mask[i]) n++;
    const on = (x, y) => det.mask[y * W + x] === 255;
    return { n, on };
  };

  const ray = await run('ray');
  const inflate = defaultLabelConfig().nipple.areola.inflate;
  const want = Math.PI * AX * AY * inflate * inflate;
  console.log(`  ray  covered ${ray.n}  (ellipse x inflate ${want | 0})`);
  assert.ok(Math.abs(ray.n - want) / want < 0.15, `ray area ${ray.n} vs ${want | 0}`);

  // Covers the areola out to its own edge along both axes, including under the
  // specular highlight, and does not follow the hair strand outward.
  const edge = (frac, ang) => [
    Math.round(CX + Math.cos(ang) * AX * frac * cosP + Math.sin(ang) * AY * frac * sinP),
    Math.round(CY - Math.cos(ang) * AX * frac * sinP + Math.sin(ang) * AY * frac * cosP),
  ];
  for (const ang of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2, 0.7, 2.4]) {
    const [ix, iy] = edge(0.9, ang);
    assert.ok(ray.on(ix, iy), `inside edge at angle ${ang.toFixed(2)} must be covered`);
    const [ox, oy] = edge(1.4, ang);
    assert.ok(!ray.on(ox, oy), `skin at angle ${ang.toFixed(2)} must stay clear`);
  }
  assert.ok(ray.on(CX + 30, CY - 14), 'highlight inside the areola must still be covered');

  const hairX = Math.round(CX + Math.cos(HAIR_ANG) * 72);
  const hairY = Math.round(CY + Math.sin(HAIR_ANG) * 72);
  assert.ok(!ray.on(hairX, hairY), 'must not follow the hair strand outward');

  // The flood fill is the thing this replaces: it leaks down the strand.
  const flood = await run('flood');
  const leaks = flood.on(hairX, hairY);
  console.log(`  flood covered ${flood.n}, follows hair strand: ${leaks}`);

  console.log('PASS ray fit on ellipse + shading + hair + highlight');
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
