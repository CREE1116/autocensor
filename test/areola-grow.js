'use strict';
// Region growing must follow the areola's colour edge rather than a fixed
// radius, and must fall back to a geometric expansion when there is no signal.
const assert = require('assert');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

const SIZE = 640, NC = 7, NCOEF = 32, NBOX = 8400, PW = 160, PH = 160;
const W = 1000, H = 700, CX = 400, CY = 300;
const NIPPLE_R = 18;

const SKIN = [246, 216, 200];
const NIPPLE = [176, 108, 104];

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
    put(2, 2 * NIPPLE_R * scale); put(3, 2 * NIPPLE_R * scale);
    put(4 + 1, 0.9);            // class 1 = nipple
    put(4 + NC + 0, 1.0);

    const proto = new Float32Array(NCOEF * PH * PW).fill(-10);
    const gx = PW / SIZE, gy = PH / SIZE;
    for (let py = 0; py < PH; py++) for (let px = 0; px < PW; px++) {
      const dx = px / gx - toIX(CX), dy = py / gy - toIY(CY);
      if (Math.hypot(dx, dy) <= NIPPLE_R * scale) proto[py * PW + px] = 10;
    }
    return {
      output0: new ort.Tensor('float32', pred, [1, 4 + NC + NCOEF, NBOX]),
      output1: new ort.Tensor('float32', proto, [1, NCOEF, PH, PW]),
    };
  },
});

function makeImage(areolaR, areolaColor) {
  const buf = Buffer.alloc(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const d = Math.hypot(x - CX, y - CY);
    const c = d <= NIPPLE_R ? NIPPLE : d <= areolaR ? areolaColor : SKIN;
    const i = (y * W + x) * 3;
    buf[i] = c[0]; buf[i + 1] = c[1]; buf[i + 2] = c[2];
  }
  return sharp(buf, { raw: { width: W, height: H, channels: 3 } }).png().toBuffer();
}

(async () => {
  const { detect } = require('../electron/detector');
  const { defaultLabelConfig } = require('../electron/labels');
  const { growAreola } = require('../electron/colorgrow');

  const run = async (img) => {
    const det = await detect(img, {
      model: 'anime-nano', labelConfig: defaultLabelConfig(), tiling: 'never',
    });
    let n = 0;
    for (let i = 0; i < det.mask.length; i++) if (det.mask[i]) n++;
    const on = (dx, dy) => det.mask[(CY + dy) * W + (CX + dx)] === 255;
    return { n, on };
  };

  // 1. A distinctly coloured areola is followed to its own edge, at two sizes.
  for (const areolaR of [46, 70]) {
    const { n, on } = await run(await makeImage(areolaR, [214, 154, 146]));
    // The fit is deliberately inflated slightly; under-covering is the costlier error.
    const inflate = defaultLabelConfig().nipple.areola.inflate;
    const want = Math.PI * areolaR * areolaR * inflate * inflate;
    assert.ok(Math.abs(n - want) / want < 0.08, `areolaR=${areolaR}: area ${n} vs ${want | 0}`);
    assert.ok(on(areolaR - 4, 0), `areolaR=${areolaR}: inside edge should be covered`);
    assert.ok(!on(Math.round(areolaR * 1.2), 0), `areolaR=${areolaR}: skin past the edge must stay clear`);
    console.log(`  areolaR=${areolaR} -> covered ${n} (circle ${want | 0})`);
  }

  // The two sizes must differ, i.e. the result tracks colour and not a constant.
  const small = await run(await makeImage(46, [214, 154, 146]));
  const large = await run(await makeImage(70, [214, 154, 146]));
  assert.ok(large.n > small.n * 1.8, `grow did not track areola size: ${small.n} vs ${large.n}`);

  // 2. Areola nearly the same colour as skin -> no signal -> fallbackExpand 2.6.
  const flat = await run(await makeImage(46, [244, 214, 199]));
  const fallback = Math.PI * Math.pow(NIPPLE_R * 2.6, 2);
  assert.ok(
    Math.abs(flat.n - fallback) / fallback < 0.1,
    `low-contrast fallback area ${flat.n} vs ${fallback | 0}`
  );
  console.log(`  low contrast -> fallback covered ${flat.n} (2.6x circle ${fallback | 0})`);

  // 3. The leak guard rejects a grow that swallows the search disc.
  const img = await makeImage(46, [214, 154, 146]);
  const { data: rgb } = await sharp(img).raw().toBuffer({ resolveWithObject: true });
  const box = [CX - 90, CY - 90, CX + 90, CY + 90];
  const bw = box[2] - box[0];
  const seed = new Uint8Array(bw * bw);
  for (let y = box[1]; y < box[3]; y++) for (let x = box[0]; x < box[2]; x++) {
    if (Math.hypot(x - CX, y - CY) <= NIPPLE_R) seed[(y - box[1]) * bw + (x - box[0])] = 255;
  }
  const before = Uint8Array.from(seed);
  const rejected = growAreola(
    seed, box, rgb, W,
    { maxScale: 5, tolerance: 0.35, minContrast: 16, leakLimit: 0.05 },
    { x: CX, y: CY }, NIPPLE_R
  );
  assert.strictEqual(rejected, null, 'leak guard should reject');
  assert.deepStrictEqual(seed, before, 'rejected grow must not mutate the mask');

  console.log('PASS areola colour growing');
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
