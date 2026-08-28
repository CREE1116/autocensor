'use strict';
// Fixed-scale expansion vs colour growing, on a small and a large areola.
const sharp = require('sharp');
const ort = require('onnxruntime-node');

const SIZE = 640, NC = 7, NCOEF = 32, NBOX = 8400, PW = 160, PH = 160;
const W = 300, H = 300, CX = 150, CY = 150, NIPPLE_R = 16;
const SKIN = [246, 216, 200], AREOLA = [214, 154, 146], NIPPLE = [176, 108, 104];

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
    put(4 + 1, 0.9); put(4 + NC + 0, 1.0);
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

function makeImage(areolaR) {
  const buf = Buffer.alloc(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const d = Math.hypot(x - CX, y - CY);
    const c = d <= NIPPLE_R ? NIPPLE : d <= areolaR ? AREOLA : SKIN;
    const i = (y * W + x) * 3;
    buf[i] = c[0]; buf[i + 1] = c[1]; buf[i + 2] = c[2];
  }
  return sharp(buf, { raw: { width: W, height: H, channels: 3 } }).png().toBuffer();
}

(async () => {
  const { detect } = require('../electron/detector');
  const { censor } = require('../electron/censor');
  const { defaultLabelConfig } = require('../electron/labels');

  const render = async (img, useGrow) => {
    const cfg = defaultLabelConfig();
    cfg.nipple.areola.method = useGrow ? 'ray' : 'off';
    if (!useGrow) cfg.nipple.expand = 2.6;
    const det = await detect(img, { model: 'anime-nano', labelConfig: cfg, tiling: 'never' });
    const r = await censor(img, det, {
      mode: 'white', shape: 'contour',
      dilateRadius: 4, featherRadius: 10, edgeGamma: 0.7, scaleWithResolution: false,
    });
    return r.buffer || img;
  };

  const rows = [];
  for (const areolaR of [26, 64]) {
    const img = await makeImage(areolaR);
    rows.push([img, await render(img, false), await render(img, true)]);
  }

  const gap = 10;
  const out = await sharp({
    create: {
      width: W * 3 + gap * 4, height: H * 2 + gap * 3,
      channels: 3, background: { r: 20, g: 22, b: 26 },
    },
  })
    .composite(
      rows.flatMap((row, r) =>
        row.map((input, c) => ({ input, left: gap + c * (W + gap), top: gap + r * (H + gap) }))
      )
    )
    .png()
    .toBuffer();

  await sharp(out).toFile('test/visual-compare.png');
  console.log('wrote test/visual-compare.png');
  console.log('columns: source | fixed 2.6x | colour grow');
  console.log('rows: small areola (r=26) | large areola (r=64), nipple r=16');
})().catch((e) => { console.error(e); process.exit(1); });
