'use strict';
// A brush stroke has to survive the whole trip: mask -> polygon -> YOLO label
// -> rasterised again. If that loses shape, the fine-tuning data is wrong.
const assert = require('assert');
const fs = require('fs');
const fsp = require('fs/promises');
const os = require('os');
const path = require('path');
const sharp = require('sharp');
const { addSample, datasetStats } = require('../electron/dataset');
const { writeMask, readMask } = require('../electron/review');
const { CLASSES } = require('../electron/labels');

const W = 500, H = 400;

function rasterise(polys, w, h) {
  const m = new Uint8Array(w * h);
  for (const poly of polys) {
    let minY = h, maxY = 0;
    for (const [, y] of poly) {
      minY = Math.min(minY, Math.floor(y));
      maxY = Math.max(maxY, Math.ceil(y));
    }
    for (let y = Math.max(0, minY); y <= Math.min(h - 1, maxY); y++) {
      const xs = [];
      for (let i = 0; i < poly.length; i++) {
        const [x1, y1] = poly[i];
        const [x2, y2] = poly[(i + 1) % poly.length];
        if (y1 === y2) continue;
        const yy = y + 0.5;
        if (yy < Math.min(y1, y2) || yy >= Math.max(y1, y2)) continue;
        xs.push(x1 + ((yy - y1) / (y2 - y1)) * (x2 - x1));
      }
      xs.sort((a, b) => a - b);
      for (let i = 0; i + 1 < xs.length; i += 2) {
        for (let x = Math.ceil(xs[i]); x < xs[i + 1]; x++) {
          if (x >= 0 && x < w) m[y * w + x] = 255;
        }
      }
    }
  }
  return m;
}

function iou(a, b) {
  let inter = 0, union = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] > 0, y = b[i] > 0;
    if (x && y) inter++;
    if (x || y) union++;
  }
  return union ? inter / union : 1;
}

(async () => {
  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), 'ac-ds-'));
  const src = path.join(tmp, 'src.png');
  await sharp({ create: { width: W, height: H, channels: 3, background: { r: 200, g: 170, b: 160 } } })
    .png()
    .toFile(src);

  // Two "brush strokes": a disc for nipple, a rounded blob for vagina.
  const nipple = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (Math.hypot(x - 140, y - 120) <= 44) nipple[y * W + x] = 255;
  }
  const vagina = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if ((x - 340) ** 2 / (60 * 60) + (y - 260) ** 2 / (35 * 35) <= 1) vagina[y * W + x] = 255;
  }

  // Mask PNG round-trip is how these reach disk between sessions.
  const mfile = path.join(tmp, 'm.png');
  await writeMask(mfile, nipple, W, H);
  const reread = await readMask(mfile, W, H);
  assert.strictEqual(iou(nipple, reread), 1, 'mask PNG round-trip must be lossless');

  const dataset = path.join(tmp, 'dataset');
  const info = await addSample({
    datasetDir: dataset,
    sourcePath: src,
    width: W,
    height: H,
    classMasks: { nipple, vagina },
  });

  assert.strictEqual(info.polygons, 2, `expected 2 polygons, got ${info.polygons}`);
  assert.ok(fs.existsSync(info.image), 'image must be copied into the dataset');

  const text = await fsp.readFile(info.label, 'utf8');
  const lines = text.trim().split('\n');
  assert.strictEqual(lines.length, 2);

  const byClass = {};
  for (const line of lines) {
    const nums = line.split(' ');
    const cls = CLASSES[Number(nums[0])];
    const coords = nums.slice(1).map(Number);
    assert.ok(coords.every((v) => v >= 0 && v <= 1), 'coordinates must be normalised');
    const poly = [];
    for (let i = 0; i < coords.length; i += 2) poly.push([coords[i] * W, coords[i + 1] * H]);
    byClass[cls] = poly;
  }
  assert.ok(byClass.nipple && byClass.vagina, `classes written: ${Object.keys(byClass)}`);

  const nIou = iou(nipple, rasterise([byClass.nipple], W, H));
  const vIou = iou(vagina, rasterise([byClass.vagina], W, H));
  assert.ok(nIou > 0.95, `nipple polygon IoU ${nIou.toFixed(3)}`);
  assert.ok(vIou > 0.95, `vagina polygon IoU ${vIou.toFixed(3)}`);

  const yaml = await fsp.readFile(path.join(dataset, 'data.yaml'), 'utf8');
  assert.ok(yaml.includes(`${CLASSES.indexOf('nipple')}: nipple`), 'data.yaml must map class ids');

  const stats = await datasetStats(dataset);
  assert.strictEqual(stats.samples, 1);
  assert.strictEqual(stats.polygons, 2);

  // Re-adding the same source overwrites rather than duplicating.
  await addSample({ datasetDir: dataset, sourcePath: src, width: W, height: H, classMasks: { nipple } });
  const stats2 = await datasetStats(dataset);
  assert.strictEqual(stats2.samples, 1, 'same source must not create a second sample');
  assert.strictEqual(stats2.polygons, 1);

  // An image with nothing painted is still a valid negative sample.
  const src2 = path.join(tmp, 'clean.png');
  await sharp({ create: { width: W, height: H, channels: 3, background: { r: 10, g: 10, b: 10 } } })
    .png()
    .toFile(src2);
  const empty = await addSample({ datasetDir: dataset, sourcePath: src2, width: W, height: H, classMasks: {} });
  assert.strictEqual(empty.polygons, 0);
  assert.strictEqual(await fsp.readFile(empty.label, 'utf8'), '');

  console.log(
    `PASS dataset: nipple IoU ${nIou.toFixed(3)}, vagina IoU ${vIou.toFixed(3)}, ` +
      `points ${byClass.nipple.length}/${byClass.vagina.length}`
  );
  await fsp.rm(tmp, { recursive: true, force: true });
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
