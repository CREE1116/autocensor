'use strict';
// End to end: build a tiny dataset, fine-tune, export, install, and detect with
// the resulting model. Needs a Python with ultralytics and network access for
// the base weights.
const assert = require('assert');
const fsp = require('fs/promises');
const os = require('os');
const path = require('path');
const sharp = require('sharp');
const train = require('../electron/train');
const { addSample } = require('../electron/dataset');
const detector = require('../electron/detector');
const { defaultLabelConfig } = require('../electron/labels');

const W = 320, H = 320;

async function makeSample(dir, i, datasetDir) {
  const cx = 90 + (i % 3) * 60;
  const cy = 100 + (i % 4) * 40;
  const r = 26 + (i % 3) * 6;

  const buf = Buffer.alloc(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const inside = Math.hypot(x - cx, y - cy) <= r;
    const k = (y * W + x) * 3;
    buf[k] = inside ? 180 : 240;
    buf[k + 1] = inside ? 110 : 210;
    buf[k + 2] = inside ? 106 : 196;
  }
  const src = path.join(dir, `s${i}.png`);
  await sharp(buf, { raw: { width: W, height: H, channels: 3 } }).png().toFile(src);

  const mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (Math.hypot(x - cx, y - cy) <= r) mask[y * W + x] = 255;
  }
  await addSample({ datasetDir, sourcePath: src, width: W, height: H, classMasks: { nipple: mask } });
  return { src, cx, cy, r };
}

(async () => {
  const python = await train.findPython();
  if (!python) {
    console.log('SKIP train-e2e: no python with ultralytics');
    return;
  }
  console.log(`  python ${python.bin} · ultralytics ${python.version} · torch ${python.torch}`);

  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), 'ac-train-'));
  const raw = path.join(tmp, 'raw');
  const datasetDir = path.join(tmp, 'dataset');
  const userModels = path.join(tmp, 'usermodels');
  await fsp.mkdir(raw, { recursive: true });

  const samples = [];
  for (let i = 0; i < 12; i++) samples.push(await makeSample(raw, i, datasetDir));

  const ready = await train.datasetReady(datasetDir);
  assert.ok(ready.ok, `dataset not ready: ${ready.reason}`);
  assert.strictEqual(ready.samples, 12);

  let sawEpoch = false;
  const r = await train.startTraining(
    {
      python: python.bin,
      datasetDir,
      base: 'yolo11n-seg',
      epochs: 2,
      imgsz: 320,
      batch: 4,
      device: 'cpu',
      runsDir: path.join(tmp, 'runs'),
      userModelsDir: userModels,
      label: 'e2e test model',
    },
    (ev) => {
      if (ev.type === 'log' && /Epoch|epochs completed/.test(ev.text)) sawEpoch = true;
      if (ev.type === 'error') console.log('  train error:', ev.message);
    }
  );

  assert.ok(r.ok, `training failed: ${r.message}`);
  assert.ok(sawEpoch, 'expected epoch progress in the log');

  const spec = r.model;
  assert.strictEqual(spec.task, 'segment');
  assert.strictEqual(spec.size, 320);
  assert.ok(spec.classes.includes('nipple'), `classes: ${spec.classes}`);
  assert.strictEqual(spec.map.nipple, 'nipple');

  const files = await fsp.readdir(userModels);
  assert.ok(files.includes(`${spec.key}.onnx`), `onnx not installed: ${files}`);
  assert.ok(files.includes(`${spec.key}.json`), `descriptor not installed: ${files}`);

  // The new model must be selectable and actually runnable, with no restart.
  detector.setUserModelsDir(userModels);
  const listed = detector.availableModels().find((m) => m.key === spec.key);
  assert.ok(listed, 'trained model must appear in availableModels()');
  assert.strictEqual(listed.custom, true);

  const cfg = defaultLabelConfig();
  cfg.nipple.areola.method = 'off';
  cfg.nipple.threshold = 0.05;
  const img = await fsp.readFile(samples[0].src);
  const det = await detector.detect(img, { models: [spec.key], labelConfig: cfg, tiling: 'never' });
  console.log(`  inference with trained model: ${det.detections.length} detections`);
  assert.ok(Array.isArray(det.detections), 'trained model must run through detect()');
  assert.strictEqual(det.mask.length, W * H);

  console.log(`PASS train e2e: ${spec.key} installed and runnable`);
  await fsp.rm(tmp, { recursive: true, force: true });
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
