'use strict';
// The path a correction actually takes: batch -> manifest -> load mask -> paint
// -> re-censor -> save -> dataset. Mirrors what the IPC handlers do.
const assert = require('assert');
const fsp = require('fs/promises');
const os = require('os');
const path = require('path');
const sharp = require('sharp');
const { runBatch } = require('../electron/batch');
const { defaultLabelConfig } = require('../electron/labels');
const { censor } = require('../electron/censor');
const review = require('../electron/review');
const { addSample, datasetStats } = require('../electron/dataset');

const { setUserModelsDir } = require('../electron/detector');
setUserModelsDir(path.join(os.homedir(), 'Library', 'Application Support', 'autocensor', 'models'));

const W = 700, H = 500;

(async () => {
  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), 'ac-rev-'));
  const inDir = path.join(tmp, 'in');
  const outDir = path.join(tmp, 'out');
  await fsp.mkdir(inDir, { recursive: true });

  await sharp({ create: { width: W, height: H, channels: 3, background: { r: 90, g: 140, b: 200 } } })
    .png()
    .toFile(path.join(inDir, 'shot.png'));

  await runBatch(
    {
      inputDir: inDir,
      outputDir: outDir,
      recursive: true,
      saveMasks: true,
      detectOptions: { models: ['anime-xl'], labelConfig: defaultLabelConfig(), tiling: 'never' },
      censorOptions: { mode: 'white', shape: 'contour' },
    },
    () => {}
  );

  const manifest = await review.loadManifest(outDir);
  assert.ok(manifest, 'manifest must be written');
  assert.strictEqual(manifest.entries.length, 1);
  const entry = manifest.entries[0];
  assert.strictEqual(entry.censored, false, 'nothing to detect in a flat image');

  // The detector found nothing, so the union mask exists but is empty.
  const unionFile = review.maskPath(outDir, entry.key, 'union');
  const mask = await review.readMask(unionFile, W, H);
  assert.ok(!mask.some((v) => v), 'auto mask should be empty here');

  // Paint a region by hand, exactly as the brush does.
  for (let y = 120; y < 260; y++) for (let x = 200; x < 380; x++) mask[y * W + x] = 255;

  const src = await fsp.readFile(entry.source);
  const result = await censor(src, { width: W, height: H, detections: [], mask }, {
    mode: 'white',
    shape: 'contour',
    dilateRadius: 2,
    featherRadius: 4,
    scaleWithResolution: false,
  });
  assert.ok(result.changed, 'hand-painted mask must produce a censored image');
  await fsp.writeFile(entry.dest, result.buffer);
  await review.writeMask(unionFile, mask, W, H);
  await review.writeMask(review.maskPath(outDir, entry.key, 'nipple'), mask, W, H);

  entry.labels = ['nipple'];
  entry.corrected = true;
  entry.censored = true;
  await review.saveManifest(outDir, manifest);

  // The saved output really is white where painted and untouched outside.
  const { data } = await sharp(entry.dest).raw().toBuffer({ resolveWithObject: true });
  const px = (x, y) => [data[(y * W + x) * 3], data[(y * W + x) * 3 + 1], data[(y * W + x) * 3 + 2]];
  assert.deepStrictEqual(px(290, 190), [255, 255, 255], 'painted area must be censored');
  assert.deepStrictEqual(px(50, 50), [90, 140, 200], 'untouched area must be unchanged');

  // Reloading gives back what was saved.
  const again = await review.loadManifest(outDir);
  assert.strictEqual(again.entries[0].corrected, true);
  assert.deepStrictEqual(again.entries[0].labels, ['nipple']);
  const reread = await review.readMask(unionFile, W, H);
  assert.deepStrictEqual(Array.from(reread.slice(0, 10)), Array.from(mask.slice(0, 10)));

  const dataset = path.join(tmp, 'dataset');
  const info = await addSample({
    datasetDir: dataset,
    sourcePath: entry.source,
    width: W,
    height: H,
    classMasks: { nipple: reread },
  });
  assert.strictEqual(info.polygons, 1);
  const stats = await datasetStats(dataset);
  assert.deepStrictEqual(stats.classes, { nipple: 1 });

  console.log(`PASS review flow: corrected 1 file, dataset ${stats.samples} sample / ${stats.polygons} polygon`);
  await fsp.rm(tmp, { recursive: true, force: true });
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
