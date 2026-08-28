const sharp = require('sharp');
const { detect, availableModels } = require('../electron/detector');
const { defaultLabelConfig } = require('../electron/labels');

(async () => {
  const img = await sharp({
    create: { width: 1600, height: 2200, channels: 3, background: { r: 224, g: 200, b: 188 } },
  }).png().toBuffer();

  for (const m of availableModels()) {
    const t = Date.now();
    const d = await detect(img, { models: [m.key], labelConfig: defaultLabelConfig(), tiling: 'auto' });
    console.log(`${m.key.padEnd(14)} ${((Date.now() - t) / 1000).toFixed(1)}s  dets=${d.detections.length}`);
  }

  const t = Date.now();
  const d = await detect(img, {
    models: ['anime-medium', 'ntd11', 'wenaka'],
    labelConfig: defaultLabelConfig(),
    tiling: 'auto',
  });
  console.log(`ENSEMBLE(3)    ${((Date.now() - t) / 1000).toFixed(1)}s  dets=${d.detections.length}`);
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
