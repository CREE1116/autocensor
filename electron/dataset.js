'use strict';

const fsp = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const { maskToPolygons, polygonToYolo } = require('./contour');
const { CLASSES } = require('./labels');

/**
 * Corrections made with the brush are exactly the cases the detector got wrong,
 * which makes them the most useful training data available. They are written in
 * ultralytics YOLO segmentation layout so `yolo train` can consume the folder
 * directly.
 */
async function ensureDataset(datasetDir) {
  await fsp.mkdir(path.join(datasetDir, 'images', 'train'), { recursive: true });
  await fsp.mkdir(path.join(datasetDir, 'labels', 'train'), { recursive: true });

  const names = CLASSES.map((c, i) => `  ${i}: ${c}`).join('\n');
  const yaml = `# written by AutoCensor
path: ${datasetDir}
train: images/train
val: images/train

names:
${names}
`;
  await fsp.writeFile(path.join(datasetDir, 'data.yaml'), yaml);
}

function sampleName(sourcePath) {
  const stem = path
    .basename(sourcePath, path.extname(sourcePath))
    .replace(/[^A-Za-z0-9._-]/g, '_')
    .slice(0, 60);
  const hash = crypto.createHash('sha1').update(sourcePath).digest('hex').slice(0, 8);
  return `${stem}_${hash}`;
}

/**
 * @param classMasks { [label]: Uint8Array(width*height) }
 * @returns { name, image, label, polygons, classes }
 */
async function addSample(opts) {
  const { datasetDir, sourcePath, width, height, classMasks, minArea = 40 } = opts;
  await ensureDataset(datasetDir);

  const name = sampleName(sourcePath);
  const imageOut = path.join(datasetDir, 'images', 'train', `${name}.png`);
  const labelOut = path.join(datasetDir, 'labels', 'train', `${name}.txt`);

  // Always PNG so the pixel data matches the mask the polygons came from.
  await sharp(sourcePath).removeAlpha().toColourspace('srgb').png().toFile(imageOut);

  const lines = [];
  const counts = {};
  for (const [label, mask] of Object.entries(classMasks)) {
    const classIndex = CLASSES.indexOf(label);
    if (classIndex < 0) continue;
    const polys = maskToPolygons(mask, width, height, { minArea });
    for (const poly of polys) lines.push(polygonToYolo(poly, classIndex, width, height));
    if (polys.length) counts[label] = polys.length;
  }

  // An image with nothing to censor is a valid negative sample, so an empty
  // label file is written rather than skipped.
  await fsp.writeFile(labelOut, lines.length ? `${lines.join('\n')}\n` : '');

  return { name, image: imageOut, label: labelOut, polygons: lines.length, classes: counts };
}

async function datasetStats(datasetDir) {
  const dir = path.join(datasetDir, 'labels', 'train');
  let files = [];
  try {
    files = await fsp.readdir(dir);
  } catch {
    return { samples: 0, polygons: 0, classes: {} };
  }
  let polygons = 0;
  const classes = {};
  for (const f of files) {
    if (!f.endsWith('.txt')) continue;
    const text = await fsp.readFile(path.join(dir, f), 'utf8');
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      polygons++;
      const idx = Number(line.split(' ')[0]);
      const label = CLASSES[idx] || String(idx);
      classes[label] = (classes[label] || 0) + 1;
    }
  }
  return { samples: files.filter((f) => f.endsWith('.txt')).length, polygons, classes };
}

module.exports = { addSample, ensureDataset, datasetStats, sampleName };
