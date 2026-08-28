'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const sharp = require('sharp');

const SIDECAR = '_autocensor';

function sidecarDir(outputDir) {
  return path.join(outputDir, SIDECAR);
}

function manifestPath(outputDir) {
  return path.join(sidecarDir(outputDir), 'manifest.json');
}

/** One flat, filesystem-safe key per source image. */
function maskKey(inputDir, file) {
  return path
    .relative(inputDir, file)
    .replace(/[\\/]/g, '__')
    .replace(/[^A-Za-z0-9._-]/g, '_');
}

function maskPath(outputDir, key, label) {
  const safeLabel = label.replace(/[^A-Za-z0-9-]/g, '_');
  return path.join(sidecarDir(outputDir), 'masks', `${key}__${safeLabel}.png`);
}

async function writeMask(file, mask, width, height) {
  await fsp.mkdir(path.dirname(file), { recursive: true });
  await sharp(Buffer.from(mask), { raw: { width, height, channels: 1 } })
    .toColourspace('b-w')
    .png({ compressionLevel: 9 })
    .toFile(file);
}

async function readMask(file, width, height) {
  const { data, info } = await sharp(file)
    .toColourspace('b-w')
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.width !== width || info.height !== height || info.channels !== 1) {
    throw new Error(
      `mask ${path.basename(file)} is ${info.width}x${info.height}x${info.channels}, expected ${width}x${height}x1`
    );
  }
  return data;
}

async function loadManifest(outputDir) {
  const file = manifestPath(outputDir);
  if (!fs.existsSync(file)) return null;
  return JSON.parse(await fsp.readFile(file, 'utf8'));
}

async function saveManifest(outputDir, manifest) {
  await fsp.mkdir(sidecarDir(outputDir), { recursive: true });
  await fsp.writeFile(manifestPath(outputDir), JSON.stringify(manifest, null, 2));
}

module.exports = {
  SIDECAR,
  sidecarDir,
  manifestPath,
  maskKey,
  maskPath,
  writeMask,
  readMask,
  loadManifest,
  saveManifest,
};
