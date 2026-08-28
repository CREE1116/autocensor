'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const { detect } = require('./detector');
const { censor } = require('./censor');
const review = require('./review');

const IMAGE_EXT = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.avif']);

/**
 * Walked with an explicit queue rather than recursion: `out.push(...subtree)`
 * throws "Maximum call stack size exceeded" once a library passes ~125k files,
 * because a spread passes every element as a separate argument.
 */
async function listImages(dir, recursive) {
  const out = [];
  const queue = [dir];

  while (queue.length) {
    const current = queue.shift();
    const entries = await fsp.readdir(current, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(current, e.name);
      if (e.isDirectory()) {
        // Never re-ingest our own sidecar masks when an output folder is reused.
        if (e.name === review.SIDECAR) continue;
        if (recursive) queue.push(full);
      } else if (IMAGE_EXT.has(path.extname(e.name).toLowerCase())) {
        out.push(full);
      }
    }
  }
  return out.sort();
}

function outputPath(file, inputDir, outputDir, format, suffix) {
  const rel = path.relative(inputDir, file);
  const dir = path.join(outputDir, path.dirname(rel));
  const stem = path.basename(rel, path.extname(rel));
  const ext = format === 'keep' ? path.extname(rel) : `.${format === 'jpeg' ? 'jpg' : format}`;
  return path.join(dir, `${stem}${suffix}${ext}`);
}

/**
 * Process every image under inputDir. `emit(event)` receives
 * {type:'start'|'file'|'progress'|'done'|'error', ...} for the UI.
 */
async function runBatch(opts, emit, shouldCancel) {
  const {
    inputDir,
    outputDir,
    recursive = true,
    overwrite = false,
    copyUnchanged = true,
    suffix = '',
    format = 'keep',
    quality = 95,
    saveMasks = true,
    detectOptions = {},
    censorOptions = {},
  } = opts;

  const files = await listImages(inputDir, recursive);
  emit({ type: 'start', total: files.length });

  // The manifest is what the review grid reads back: which source produced which
  // output, what was found, and where the per-label masks live.
  const manifest = {
    version: 1,
    createdAt: new Date().toISOString(),
    inputDir,
    outputDir,
    censorOptions,
    detectOptions: { ...detectOptions, labelConfig: undefined },
    entries: [],
  };

  let processed = 0;
  let censored = 0;
  let skipped = 0;
  let failed = 0;

  for (const file of files) {
    if (shouldCancel && shouldCancel()) {
      if (saveMasks && manifest.entries.length) await review.saveManifest(outputDir, manifest);
      emit({ type: 'cancelled', processed, censored, skipped, failed, outputDir });
      return { processed, censored, skipped, failed, cancelled: true };
    }

    // sharp writes png/jpg/webp; anything else (bmp, tiff, avif) becomes png.
    const ENCODABLE = ['png', 'jpg', 'webp'];
    let outFormat =
      format === 'keep'
        ? path.extname(file).toLowerCase().replace('.', '').replace('jpeg', 'jpg')
        : format;
    let destFormat = format;
    if (!ENCODABLE.includes(outFormat)) {
      outFormat = 'png';
      destFormat = 'png';
    }
    const dest = outputPath(file, inputDir, outputDir, destFormat, suffix);

    emit({ type: 'file', file, dest, index: processed, total: files.length });

    try {
      if (!overwrite && fs.existsSync(dest)) {
        skipped++;
        processed++;
        emit({ type: 'progress', file, status: 'skipped', processed, total: files.length });
        continue;
      }

      const buf = await fsp.readFile(file);
      const det = await detect(buf, { ...detectOptions, perLabelMasks: saveMasks });
      const result = await censor(buf, det, {
        ...censorOptions,
        format: outFormat,
        quality,
      });

      await fsp.mkdir(path.dirname(dest), { recursive: true });

      const key = review.maskKey(inputDir, file);
      if (saveMasks) {
        await review.writeMask(
          review.maskPath(outputDir, key, 'union'),
          det.mask,
          det.width,
          det.height
        );
        for (const [label, mask] of Object.entries(det.labelMasks || {})) {
          await review.writeMask(
            review.maskPath(outputDir, key, label),
            mask,
            det.width,
            det.height
          );
        }
      }

      manifest.entries.push({
        key,
        source: file,
        dest,
        width: det.width,
        height: det.height,
        detections: det.detections,
        labels: Object.keys(det.labelMasks || {}),
        censored: result.changed,
      });

      if (result.changed) {
        await fsp.writeFile(dest, result.buffer);
        censored++;
        emit({
          type: 'progress',
          file,
          dest,
          status: 'censored',
          detections: det.detections,
          processed: processed + 1,
          total: files.length,
        });
      } else if (copyUnchanged) {
        await fsp.copyFile(file, dest);
        emit({
          type: 'progress',
          file,
          dest,
          status: 'clean',
          detections: [],
          processed: processed + 1,
          total: files.length,
        });
      } else {
        emit({
          type: 'progress',
          file,
          status: 'clean-skipped',
          processed: processed + 1,
          total: files.length,
        });
      }
      processed++;
    } catch (err) {
      failed++;
      processed++;
      emit({
        type: 'error',
        file,
        message: err.message,
        processed,
        total: files.length,
      });
    }
  }

  if (saveMasks && manifest.entries.length) await review.saveManifest(outputDir, manifest);

  emit({ type: 'done', processed, censored, skipped, failed, outputDir });
  return { processed, censored, skipped, failed, cancelled: false };
}

module.exports = { runBatch, listImages };
