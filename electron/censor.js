'use strict';

const sharp = require('sharp');

/** Grayscale sharp pipelines silently widen to 3 channels; force b-w on the way out. */
async function grayRaw(pipeline, width, height) {
  const { data, info } = await pipeline
    .toColourspace('b-w')
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.width !== width || info.height !== height || info.channels !== 1) {
    throw new Error(
      `mask pipeline produced ${info.width}x${info.height}x${info.channels}, expected ${width}x${height}x1`
    );
  }
  return data;
}

function maskFromBoxes(detections, width, height, shape) {
  const mask = new Uint8Array(width * height);
  for (const d of detections) {
    const [x0, y0, x1, y1] = d.box;
    if (shape === 'ellipse') {
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      const rx = Math.max(1, (x1 - x0) / 2);
      const ry = Math.max(1, (y1 - y0) / 2);
      for (let y = y0; y < y1; y++) {
        const dy = (y + 0.5 - cy) / ry;
        for (let x = x0; x < x1; x++) {
          const dx = (x + 0.5 - cx) / rx;
          if (dx * dx + dy * dy <= 1) mask[y * width + x] = 255;
        }
      }
    } else {
      for (let y = y0; y < y1; y++) {
        const row = y * width;
        for (let x = x0; x < x1; x++) mask[row + x] = 255;
      }
    }
  }
  return mask;
}

/** Approximate dilation: blur widens the mask, then a low threshold re-binarises it. */
async function dilate(mask, width, height, radius) {
  if (radius < 0.3) return mask;
  const blurred = await grayRaw(
    sharp(Buffer.from(mask), { raw: { width, height, channels: 1 } }).blur(radius),
    width,
    height
  );
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) out[i] = blurred[i] > 12 ? 255 : 0;
  return out;
}

async function feather(mask, width, height, radius) {
  if (radius < 0.3) return mask;
  return grayRaw(
    sharp(Buffer.from(mask), { raw: { width, height, channels: 1 } }).blur(radius),
    width,
    height
  );
}

/**
 * Turn the binary mask into a soft-brush alpha ramp: dilate past the wanted
 * edge first, then blur by the same amount, so the core stays fully opaque and
 * only the outside fades. Blurring alone would eat into small blobs and leave
 * them translucent in the middle.
 *
 * `gamma` biases the ramp - below 1 keeps it opaque longer before falling off.
 */
async function softEdge(mask, width, height, dilateRadius, featherRadius, gamma) {
  const grown = await dilate(mask, width, height, dilateRadius + featherRadius * 1.2);
  const ramped = await feather(grown, width, height, featherRadius);
  if (gamma === 1 || featherRadius < 0.3) return ramped;
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) {
    const a = ramped[i];
    out[i] = a === 0 || a === 255 ? a : Math.round(255 * Math.pow(a / 255, gamma));
  }
  return out;
}

/** FANZA/DLsite rule of thumb: at least 4px, and long-edge/100 for large images. */
function mosaicBlock(width, height) {
  return Math.max(4, Math.ceil(Math.max(width, height) / 100));
}

async function buildEffectLayer(baseBuffer, width, height, mode, strength) {
  if (mode === 'white' || mode === 'black') {
    const v = mode === 'white' ? 255 : 0;
    const buf = Buffer.alloc(width * height * 3, v);
    return buf;
  }
  if (mode === 'blur') {
    const sigma = Math.max(2, Math.round((Math.max(width, height) / 100) * strength));
    const { data } = await sharp(baseBuffer)
      .blur(sigma)
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    return data;
  }
  if (mode === 'mosaic') {
    const block = Math.max(2, Math.round(mosaicBlock(width, height) * strength));
    const smallW = Math.max(1, Math.round(width / block));
    const smallH = Math.max(1, Math.round(height / block));
    // Two resizes cannot share one pipeline - sharp drops the first one.
    const small = await sharp(baseBuffer).resize(smallW, smallH, { fit: 'fill' }).toBuffer();
    const { data } = await sharp(small)
      .resize(width, height, { fit: 'fill', kernel: 'nearest' })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    return data;
  }
  throw new Error(`unknown censor mode: ${mode}`);
}

/**
 * Apply one combined mask to the whole image in a single composite step.
 * Per-detection compositing double-darkens overlaps and leaves feather seams.
 */
async function censor(imageBuffer, detection, options = {}) {
  const {
    mode = 'white',
    shape = 'contour',
    dilateRadius = 4,
    featherRadius = 10,
    edgeGamma = 0.7,
    scaleWithResolution = true,
    strength = 1,
    format = 'png',
    quality = 95,
  } = options;

  const { width, height, detections } = detection;
  let mask = shape === 'contour' ? detection.mask : maskFromBoxes(detections, width, height, shape);

  let covered = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) covered++;
  if (covered === 0) {
    return { buffer: null, changed: false, coveredPixels: 0 };
  }

  // The sliders are expressed against a 1000px reference so the same setting
  // looks the same on a 900px sketch and a 4000px illustration.
  const k = scaleWithResolution ? Math.max(width, height) / 1000 : 1;
  mask = await softEdge(mask, width, height, dilateRadius * k, featherRadius * k, edgeGamma);

  const base = await sharp(imageBuffer)
    .removeAlpha()
    .toColourspace('srgb')
    .png()
    .toBuffer();
  const { data: rgb, info } = await sharp(base).raw().toBuffer({ resolveWithObject: true });
  if (info.width !== width || info.height !== height || info.channels !== 3) {
    throw new Error(`base raw is ${info.width}x${info.height}x${info.channels}`);
  }

  const layer = await buildEffectLayer(base, width, height, mode, strength);

  const out = Buffer.from(rgb);
  for (let i = 0, px = 0; px < mask.length; px++, i += 3) {
    const a = mask[px];
    if (a === 0) continue;
    if (a === 255) {
      out[i] = layer[i];
      out[i + 1] = layer[i + 1];
      out[i + 2] = layer[i + 2];
    } else {
      const f = a / 255;
      out[i] = out[i] + (layer[i] - out[i]) * f;
      out[i + 1] = out[i + 1] + (layer[i + 1] - out[i + 1]) * f;
      out[i + 2] = out[i + 2] + (layer[i + 2] - out[i + 2]) * f;
    }
  }

  let pipeline = sharp(out, { raw: { width, height, channels: 3 } });
  if (format === 'jpg' || format === 'jpeg') pipeline = pipeline.jpeg({ quality });
  else if (format === 'webp') pipeline = pipeline.webp({ quality });
  else pipeline = pipeline.png();

  return { buffer: await pipeline.toBuffer(), changed: true, coveredPixels: covered };
}

module.exports = { censor, mosaicBlock };
