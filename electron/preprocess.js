'use strict';

const sharp = require('sharp');

/**
 * Ultralytics-style letterbox: scale to fit, then pad to a square with gray 114,
 * centered. Each sharp stage is materialised with toBuffer() because sharp
 * reorders resize/extend inside a single pipeline.
 *
 * Returns { data: Float32Array(1*3*size*size), scale, padX, padY }.
 */
async function letterbox(inputBuffer, size) {
  const meta = await sharp(inputBuffer).metadata();
  const w = meta.width;
  const h = meta.height;
  const scale = Math.min(size / w, size / h);
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);

  const resized = await sharp(inputBuffer)
    .resize(nw, nh, { fit: 'fill', kernel: 'lanczos3' })
    .removeAlpha()
    .toColourspace('srgb')
    .toBuffer();

  const padX = Math.floor((size - nw) / 2);
  const padY = Math.floor((size - nh) / 2);

  const padded = await sharp(resized)
    .extend({
      top: padY,
      bottom: size - nh - padY,
      left: padX,
      right: size - nw - padX,
      background: { r: 114, g: 114, b: 114 },
    })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const { data, info } = padded;
  if (info.width !== size || info.height !== size || info.channels !== 3) {
    throw new Error(
      `letterbox produced ${info.width}x${info.height}x${info.channels}, expected ${size}x${size}x3`
    );
  }

  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = data[i * 3] / 255;
    out[plane + i] = data[i * 3 + 1] / 255;
    out[2 * plane + i] = data[i * 3 + 2] / 255;
  }

  return { data: out, scale, padX, padY, srcWidth: w, srcHeight: h };
}

module.exports = { letterbox };
