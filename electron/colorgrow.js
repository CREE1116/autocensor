'use strict';

/**
 * The detection model has a `nipple` class but no `areola` class, and an areola
 * is a colour region rather than a shape - its size relative to the nipple
 * varies per drawing, so any fixed scale factor either overshoots onto bare skin
 * or leaves a ring uncovered.
 *
 * Instead we treat the nipple contour as a seed and flood outward across pixels
 * that are not skin-coloured, bounded by a radius. Flat cel-shaded art has very
 * few distinct colours per region, which is exactly the case region growing
 * handles well.
 */

function dist2(r1, g1, b1, r2, g2, b2) {
  const dr = r1 - r2;
  const dg = g1 - g2;
  const db = b1 - b2;
  return dr * dr + dg * dg + db * db;
}

function median(values) {
  if (values.length === 0) return 0;
  const sorted = Float64Array.from(values).sort();
  return sorted[sorted.length >> 1];
}

/**
 * @param mask   Uint8Array(bw*bh), 255 on the seed contour. Mutated in place.
 * @param box    [x0,y0,x1,y1] in original image coordinates.
 * @param rgb    Raw RGB of the whole image (3 channels, imgW wide).
 * @param params { maxScale, tolerance, minContrast, leakLimit }
 * @returns stats for logging, or null when the grow was rejected.
 */
function growAreola(mask, box, rgb, imgW, params, center, seedRadius) {
  const {
    maxScale = 5,
    tolerance = 0.35,
    minContrast = 16,
    leakLimit = 0.75,
    minRoundness = 0.45,
    minGrowRatio = 1,
  } = params;

  const [bx0, by0, bx1, by1] = box;
  const bw = bx1 - bx0;
  const bh = by1 - by0;
  const rOuter = seedRadius * maxScale;
  const rOuter2 = rOuter * rOuter;

  const at = (x, y) => ((y * imgW + x) * 3) | 0;

  // Skin reference: a thin ring at the very edge of the search disc, excluding
  // the seed. It has to sit outside the areola, so the disc is deliberately much
  // wider than any plausible areola and the ring is kept narrow.
  const skinR = [];
  const skinG = [];
  const skinB = [];
  const rInner2 = (rOuter * 0.88) * (rOuter * 0.88);
  for (let y = by0; y < by1; y++) {
    const dy = y - center.y;
    for (let x = bx0; x < bx1; x++) {
      const dx = x - center.x;
      const d2 = dx * dx + dy * dy;
      if (d2 < rInner2 || d2 > rOuter2) continue;
      if (mask[(y - by0) * bw + (x - bx0)]) continue;
      const i = at(x, y);
      skinR.push(rgb[i]);
      skinG.push(rgb[i + 1]);
      skinB.push(rgb[i + 2]);
    }
  }
  if (skinR.length < 24) return null;
  const sR = median(skinR);
  const sG = median(skinG);
  const sB = median(skinB);

  // Seed reference: the mean colour of the nipple contour.
  let nR = 0;
  let nG = 0;
  let nB = 0;
  let seedCount = 0;
  for (let y = by0; y < by1; y++) {
    for (let x = bx0; x < bx1; x++) {
      if (!mask[(y - by0) * bw + (x - bx0)]) continue;
      const i = at(x, y);
      nR += rgb[i];
      nG += rgb[i + 1];
      nB += rgb[i + 2];
      seedCount++;
    }
  }
  if (seedCount === 0) return null;
  nR /= seedCount;
  nG /= seedCount;
  nB /= seedCount;

  // If nipple and skin are barely different, there is no colour signal to follow
  // and growing would run away across the whole body.
  const contrast = Math.sqrt(dist2(nR, nG, nB, sR, sG, sB));
  if (contrast < minContrast) return null;
  const thresh2 = Math.pow(tolerance * contrast, 2);

  // Flood outward from the seed over pixels that are not skin-coloured.
  const out = Uint8Array.from(mask);
  const queue = new Int32Array(bw * bh);
  let head = 0;
  let tail = 0;
  for (let k = 0; k < out.length; k++) if (out[k]) queue[tail++] = k;

  let grown = 0;
  let rMax2 = 0;
  while (head < tail) {
    const k = queue[head++];
    const lx = k % bw;
    const ly = (k - lx) / bw;
    for (let n = 0; n < 4; n++) {
      const nx = lx + (n === 0 ? -1 : n === 1 ? 1 : 0);
      const ny = ly + (n === 2 ? -1 : n === 3 ? 1 : 0);
      if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue;
      const nk = ny * bw + nx;
      if (out[nk]) continue;
      const gx = bx0 + nx;
      const gy = by0 + ny;
      const dx = gx - center.x;
      const dy = gy - center.y;
      if (dx * dx + dy * dy > rOuter2) continue;
      const i = at(gx, gy);
      if (dist2(rgb[i], rgb[i + 1], rgb[i + 2], sR, sG, sB) <= thresh2) continue;
      out[nk] = 255;
      grown++;
      const r2 = dx * dx + dy * dy;
      if (r2 > rMax2) rMax2 = r2;
      queue[tail++] = nk;
    }
  }

  // A grow that swallows most of the search disc means the colour test failed to
  // separate anything - keep the plain contour rather than a huge blob.
  const total = grown + seedCount;
  if (total > leakLimit * Math.PI * rOuter2) return null;

  // Barely growing means the areola is not separable from skin here. Report
  // failure so the caller can fall back to a geometric expansion, rather than
  // silently covering the nipple alone.
  if (grown < minGrowRatio * seedCount) return null;

  // An areola is a compact blob. A region that reaches far out while covering
  // little of the disc it spans has run down a shadow or a line instead.
  const roundness = rMax2 > 0 ? total / (Math.PI * rMax2) : 1;
  if (roundness < minRoundness) return null;

  mask.set(out);
  return {
    contrast,
    grown,
    seedCount,
    roundness,
    radius: Math.sqrt(rMax2),
    skin: [sR, sG, sB],
    nipple: [nR, nG, nB],
  };
}

module.exports = { growAreola };
