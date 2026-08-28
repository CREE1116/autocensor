'use strict';

/**
 * Areola boundary by radial ray casting.
 *
 * An areola is concentric with the nipple, so its outline is a function of angle
 * around the nipple's centre: r(theta). Casting one ray per angle and finding
 * where that ray leaves the areola turns a 2D segmentation problem into N
 * independent 1D edge searches. That beats flood filling on every count that
 * matters here - a ray cannot leak sideways through a shadow, each ray carries
 * its own skin reference so lighting gradients across the body do not matter,
 * and a handful of rays ruined by hair, a hand or a specular highlight are just
 * outliers among ~180 samples.
 */

function dist(rgb, i, c) {
  const dr = rgb[i] - c[0];
  const dg = rgb[i + 1] - c[1];
  const db = rgb[i + 2] - c[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

/** Bilinear sample, clamped at the image edge. Writes into `out`. */
function sample(rgb, W, H, x, y, out) {
  const x0 = Math.max(0, Math.min(W - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(H - 1, Math.floor(y)));
  const x1 = Math.min(W - 1, x0 + 1);
  const y1 = Math.min(H - 1, y0 + 1);
  const fx = Math.max(0, Math.min(1, x - x0));
  const fy = Math.max(0, Math.min(1, y - y0));
  const ia = (y0 * W + x0) * 3;
  const ib = (y0 * W + x1) * 3;
  const ic = (y1 * W + x0) * 3;
  const id = (y1 * W + x1) * 3;
  for (let c = 0; c < 3; c++) {
    out[c] =
      rgb[ia + c] * (1 - fx) * (1 - fy) +
      rgb[ib + c] * fx * (1 - fy) +
      rgb[ic + c] * (1 - fx) * fy +
      rgb[id + c] * fx * fy;
  }
}

function median(arr) {
  if (arr.length === 0) return 0;
  const s = Float64Array.from(arr).sort();
  return s[s.length >> 1];
}

/**
 * @param mask   Uint8Array(bw*bh) holding the nipple contour. Mutated in place.
 * @param box    [x0,y0,x1,y1] in image coordinates.
 * @param rgb    Raw RGB of the whole image.
 * @returns stats, or null when no usable areola boundary was found.
 */
function fitAreolaRays(mask, box, rgb, imgW, params, center, seedRadius) {
  const {
    maxScale = 5,
    tolerance = 0.45,
    minContrast = 14,
    rays = 180,
    smoothWindow = 9,
    minValidRays = 0.4,
    minGrowRatio = 1,
    outlierHigh = 1.8,
    outlierLow = 0.5,
    inflate = 1.05,
  } = params;

  const [bx0, by0, bx1, by1] = box;
  const bw = bx1 - bx0;
  const bh = by1 - by0;
  const imgH = Math.ceil(rgb.length / 3 / imgW);

  const rInner = seedRadius * 1.05;
  const rOuter = seedRadius * maxScale;
  const step = Math.max(0.5, seedRadius / 40);
  const px = [0, 0, 0];

  const radii = new Float64Array(rays);
  let valid = 0;

  for (let a = 0; a < rays; a++) {
    const theta = (a / rays) * Math.PI * 2;
    const ux = Math.cos(theta);
    const uy = Math.sin(theta);
    const rayX = (r) => center.x + ux * r;
    const rayY = (r) => center.y + uy * r;

    // Skin reference for THIS ray: the far end of it, so shading that varies
    // across the body never shifts the comparison.
    const skinSamples = [[], [], []];
    for (let r = rOuter * 0.85; r <= rOuter; r += step) {
      sample(rgb, imgW, imgH, rayX(r), rayY(r), px);
      for (let c = 0; c < 3; c++) skinSamples[c].push(px[c]);
    }
    const skin = [median(skinSamples[0]), median(skinSamples[1]), median(skinSamples[2])];

    // Areola contrast measured just outside the nipple contour.
    sample(rgb, imgW, imgH, rayX(seedRadius * 1.25), rayY(seedRadius * 1.25), px);
    const dA = Math.sqrt(
      (px[0] - skin[0]) ** 2 + (px[1] - skin[1]) ** 2 + (px[2] - skin[2]) ** 2
    );
    if (dA < minContrast) {
      radii[a] = NaN; // nothing distinguishable along this ray
      continue;
    }

    // Walk outward; the boundary is the last radius still unlike skin, allowing
    // a short gap so a highlight or a line drawn across the areola is bridged.
    const T = Math.max(minContrast * 0.6, tolerance * dA);
    const gap = Math.max(2, seedRadius * 0.25);
    let last = rInner;
    for (let r = rInner; r <= rOuter; r += step) {
      sample(rgb, imgW, imgH, rayX(r), rayY(r), px);
      const d = Math.sqrt(
        (px[0] - skin[0]) ** 2 + (px[1] - skin[1]) ** 2 + (px[2] - skin[2]) ** 2
      );
      if (d >= T) last = r;
      else if (r - last > gap) break;
    }
    radii[a] = last;
    valid++;
  }

  if (valid < minValidRays * rays) return null;

  // Reject rays that ran far past their neighbours (hair, an arm, a background
  // object aligned with the ray) and replace them with the median.
  const present = [];
  for (let a = 0; a < rays; a++) if (!Number.isNaN(radii[a])) present.push(radii[a]);
  const m = median(present);
  if (m <= seedRadius * 1.05) return null;
  for (let a = 0; a < rays; a++) {
    const r = radii[a];
    if (Number.isNaN(r) || r > m * outlierHigh || r < m * outlierLow) radii[a] = m;
  }

  // Circular median filter: keeps the real shape, removes single-ray spikes.
  const smooth = new Float64Array(rays);
  const half = smoothWindow >> 1;
  const win = [];
  for (let a = 0; a < rays; a++) {
    win.length = 0;
    for (let k = -half; k <= half; k++) win.push(radii[(a + k + rays) % rays]);
    smooth[a] = median(win);
  }

  // An areola seen in perspective is an ellipse, so fitting three parameters to
  // ~180 radii is far steadier than trusting each ray: a highlight that cuts a
  // notch or a hair that stretches a spike moves the fit barely at all.
  const conic = fitEllipseIRLS(smooth, rays, inflate);
  const maxAxis = seedRadius * maxScale * 1.05;
  const usable =
    conic &&
    conic.a >= seedRadius * 1.05 &&
    conic.a <= maxAxis &&
    conic.b >= seedRadius * 0.9;

  const out = Uint8Array.from(mask);
  let seedCount = 0;
  for (let k = 0; k < mask.length; k++) if (mask[k]) seedCount++;
  let filled = 0;
  const perRad = rays / (Math.PI * 2);

  for (let ly = 0; ly < bh; ly++) {
    const dy = by0 + ly - center.y;
    for (let lx = 0; lx < bw; lx++) {
      const dx = bx0 + lx - center.x;
      const r = Math.hypot(dx, dy);
      if (r > rOuter) continue;

      let inside;
      if (usable) {
        inside = conic.A * dx * dx + conic.B * dx * dy + conic.C * dy * dy <= 1;
      } else {
        // Ellipse fit failed (very irregular boundary): fall back to the
        // smoothed star polygon the rays traced directly.
        let t = Math.atan2(dy, dx) * perRad;
        if (t < 0) t += rays;
        const i0 = Math.floor(t) % rays;
        const i1 = (i0 + 1) % rays;
        const f = t - Math.floor(t);
        inside = r <= smooth[i0] * (1 - f) + smooth[i1] * f;
      }
      if (!inside) continue;
      const k = ly * bw + lx;
      if (!out[k]) filled++;
      out[k] = 255;
    }
  }

  if (filled < minGrowRatio * seedCount) return null;

  mask.set(out);
  return {
    radiusMedian: m,
    filled,
    seedCount,
    validRays: valid,
    rays,
    shape: usable ? 'ellipse' : 'star',
    axes: usable ? [conic.a, conic.b] : null,
  };
}

/** Solve a symmetric 3x3 system by Gaussian elimination with partial pivoting. */
function solve3(S, rhs) {
  const m = [
    [S[0], S[1], S[2], rhs[0]],
    [S[3], S[4], S[5], rhs[1]],
    [S[6], S[7], S[8], rhs[2]],
  ];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    [m[col], m[piv]] = [m[piv], m[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let c = col; c < 4; c++) m[r][c] -= f * m[col][c];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}

/**
 * Fit `A x^2 + B xy + C y^2 = 1` to the ray endpoints, re-weighting to drop
 * rays whose radius disagrees with the fit (occlusion, background objects).
 */
function fitEllipseIRLS(radii, rays, inflate) {
  const cos = new Float64Array(rays);
  const sin = new Float64Array(rays);
  for (let a = 0; a < rays; a++) {
    const t = (a / rays) * Math.PI * 2;
    cos[a] = Math.cos(t);
    sin[a] = Math.sin(t);
  }
  let weight = new Float64Array(rays).fill(1);
  let sol = null;

  for (let iter = 0; iter < 3; iter++) {
    const S = new Float64Array(9);
    const rhs = new Float64Array(3);
    for (let a = 0; a < rays; a++) {
      if (weight[a] === 0) continue;
      const x = radii[a] * cos[a];
      const y = radii[a] * sin[a];
      const f = [x * x, x * y, y * y];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) S[i * 3 + j] += weight[a] * f[i] * f[j];
        rhs[i] += weight[a] * f[i];
      }
    }
    sol = solve3(S, rhs);
    if (!sol) return null;
    const [A, B, C] = sol;
    if (A <= 0 || C <= 0 || 4 * A * C - B * B <= 0) return null;

    // Residual per ray, in pixels of radius.
    const resid = new Float64Array(rays);
    for (let a = 0; a < rays; a++) {
      const q = A * cos[a] * cos[a] + B * cos[a] * sin[a] + C * sin[a] * sin[a];
      resid[a] = Math.abs(radii[a] - 1 / Math.sqrt(q));
    }
    const med = median(Array.from(resid));
    const mad = median(Array.from(resid, (r) => Math.abs(r - med))) || 1e-6;
    const cut = med + 2.5 * mad;
    weight = Float64Array.from(resid, (r) => (r <= cut ? 1 : 0));
  }

  // Inflate slightly: for censoring, stopping a pixel short of the edge is the
  // costlier error.
  const k = 1 / (inflate * inflate);
  const [A, B, C] = sol;
  const tr = A + C;
  const det = A * C - (B * B) / 4;
  const disc = Math.sqrt(Math.max(0, (tr / 2) * (tr / 2) - det));
  const l1 = tr / 2 - disc;
  const l2 = tr / 2 + disc;
  if (l1 <= 0 || l2 <= 0) return null;
  return {
    A: A * k,
    B: B * k,
    C: C * k,
    a: inflate / Math.sqrt(l1),
    b: inflate / Math.sqrt(l2),
  };
}

module.exports = { fitAreolaRays, fitEllipseIRLS };
