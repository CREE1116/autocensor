'use strict';

/**
 * Binary mask -> simplified polygons, so a brush stroke can be written out as a
 * YOLO segmentation label.
 */

/** 8-connected components, ignoring specks too small to be a real annotation. */
function components(mask, w, h, minArea) {
  const seen = new Uint8Array(w * h);
  const out = [];
  const stack = [];

  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    stack.length = 0;
    stack.push(start);
    seen[start] = 1;
    const pixels = [];

    while (stack.length) {
      const k = stack.pop();
      pixels.push(k);
      const x = k % w;
      const y = (k - x) / w;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const nk = ny * w + nx;
          if (seen[nk] || !mask[nk]) continue;
          seen[nk] = 1;
          stack.push(nk);
        }
      }
    }
    if (pixels.length >= minArea) out.push(pixels);
  }
  return out;
}

// Clockwise Moore neighbourhood.
const NEIGH = [
  [1, 0], [1, 1], [0, 1], [-1, 1],
  [-1, 0], [-1, -1], [0, -1], [1, -1],
];

/** Moore boundary tracing with Jacob's stopping criterion. */
function traceBoundary(inSet, w, h, startX, startY) {
  const contour = [[startX, startY]];
  let cx = startX;
  let cy = startY;
  let dir = 6; // arrived from above-left, so start looking upward

  for (let guard = 0; guard < 8 * w * h; guard++) {
    let found = false;
    for (let i = 0; i < 8; i++) {
      const d = (dir + i) % 8;
      const nx = cx + NEIGH[d][0];
      const ny = cy + NEIGH[d][1];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      if (!inSet(nx, ny)) continue;
      cx = nx;
      cy = ny;
      // Resume the scan from just behind where we came in.
      dir = (d + 6) % 8;
      found = true;
      break;
    }
    if (!found) break; // isolated pixel
    // Back at the start: the ring is closed. A boundary that legitimately
    // revisits the start pixel gets cut here, which costs a little detail but
    // never produces the doubled ring that breaks the polygon's area.
    if (cx === startX && cy === startY) break;
    contour.push([cx, cy]);
    if (contour.length > 4 * (w + h) + 16) break;
  }
  return contour;
}

function perpDistance(p, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len = Math.hypot(dx, dy);
  if (len === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  return Math.abs(dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / len;
}

/** Douglas-Peucker, iterative to avoid deep recursion on long contours. */
function simplify(points, epsilon) {
  if (points.length < 3) return points.slice();
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack = [[0, points.length - 1]];

  while (stack.length) {
    const [i, j] = stack.pop();
    let maxD = 0;
    let idx = -1;
    for (let k = i + 1; k < j; k++) {
      const d = perpDistance(points[k], points[i], points[j]);
      if (d > maxD) {
        maxD = d;
        idx = k;
      }
    }
    if (idx > 0 && maxD > epsilon) {
      keep[idx] = 1;
      stack.push([i, idx], [idx, j]);
    }
  }
  const out = [];
  for (let k = 0; k < points.length; k++) if (keep[k]) out.push(points[k]);
  return out;
}

/**
 * @returns array of polygons, each an array of [x, y] in pixel coordinates.
 */
function maskToPolygons(mask, w, h, options = {}) {
  const { minArea = 40, epsilon = null, maxPoints = 200 } = options;
  const polys = [];

  for (const pixels of components(mask, w, h, minArea)) {
    const member = new Set(pixels);
    const inSet = (x, y) => member.has(y * w + x);
    let start = pixels[0];
    for (const k of pixels) if (k < start) start = k;
    const sx = start % w;
    const sy = (start - sx) / w;

    const contour = traceBoundary(inSet, w, h, sx, sy);
    if (contour.length < 6) continue;

    let eps = epsilon;
    if (eps === null) {
      let perim = 0;
      for (let i = 1; i < contour.length; i++) {
        perim += Math.hypot(contour[i][0] - contour[i - 1][0], contour[i][1] - contour[i - 1][1]);
      }
      eps = Math.max(1, perim * 0.004);
    }

    let poly = simplify(contour, eps);
    // Keep label files small; a coarser epsilon is better than a truncated ring.
    while (poly.length > maxPoints) {
      eps *= 1.5;
      poly = simplify(contour, eps);
    }
    if (poly.length >= 3) polys.push(poly);
  }
  return polys;
}

/** One YOLO segmentation label line: `cls x1 y1 x2 y2 ...`, normalised. */
function polygonToYolo(poly, classIndex, w, h) {
  const parts = [String(classIndex)];
  for (const [x, y] of poly) {
    parts.push((Math.min(w, Math.max(0, x)) / w).toFixed(6));
    parts.push((Math.min(h, Math.max(0, y)) / h).toFixed(6));
  }
  return parts.join(' ');
}

module.exports = { maskToPolygons, polygonToYolo, simplify, components };
