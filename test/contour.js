'use strict';
const assert = require('assert');
const { maskToPolygons, polygonToYolo } = require('../electron/contour');

function polyArea(p) {
  let a = 0;
  for (let i = 0; i < p.length; i++) {
    const [x1, y1] = p[i];
    const [x2, y2] = p[(i + 1) % p.length];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

const W = 400, H = 300;

// Two separate blobs: a rectangle and a disc.
const mask = new Uint8Array(W * H);
for (let y = 40; y < 100; y++) for (let x = 30; x < 130; x++) mask[y * W + x] = 255;
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  if (Math.hypot(x - 280, y - 180) <= 50) mask[y * W + x] = 255;
}
// A speck that must be dropped.
mask[10 * W + 380] = 255;
mask[10 * W + 381] = 255;

const polys = maskToPolygons(mask, W, H, { minArea: 40 });
assert.strictEqual(polys.length, 2, `expected 2 polygons, got ${polys.length}`);

const areas = polys.map(polyArea).sort((a, b) => a - b);
const wantRect = 100 * 60;
const wantDisc = Math.PI * 50 * 50;
assert.ok(Math.abs(areas[0] - wantRect) / wantRect < 0.05, `rect area ${areas[0]} vs ${wantRect}`);
assert.ok(Math.abs(areas[1] - wantDisc) / wantDisc < 0.06, `disc area ${areas[1]} vs ${wantDisc | 0}`);

for (const p of polys) assert.ok(p.length <= 200, `polygon has ${p.length} points`);

const line = polygonToYolo(polys[0], 3, W, H);
const nums = line.split(' ');
assert.strictEqual(nums[0], '3');
assert.strictEqual((nums.length - 1) % 2, 0);
for (const v of nums.slice(1)) {
  const n = Number(v);
  assert.ok(n >= 0 && n <= 1, `coordinate ${v} not normalised`);
}

console.log(`PASS contour: rect ${areas[0] | 0}/${wantRect}, disc ${areas[1] | 0}/${wantDisc | 0}, points ${polys.map((p) => p.length).join(',')}`);
