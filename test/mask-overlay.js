'use strict';
// The overlay used `destination-in` against an opaque black/white mask, which
// clips by alpha - and an opaque mask has alpha 255 everywhere, so nothing was
// clipped and the mask never showed. Coverage has to move into alpha first.
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const src = fs.readFileSync(path.join(__dirname, '..', 'renderer', 'review.js'), 'utf8');
const sandbox = { document: { getElementById: () => ({}) }, console };
sandbox.window = sandbox;
vm.runInContext(src, vm.createContext(sandbox), { filename: 'review.js' });

const { maskPixelsToAlpha } = sandbox.window.reviewInternals;

// black, white, mid-dark, mid-light
const px = new Uint8ClampedArray([
  0, 0, 0, 255,
  255, 255, 255, 255,
  100, 100, 100, 255,
  200, 200, 200, 255,
]);
maskPixelsToAlpha(px);

assert.strictEqual(px[3], 0, 'black must become transparent');
assert.strictEqual(px[7], 255, 'white must become opaque');
assert.strictEqual(px[11], 0, 'below the midpoint is not covered');
assert.strictEqual(px[15], 255, 'above the midpoint is covered');
for (let i = 0; i < px.length; i += 4) {
  assert.strictEqual(px[i], 255, 'colour must be uniform so the tint is flat');
}

// The old form: an opaque mask offers no alpha to clip with.
const opaque = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255]);
assert.strictEqual(opaque[3], opaque[7], 'opaque mask has identical alpha everywhere');

console.log('PASS mask overlay: coverage moved into alpha (0/255), colour flat');
