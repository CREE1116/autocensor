'use strict';
// A library big enough to break `out.push(...subtree)` (V8 gives up around
// 125k spread arguments) must still enumerate.
const assert = require('assert');
const path = require('path');
const Module = require('module');

const FILES_PER_DIR = 40000;
const DIRS = 5; // 200k images, well past the spread limit

const realLoad = Module._load;
Module._load = function patched(request, parent, isMain) {
  if (request === 'fs/promises') {
    return {
      readdir: async (dir) => {
        const depth = dir.split(path.sep).filter((p) => p.startsWith('sub')).length;
        if (depth === 0) {
          return Array.from({ length: DIRS }, (_, i) => ({
            name: `sub${i}`,
            isDirectory: () => true,
          }));
        }
        return Array.from({ length: FILES_PER_DIR }, (_, i) => ({
          name: `img${i}.png`,
          isDirectory: () => false,
        }));
      },
    };
  }
  return realLoad(request, parent, isMain);
};

const { listImages } = require('../electron/batch');
Module._load = realLoad;

(async () => {
  const files = await listImages('/root', true);
  assert.strictEqual(files.length, DIRS * FILES_PER_DIR, `got ${files.length}`);
  assert.ok(files[0].endsWith('.png'));

  // The old form of the same walk, to show what was actually failing.
  let spreadFailed = false;
  try {
    const acc = [];
    acc.push(...files);
  } catch (e) {
    spreadFailed = e.message.includes('Maximum call stack size exceeded');
  }
  assert.ok(spreadFailed, 'expected the spread form to blow the stack at this size');

  console.log(`PASS large tree: ${files.length} files listed; spread form still throws`);
})().catch((e) => { console.error('FAIL', e.message); process.exit(1); });
