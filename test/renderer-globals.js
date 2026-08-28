'use strict';
// renderer/*.js are classic scripts: a top-level `function foo()` IS
// `window.foo`. Re-exporting one as `window.foo = () => foo()` overwrites that
// binding with the wrapper, so the wrapper calls itself forever. That shipped
// once and broke both batch and preview with "Maximum call stack size exceeded".
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dir = path.join(__dirname, '..', 'renderer');

// 1. Static check across every renderer script.
for (const file of fs.readdirSync(dir).filter((f) => f.endsWith('.js'))) {
  const src = fs.readFileSync(path.join(dir, file), 'utf8');
  const declared = new Set(
    [...src.matchAll(/^function\s+([A-Za-z_$][\w$]*)/gm)].map((m) => m[1])
  );
  for (const m of src.matchAll(/window\.([A-Za-z_$][\w$]*)\s*=\s*\(\)\s*=>\s*([A-Za-z_$][\w$]*)\(/g)) {
    const [, exported, called] = m;
    assert.ok(
      !(exported === called && declared.has(called)),
      `${file}: window.${exported} = () => ${called}() shadows the top-level function and recurses forever`
    );
  }
}

// 2. Load renderer.js for real against a stub DOM and call the exported getter.
function stubElement() {
  const el = {
    value: '1',
    checked: true,
    textContent: '',
    innerHTML: '',
    className: '',
    style: { setProperty() {} },
    classList: { add() {}, remove() {}, contains: () => false },
    dataset: {},
    appendChild() {},
    removeChild() {},
    // renderLabels() destructures three inputs out of each row it builds.
    querySelectorAll: () => [stubElement(), stubElement(), stubElement()],
    querySelector: () => stubElement(),
    addEventListener() {},
    scrollTop: 0,
    scrollHeight: 0,
    clientHeight: 0,
    childNodes: [],
  };
  return el;
}

const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, stubElement());
    return elements.get(id);
  },
  querySelectorAll: () => [],
  createElement: () => stubElement(),
  addEventListener() {},
};

const sandbox = {
  document,
  console,
  addEventListener() {},
  alert() {},
  confirm: () => true,
  Image: function Image() {},
  setTimeout,
  clearTimeout,
};
sandbox.window = sandbox;
const { CLASSES, KO, defaultLabelConfig } = require('../electron/labels');

sandbox.api = {
  meta: async () => ({
    classes: CLASSES,
    ko: KO,
    labelConfig: defaultLabelConfig(),
    models: [{ key: 'anime-nano', label: 'nano', custom: false }],
    defaultModels: ['anime-nano'],
  }),
  modelsList: async () => [],
  onBatchEvent() {},
  onTrainEvent() {},
  trainCheck: async () => ({ python: null, device: 'cpu' }),
  datasetStats: async () => ({ samples: 0, polygons: 0, classes: {} }),
  modelsStatus: async () => [],
  modelDownload: async () => ({}),
  modelDownloadCancel() {},
  modelDelete: async () => true,
  onModelDownloadProgress() {},
};

const ctx = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(dir, 'renderer.js'), 'utf8'), ctx, {
  filename: 'renderer.js',
});

assert.strictEqual(typeof ctx.window.censorOptions, 'function', 'censorOptions must be exported');

let opts;
try {
  opts = ctx.window.censorOptions();
} catch (e) {
  assert.fail(`window.censorOptions() threw: ${e.message}`);
}
assert.strictEqual(typeof opts, 'object');
for (const key of ['mode', 'shape', 'dilateRadius', 'featherRadius', 'edgeGamma', 'strength']) {
  assert.ok(key in opts, `censorOptions() missing ${key}`);
}

// init() is async; let it settle so a throw inside it fails the test too.
let initError = null;
process.on('unhandledRejection', (e) => {
  initError = e;
});
setTimeout(() => {
  assert.ok(!initError, `init() rejected: ${initError && initError.message}`);
  console.log(
    `PASS renderer globals: window.censorOptions() -> ${Object.keys(opts).length} keys, ` +
      'no recursion, init() clean'
  );
}, 50);
