'use strict';

const assert = require('assert');
const path = require('path');
const fsp = require('fs/promises');
const os = require('os');
const detector = require('../electron/detector');

(async () => {
  console.log('--- testing custom model delete feature ---');

  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), 'ac-del-test-'));
  detector.setModelsDir(path.join(__dirname, '..', 'models'));
  detector.setUserModelsDir(tmp);

  // 1. Initially no custom model
  let models = await detector.availableModels();
  assert.ok(!models.some((m) => m.key === 'custom_test_model'));

  // 2. Create custom model files
  const descriptor = {
    key: 'custom_test_model',
    file: 'custom_test_model.onnx',
    name: 'custom_test_model',
    label: 'Custom Test Model',
    size: 640,
    task: 'segment',
    classes: ['nipple', 'vagina'],
    map: { nipple: 'nipple', vagina: 'vagina' },
  };
  await fsp.writeFile(path.join(tmp, 'custom_test_model.json'), JSON.stringify(descriptor));
  await fsp.writeFile(path.join(tmp, 'custom_test_model.onnx'), 'mock onnx data');

  models = await detector.availableModels();
  const custom = models.find((m) => m.key === 'custom_test_model');
  assert.ok(custom, 'custom model should be found');
  assert.strictEqual(custom.custom, true);
  console.log('PASS custom model discovery');

  // 3. Delete the custom model
  const jsonPath = path.join(tmp, 'custom_test_model.json');
  const onnxPath = path.join(tmp, 'custom_test_model.onnx');
  await fsp.unlink(jsonPath);
  await fsp.unlink(onnxPath);

  models = await detector.availableModels();
  assert.ok(!models.some((m) => m.key === 'custom_test_model'), 'custom model should be gone after delete');
  console.log('PASS custom model deletion');

  await fsp.rm(tmp, { recursive: true, force: true });
  console.log('ALL MODEL DELETE TESTS PASSED');
})().catch((err) => {
  console.error('FAIL model-delete-test:', err);
  process.exit(1);
});
