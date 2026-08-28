'use strict';

const assert = require('assert');
const http = require('http');
const fsp = require('fs/promises');
const fs = require('fs');
const path = require('path');
const os = require('os');
const {
  downloadFile,
  getModelsStatus,
  DOWNLOAD_REGISTRY,
  formatBytes,
} = require('../electron/download');

(async () => {
  console.log('--- testing download module ---');

  // Test 1: formatBytes helper
  assert.strictEqual(formatBytes(0), '0 B');
  assert.strictEqual(formatBytes(1024), '1 KB');
  assert.strictEqual(formatBytes(1024 * 1024 * 5), '5 MB');
  console.log('PASS formatBytes');

  // Test 2: getModelsStatus with empty / mock directories
  const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), 'ac-dl-test-'));
  const modelsDir = path.join(tmp, 'models');
  const userModelsDir = path.join(tmp, 'usermodels');
  const srcModelsDir = path.join(tmp, 'srcmodels');
  await fsp.mkdir(modelsDir, { recursive: true });
  await fsp.mkdir(userModelsDir, { recursive: true });
  await fsp.mkdir(srcModelsDir, { recursive: true });

  // Initially none installed
  let statuses = getModelsStatus({ modelsDir, userModelsDir, srcModelsDir });
  assert.ok(statuses.length >= 6);
  assert.ok(statuses.every((s) => s.installed === false));

  // Create a dummy model file in modelsDir
  await fsp.writeFile(path.join(modelsDir, 'nsfw-anime-nano-x640.onnx'), 'mock model data');
  statuses = getModelsStatus({ modelsDir, userModelsDir, srcModelsDir });
  const nano = statuses.find((s) => s.key === 'anime-nano');
  assert.ok(nano);
  assert.strictEqual(nano.installed, true);
  assert.strictEqual(nano.size, 15);
  console.log('PASS getModelsStatus');

  // Test 3: Local HTTP server with redirect & progress tracking
  const payload = 'A'.repeat(50000);
  const server = http.createServer((req, res) => {
    if (req.url === '/redirect') {
      res.writeHead(302, { Location: '/target' });
      res.end();
      return;
    }
    if (req.url === '/target') {
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Length': Buffer.byteLength(payload),
      });
      // Send in chunks to test progress
      res.write(payload.slice(0, 20000));
      setTimeout(() => {
        res.write(payload.slice(20000, 40000));
        setTimeout(() => {
          res.end(payload.slice(40000));
        }, 30);
      }, 30);
      return;
    }
    res.writeHead(404);
    res.end();
  });

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  const targetUrl = `http://127.0.0.1:${port}/redirect`;

  const destFile = path.join(tmp, 'downloaded.bin');
  const progressEvents = [];

  const dlResult = await downloadFile(targetUrl, destFile, {}, (p) => {
    progressEvents.push(p);
  });

  assert.strictEqual(dlResult.destPath, destFile);
  assert.strictEqual(dlResult.size, 50000);
  assert.ok(fs.existsSync(destFile));
  const downloadedContent = await fsp.readFile(destFile, 'utf8');
  assert.strictEqual(downloadedContent, payload);
  assert.ok(progressEvents.length > 0, 'expected progress events');
  assert.ok(progressEvents[progressEvents.length - 1].percent >= 99);
  console.log('PASS downloadFile with redirect & progress');

  server.close();
  await fsp.rm(tmp, { recursive: true, force: true });
  console.log('ALL DOWNLOAD TESTS PASSED');
})().catch((err) => {
  console.error('FAIL download-test:', err);
  process.exit(1);
});
