'use strict';

const path = require('path');
const fsp = require('fs/promises');
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const {
  detect,
  setModelsDir,
  setUserModelsDir,
  availableModels,
  DEFAULT_ENSEMBLE,
} = require('./detector');
const { censor } = require('./censor');
const { runBatch } = require('./batch');
const { CLASSES, KO, defaultLabelConfig } = require('./labels');
const review = require('./review');
const { addSample, datasetStats } = require('./dataset');
const train = require('./train');
const download = require('./download');
const sharp = require('sharp');
const fs = require('fs');

let win = null;
let cancelRequested = false;

function modelsDir() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'models')
    : path.join(__dirname, '..', 'models');
}

function srcModelsDir() {
  const devPath = path.join(__dirname, '..', 'tools', 'src-models');
  if (fs.existsSync(devPath)) return devPath;
  return modelsDir();
}

// Fine-tuned models and training runs live in userData so they survive a
// reinstall and never bloat the app bundle.
function userModelsDir() {
  return path.join(app.getPath('userData'), 'models');
}

function runsDir() {
  return path.join(app.getPath('userData'), 'runs');
}

function createWindow() {
  win = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 900,
    minHeight: 640,
    title: 'AutoCensor',
    backgroundColor: '#14161a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  setModelsDir(modelsDir());
  setUserModelsDir(userModelsDir());
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

ipcMain.handle('meta', () => ({
  classes: CLASSES,
  ko: KO,
  labelConfig: defaultLabelConfig(),
  models: availableModels(),
  defaultModels: DEFAULT_ENSEMBLE,
}));

ipcMain.handle('pick-folder', async (_e, title) => {
  const r = await dialog.showOpenDialog(win, {
    title,
    properties: ['openDirectory', 'createDirectory'],
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle('pick-image', async () => {
  const r = await dialog.showOpenDialog(win, {
    title: '이미지 선택',
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'avif'] }],
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle('open-path', (_e, p) => shell.openPath(p));

/** Single-image preview: returns before/after data URLs plus the detection list. */
ipcMain.handle('preview', async (_e, { file, detectOptions, censorOptions }) => {
  const buf = await fsp.readFile(file);
  const det = await detect(buf, detectOptions);
  const result = await censor(buf, det, { ...censorOptions, format: 'png' });
  return {
    detections: det.detections,
    width: det.width,
    height: det.height,
    original: `data:image/png;base64,${buf.toString('base64')}`,
    censored: result.changed
      ? `data:image/png;base64,${result.buffer.toString('base64')}`
      : null,
  };
});

ipcMain.handle('batch-start', async (_e, opts) => {
  cancelRequested = false;
  const emit = (ev) => {
    if (win && !win.isDestroyed()) win.webContents.send('batch-event', ev);
  };
  try {
    return await runBatch(opts, emit, () => cancelRequested);
  } catch (err) {
    emit({ type: 'fatal', message: err.message, stack: err.stack });
    throw err;
  }
});

ipcMain.handle('batch-cancel', () => {
  cancelRequested = true;
  return true;
});


// ---------------------------------------------------------------- review tab

/** Masks travel to and from the renderer as PNG data URLs. */
async function maskFromDataUrl(dataUrl, width, height) {
  const base64 = dataUrl.slice(dataUrl.indexOf(',') + 1);
  const { data, info } = await sharp(Buffer.from(base64, 'base64'))
    .toColourspace('b-w')
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.width !== width || info.height !== height) {
    throw new Error(`mask is ${info.width}x${info.height}, expected ${width}x${height}`);
  }
  const out = new Uint8Array(width * height);
  for (let i = 0; i < out.length; i++) out[i] = data[i * info.channels] > 127 ? 255 : 0;
  return out;
}

async function maskToDataUrl(file, width, height) {
  if (!fs.existsSync(file)) return null;
  const mask = await review.readMask(file, width, height);
  const png = await sharp(Buffer.from(mask), { raw: { width, height, channels: 1 } })
    .toColourspace('b-w')
    .png()
    .toBuffer();
  return `data:image/png;base64,${png.toString('base64')}`;
}

ipcMain.handle('review-load', async (_e, outputDir) => {
  const manifest = await review.loadManifest(outputDir);
  if (!manifest) return null;
  const entries = manifest.entries.filter((en) => fs.existsSync(en.dest));
  return { ...manifest, entries };
});

ipcMain.handle('review-thumb', async (_e, { file, size = 280 }) => {
  const buf = await sharp(file)
    .resize(size, size, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 78 })
    .toBuffer();
  return `data:image/jpeg;base64,${buf.toString('base64')}`;
});

ipcMain.handle('review-open', async (_e, { outputDir, entry }) => {
  const { key, source, width, height } = entry;
  const original = await fsp.readFile(source);
  const labelMasks = {};
  for (const label of entry.labels || []) {
    const url = await maskToDataUrl(review.maskPath(outputDir, key, label), width, height);
    if (url) labelMasks[label] = url;
  }
  return {
    original: `data:image/png;base64,${(await sharp(original).removeAlpha().png().toBuffer()).toString('base64')}`,
    union: await maskToDataUrl(review.maskPath(outputDir, key, 'union'), width, height),
    labelMasks,
  };
});

/** The censored output, loaded only when the user switches to the result view. */
ipcMain.handle('review-dest', async (_e, file) => {
  const png = await sharp(file).removeAlpha().png().toBuffer();
  return `data:image/png;base64,${png.toString('base64')}`;
});

/** Re-censor one image from a hand-corrected mask, using the current settings. */
ipcMain.handle('review-apply', async (_e, payload) => {
  const { outputDir, entry, unionMask, classMasks, censorOptions } = payload;
  const { source, dest, width, height, key } = entry;

  const mask = await maskFromDataUrl(unionMask, width, height);
  const buf = await fsp.readFile(source);
  const ext = path.extname(dest).toLowerCase().replace('.', '').replace('jpeg', 'jpg');
  const format = ['png', 'jpg', 'webp'].includes(ext) ? ext : 'png';

  const result = await censor(buf, { width, height, detections: [], mask }, {
    ...censorOptions,
    format,
  });
  if (result.changed) await fsp.writeFile(dest, result.buffer);
  else await fsp.copyFile(source, dest);

  await review.writeMask(review.maskPath(outputDir, key, 'union'), mask, width, height);
  const labels = [];
  for (const [label, url] of Object.entries(classMasks || {})) {
    const m = await maskFromDataUrl(url, width, height);
    if (!m.some((v) => v)) continue;
    await review.writeMask(review.maskPath(outputDir, key, label), m, width, height);
    labels.push(label);
  }

  const manifest = await review.loadManifest(outputDir);
  if (manifest) {
    const found = manifest.entries.find((en) => en.key === key);
    if (found) {
      found.labels = labels;
      found.censored = result.changed;
      found.corrected = true;
    }
    await review.saveManifest(outputDir, manifest);
  }
  return { changed: result.changed, coveredPixels: result.coveredPixels, labels };
});

ipcMain.handle('dataset-add', async (_e, payload) => {
  const { datasetDir, entry, classMasks } = payload;
  const { source, width, height } = entry;
  const decoded = {};
  for (const [label, url] of Object.entries(classMasks || {})) {
    const m = await maskFromDataUrl(url, width, height);
    if (m.some((v) => v)) decoded[label] = m;
  }
  const info = await addSample({
    datasetDir,
    sourcePath: source,
    width,
    height,
    classMasks: decoded,
  });
  return { ...info, stats: await datasetStats(datasetDir) };
});

ipcMain.handle('dataset-stats', (_e, datasetDir) => datasetStats(datasetDir));


// -------------------------------------------------------------- fine-tuning

ipcMain.handle('models-list', () => availableModels());

ipcMain.handle('train-check', async (_e, extraPaths) => {
  const python = await train.findPython(extraPaths || []);
  return {
    python,
    device: train.defaultDevice(python),
    basePresets: train.BASE_PRESETS,
    userModelsDir: userModelsDir(),
    runsDir: runsDir(),
  };
});

ipcMain.handle('train-dataset-ready', (_e, dir) => train.datasetReady(dir));

ipcMain.handle('pick-weights', async () => {
  const r = await dialog.showOpenDialog(win, {
    title: '베이스 가중치 선택 (.pt)',
    properties: ['openFile'],
    filters: [{ name: 'PyTorch weights', extensions: ['pt'] }],
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle('train-start', async (_e, opts) => {
  const emit = (ev) => {
    if (win && !win.isDestroyed()) win.webContents.send('train-event', ev);
  };
  try {
    return await train.startTraining(
      { ...opts, runsDir: runsDir(), userModelsDir: userModelsDir() },
      emit
    );
  } catch (err) {
    emit({ type: 'error', message: err.message });
    return { ok: false, message: err.message };
  }
});

ipcMain.handle('train-cancel', () => train.cancel());

// -------------------------------------------------------------- model downloads

ipcMain.handle('models-status', () => {
  return download.getModelsStatus({
    modelsDir: modelsDir(),
    userModelsDir: userModelsDir(),
    srcModelsDir: srcModelsDir(),
  });
});

ipcMain.handle('model-download', async (_e, { key, token }) => {
  const emit = (ev) => {
    if (win && !win.isDestroyed()) win.webContents.send('model-download-progress', ev);
  };
  try {
    return await download.downloadModel(
      key,
      {
        modelsDir: modelsDir(),
        userModelsDir: userModelsDir(),
        srcModelsDir: srcModelsDir(),
      },
      { token },
      emit
    );
  } catch (err) {
    emit({ key, stage: 'error', message: err.message });
    throw err;
  }
});

ipcMain.handle('model-download-cancel', () => download.cancelDownload());

ipcMain.handle('model-delete', async (_e, key) => {
  const uDir = userModelsDir();
  const spec = detector.MODELS[key];
  const onnxFile = spec ? spec.file : `${key}.onnx`;
  const ptFile = onnxFile.replace(/\.onnx$/i, '.pt');

  const targets = [
    path.join(uDir, `${key}.json`),
    path.join(uDir, onnxFile),
    path.join(uDir, ptFile),
  ];
  for (const p of targets) {
    try {
      if (fs.existsSync(p)) await fsp.unlink(p);
    } catch {
      // ignore
    }
  }
  detector.unloadModel(key);
  return true;
});
