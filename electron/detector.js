'use strict';

const path = require('path');
const fs = require('fs');
const sharp = require('sharp');
const ort = require('onnxruntime-node');
const { letterbox } = require('./preprocess');
const { growAreola } = require('./colorgrow');
const { fitAreolaRays } = require('./rayfit');
const { MODELS, DEFAULT_ENSEMBLE } = require('./models');

const NUM_COEFFS = 32;

let modelsDir = null;
let userModelsDir = null;
const sessions = new Map();

function setModelsDir(dir) {
  modelsDir = dir;
}

/**
 * Models produced by fine-tuning live outside the app bundle so they survive a
 * reinstall. Each one is a `.onnx` next to a `.json` descriptor in the same
 * shape as an entry in models.js.
 */
function setUserModelsDir(dir) {
  userModelsDir = dir;
}

function loadUserModels() {
  if (!userModelsDir || !fs.existsSync(userModelsDir)) return {};
  const out = {};
  for (const f of fs.readdirSync(userModelsDir)) {
    if (!f.endsWith('.json')) continue;
    try {
      const spec = JSON.parse(fs.readFileSync(path.join(userModelsDir, f), 'utf8'));
      if (spec.key && spec.file) out[spec.key] = { ...spec, dir: userModelsDir, custom: true };
    } catch {
      // A malformed descriptor should not take the whole model list down.
    }
  }
  return out;
}

function specFor(name) {
  return loadUserModels()[name] || MODELS[name] || null;
}

function findModelFile(file, customDir) {
  const candidates = [
    customDir ? path.join(customDir, file) : null,
    userModelsDir ? path.join(userModelsDir, file) : null,
    modelsDir ? path.join(modelsDir, file) : null,
    path.join(process.resourcesPath || '', 'models', file),
    path.join(__dirname, '..', 'models', file),
    path.join(__dirname, '..', 'tools', 'src-models', file),
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

function availableModels() {
  const all = { ...MODELS, ...loadUserModels() };
  return Object.entries(all).map(([key, spec]) => {
    const modelPath = findModelFile(spec.file, spec.dir);
    const installed = !!modelPath;
    return {
      key,
      label: spec.label,
      size: spec.size,
      task: spec.task,
      custom: !!spec.custom,
      installed,
    };
  });
}

async function getSession(name) {
  if (sessions.has(name)) return sessions.get(name);
  const spec = specFor(name);
  if (!spec) throw new Error(`unknown model: ${name}`);
  const file = findModelFile(spec.file, spec.dir);
  const isMocked = !ort.InferenceSession.create.toString().includes('[native code]');
  if (!file && !isMocked) throw new Error(`model file missing: ${spec.file}`);
  const session = await ort.InferenceSession.create(file || spec.file, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: false,
    enableMemPattern: false,
  });
  const entry = { name, session, ...spec };
  sessions.set(name, entry);
  return entry;
}

function unloadModel(name) {
  if (sessions.has(name)) {
    sessions.delete(name);
  }
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  if (inter <= 0) return 0;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

function nms(dets, threshold) {
  const kept = [];
  const byClass = new Map();
  for (const d of dets) {
    if (!byClass.has(d.label)) byClass.set(d.label, []);
    byClass.get(d.label).push(d);
  }
  for (const group of byClass.values()) {
    group.sort((a, b) => b.score - a.score);
    const alive = [];
    for (const d of group) {
      if (alive.some((k) => iou(k.box, d.box) > threshold)) continue;
      alive.push(d);
    }
    for (const d of alive) kept.push(d);
  }
  return kept;
}

/** Run one forward pass over `regionBuffer` and paint what it finds into ctx.mask. */
async function runPass(entry, regionBuffer, offsetX, offsetY, ctx) {
  const { session, size, classes, map, task } = entry;
  const numClasses = classes.length;
  const { data, scale, padX, padY } = await letterbox(regionBuffer, size);

  const input = new ort.Tensor('float32', data, [1, 3, size, size]);
  const out = await session.run({ [session.inputNames[0]]: input });
  const pred = out[session.outputNames[0]];

  const isSeg = task === 'segment';
  let proto = null;
  let protoW = 0;
  let protoH = 0;
  if (isSeg && out[session.outputNames[1]]) {
    const protoT = out[session.outputNames[1]];
    [, , protoH, protoW] = protoT.dims;
    proto = protoT.data;
  }
  const protoPlane = protoH * protoW;
  const p = pred.data;

  const raw = [];
  const dims = pred.dims;
  const dim1 = dims[1];
  const dim2 = dims[2];

  if (dim2 === 38 || (dim2 === 6 && !isSeg)) {
    // End-to-end / Decoded format: [1, numBoxes, 38 or 6]
    // Row format: [x0, y0, x1, y1, score, class_id, 32 mask coefficients...]
    const numBoxes = dim1;
    const stride = dim2;
    for (let i = 0; i < numBoxes; i++) {
      const offset = i * stride;
      const score = p[offset + 4];
      const classId = Math.round(p[offset + 5]);
      if (classId < 0 || classId >= numClasses) continue;

      const label = map[classes[classId]];
      if (!label) continue;
      const cfg = ctx.labelConfig[label];
      if (!cfg || !cfg.enabled) continue;

      const mStrength =
        (ctx.modelConfigs && ctx.modelConfigs[entry.name] && ctx.modelConfigs[entry.name].strength) ||
        1.0;
      const effectiveThreshold = Math.max(
        0.05,
        Math.min(0.95, cfg.threshold / Math.max(0.2, mStrength))
      );
      if (score < effectiveThreshold) continue;

      const x0 = p[offset + 0];
      const y0 = p[offset + 1];
      const x1 = p[offset + 2];
      const y1 = p[offset + 3];

      raw.push({
        label,
        score,
        box: [x0, y0, x1, y1],
        index: i,
        isEndToEnd: true,
        coeffOffset: offset + 6,
      });
    }
  } else if (dim2 > dim1 && (dim1 === 4 + numClasses + (isSeg ? NUM_COEFFS : 0) || dim1 < 100)) {
    // Standard Column-major format: [1, channels, numBoxes] (e.g. [1, 48, 8400])
    const numBoxes = dim2;

    for (let i = 0; i < numBoxes; i++) {
      let best = -1;
      let bestScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const s = p[(4 + c) * numBoxes + i];
        if (s > bestScore) {
          bestScore = s;
          best = c;
        }
      }
      if (best < 0) continue;
      const label = map[classes[best]];
      if (!label) continue;
      const cfg = ctx.labelConfig[label];
      if (!cfg || !cfg.enabled) continue;

      const mStrength =
        (ctx.modelConfigs && ctx.modelConfigs[entry.name] && ctx.modelConfigs[entry.name].strength) ||
        1.0;
      const effectiveThreshold = Math.max(
        0.05,
        Math.min(0.95, cfg.threshold / Math.max(0.2, mStrength))
      );
      if (bestScore < effectiveThreshold) continue;

      const cx = p[i];
      const cy = p[numBoxes + i];
      const w = p[2 * numBoxes + i];
      const h = p[3 * numBoxes + i];
      raw.push({
        label,
        score: bestScore,
        box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        index: i,
        isEndToEnd: false,
        numBoxes,
      });
    }
  } else {
    // Row-major format: [1, numBoxes, channels] (e.g. [1, 8400, 48])
    const numBoxes = dim1;
    const channels = dim2;

    for (let i = 0; i < numBoxes; i++) {
      const offset = i * channels;
      let best = -1;
      let bestScore = 0;
      for (let c = 0; c < numClasses; c++) {
        const s = p[offset + 4 + c];
        if (s > bestScore) {
          bestScore = s;
          best = c;
        }
      }
      if (best < 0) continue;
      const label = map[classes[best]];
      if (!label) continue;
      const cfg = ctx.labelConfig[label];
      if (!cfg || !cfg.enabled) continue;

      const mStrength =
        (ctx.modelConfigs && ctx.modelConfigs[entry.name] && ctx.modelConfigs[entry.name].strength) ||
        1.0;
      const effectiveThreshold = Math.max(
        0.05,
        Math.min(0.95, cfg.threshold / Math.max(0.2, mStrength))
      );
      if (bestScore < effectiveThreshold) continue;

      const cx = p[offset + 0];
      const cy = p[offset + 1];
      const w = p[offset + 2];
      const h = p[offset + 3];
      raw.push({
        label,
        score: bestScore,
        box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        index: i,
        isEndToEnd: false,
        isRowMajor: true,
        coeffOffset: offset + 4 + numClasses,
      });
    }
  }

  const kept = nms(raw, ctx.nmsIou);
  const results = [];

  for (const det of kept) {
    const toOrigX = (v) => (v - padX) / scale + offsetX;
    const toOrigY = (v) => (v - padY) / scale + offsetY;
    const x0 = toOrigX(det.box[0]);
    const y0 = toOrigY(det.box[1]);
    const x1 = toOrigX(det.box[2]);
    const y1 = toOrigY(det.box[3]);

    const cfg = ctx.labelConfig[det.label];
    const cxOrig = (x0 + x1) / 2;
    const cyOrig = (y0 + y1) / 2;
    const seedRadius = Math.max(1, (x1 - x0 + y1 - y0) / 4);

    const tightBox = [
      Math.max(0, Math.floor(x0)),
      Math.max(0, Math.floor(y0)),
      Math.min(ctx.width, Math.ceil(x1)),
      Math.min(ctx.height, Math.ceil(y1)),
    ];
    if (tightBox[2] <= tightBox[0] || tightBox[3] <= tightBox[1]) continue;

    const expand = cfg.expand || 1;
    const areola =
      ctx.rgb && cfg.areola && cfg.areola.method && cfg.areola.method !== 'off'
        ? cfg.areola
        : null;
    const boxScale = areola ? Math.max(expand, areola.maxScale || 1) : expand;
    const hw = ((x1 - x0) / 2) * boxScale;
    const hh = ((y1 - y0) / 2) * boxScale;

    const workBox = [
      Math.max(0, Math.floor(cxOrig - hw)),
      Math.max(0, Math.floor(cyOrig - hh)),
      Math.min(ctx.width, Math.ceil(cxOrig + hw)),
      Math.min(ctx.height, Math.ceil(cyOrig + hh)),
    ];
    if (workBox[2] <= workBox[0] || workBox[3] <= workBox[1]) continue;

    results.push({ label: det.label, score: det.score, box: tightBox, model: entry.name });

    const bw = workBox[2] - workBox[0];
    const bh = workBox[3] - workBox[1];
    const local = new Uint8Array(bw * bh);

    // Detection-only models have no mask prototypes; an ellipse inscribed in the
    // box is the closest stand-in, and for a nipple it is also a usable seed for
    // the areola fit.
    let small = null;
    if (isSeg && proto) {
      small = new Float32Array(protoPlane);
      for (let c = 0; c < NUM_COEFFS; c++) {
        let coeff = 0;
        if (det.isEndToEnd || det.isRowMajor) {
          coeff = p[det.coeffOffset + c];
        } else {
          coeff = p[(4 + numClasses + c) * det.numBoxes + det.index];
        }
        if (coeff === 0 || isNaN(coeff)) continue;
        const base = c * protoPlane;
        for (let k = 0; k < protoPlane; k++) small[k] += coeff * proto[base + k];
      }
    }

    const paint = (factor) => {
      if (isSeg) {
        paintMask(small, protoW, protoH, {
          box: workBox,
          local,
          scale,
          padX,
          padY,
          size,
          offsetX,
          offsetY,
          expand: factor,
          centerX: cxOrig,
          centerY: cyOrig,
        });
      } else {
        paintEllipse(local, workBox, cxOrig, cyOrig, ((x1 - x0) / 2) * factor, ((y1 - y0) / 2) * factor);
      }
    };

    paint(expand);

    if (areola) {
      const fit = areola.method === 'flood' ? growAreola : fitAreolaRays;
      const stats = fit(
        local,
        workBox,
        ctx.rgb,
        ctx.width,
        areola,
        { x: cxOrig, y: cyOrig },
        seedRadius
      );
      // Rejected fit (flat lighting, tiny region, runaway): fall back to scaling
      // the contour geometrically.
      if (!stats) paint(areola.fallbackExpand || expand);
    }

    // Per-label masks are what makes a correction round-trip into a training
    // label later, so they are kept separately from the union used for censoring.
    let labelMask = null;
    if (ctx.labelMasks) {
      labelMask = ctx.labelMasks[det.label];
      if (!labelMask) {
        labelMask = new Uint8Array(ctx.width * ctx.height);
        ctx.labelMasks[det.label] = labelMask;
      }
    }

    for (let ly = 0; ly < bh; ly++) {
      const src = ly * bw;
      const dst = (workBox[1] + ly) * ctx.width + workBox[0];
      for (let lx = 0; lx < bw; lx++) {
        if (!local[src + lx]) continue;
        ctx.mask[dst + lx] = 255;
        if (labelMask) labelMask[dst + lx] = 255;
      }
    }
  }

  return results;
}

/**
 * Sample the low-resolution prototype mask over the detection's box into a
 * box-local buffer. Sampling per output pixel (rather than upsampling the mask)
 * keeps everything in one coordinate transform.
 *
 * Pulling the sample point back toward the centre by 1/expand scales the contour
 * up about that centre, so the blob keeps its shape.
 */
function paintMask(small, protoW, protoH, opt) {
  const { box, local, scale, padX, padY, size, offsetX, offsetY } = opt;
  const expand = opt.expand || 1;
  const cx = opt.centerX;
  const cy = opt.centerY;
  const bw = box[2] - box[0];
  const gx = protoW / size;
  const gy = protoH / size;
  const shrink = (v, c) => (expand === 1 ? v : c + (v - c) / expand);

  for (let y = box[1]; y < box[3]; y++) {
    const yi = (shrink(y, cy) - offsetY) * scale + padY;
    const yp = yi * gy;
    const y0 = Math.floor(yp);
    const fy = yp - y0;
    const ya = Math.min(protoH - 1, Math.max(0, y0));
    const yb = Math.min(protoH - 1, Math.max(0, y0 + 1));
    const rowA = ya * protoW;
    const rowB = yb * protoW;
    const outRow = (y - box[1]) * bw - box[0];

    for (let x = box[0]; x < box[2]; x++) {
      const xi = (shrink(x, cx) - offsetX) * scale + padX;
      const xp = xi * gx;
      const x0 = Math.floor(xp);
      const fx = xp - x0;
      const xa = Math.min(protoW - 1, Math.max(0, x0));
      const xb = Math.min(protoW - 1, Math.max(0, x0 + 1));

      const v =
        small[rowA + xa] * (1 - fx) * (1 - fy) +
        small[rowA + xb] * fx * (1 - fy) +
        small[rowB + xa] * (1 - fx) * fy +
        small[rowB + xb] * fx * fy;

      if (sigmoid(v) > 0.5) local[outRow + x] = 255;
    }
  }
}

function paintEllipse(local, box, cx, cy, rx, ry) {
  const bw = box[2] - box[0];
  const ax = Math.max(1, rx);
  const ay = Math.max(1, ry);
  for (let y = box[1]; y < box[3]; y++) {
    const dy = (y + 0.5 - cy) / ay;
    const row = (y - box[1]) * bw - box[0];
    for (let x = box[0]; x < box[2]; x++) {
      const dx = (x + 0.5 - cx) / ax;
      if (dx * dx + dy * dy <= 1) local[row + x] = 255;
    }
  }
}

function planTiles(width, height, size, overlap) {
  const step = Math.round(size * (1 - overlap));
  const tiles = [];
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      tiles.push({ x, y, w: Math.min(size, width - x), h: Math.min(size, height - y) });
      if (x + size >= width) break;
    }
    if (y + size >= height) break;
  }
  return tiles;
}

/**
 * Detect target regions in one image using one or more models.
 * Returns { width, height, detections, mask } where mask is a Uint8Array of
 * width*height, 255 where something should be censored.
 */
async function detect(imageBuffer, options = {}) {
  const {
    models = DEFAULT_ENSEMBLE,
    model, // legacy single-model form
    labelConfig,
    nmsIou = 0.5,
    tiling = 'auto',
    tileOverlap = 0.2,
    perLabelMasks = false,
    modelConfigs = {},
    onProgress,
  } = options;

  const names = model ? [model] : models;
  const meta = await sharp(imageBuffer).metadata();
  const width = meta.width;
  const height = meta.height;

  const base = await sharp(imageBuffer).removeAlpha().toColourspace('srgb').png().toBuffer();
  const rawBase = await sharp(base).raw().toBuffer({ resolveWithObject: true });
  if (rawBase.info.channels !== 3) {
    throw new Error(`base raw has ${rawBase.info.channels} channels, expected 3`);
  }

  const ctx = {
    width,
    height,
    mask: new Uint8Array(width * height),
    rgb: rawBase.data,
    labelMasks: perLabelMasks ? {} : null,
    labelConfig,
    modelConfigs: modelConfigs || {},
    nmsIou,
  };

  const detections = [];
  const tileCache = new Map();

  for (let mi = 0; mi < names.length; mi++) {
    const entry = await getSession(names[mi]);
    if (onProgress) {
      onProgress({ stage: 'model', name: entry.name, done: mi, total: names.length });
    }

    // Whole-image pass: catches objects too large to fit inside one tile.
    for (const d of await runPass(entry, base, 0, 0, ctx)) detections.push(d);

    const isLargeModel = entry.size >= 1280;
    const thresholdFactor = isLargeModel ? 1.8 : 1.4;
    const useTiles =
      tiling === 'always' || (tiling === 'auto' && Math.max(width, height) > entry.size * thresholdFactor);
    if (!useTiles) continue;

    const tiles = planTiles(width, height, entry.size, tileOverlap);
    for (let i = 0; i < tiles.length; i++) {
      const t = tiles[i];
      const key = `${entry.size}:${t.x},${t.y},${t.w},${t.h}`;
      let crop = tileCache.get(key);
      if (!crop) {
        crop = await sharp(base)
          .extract({ left: t.x, top: t.y, width: t.w, height: t.h })
          .png()
          .toBuffer();
        tileCache.set(key, crop);
      }
      for (const d of await runPass(entry, crop, t.x, t.y, ctx)) detections.push(d);
      if (onProgress) {
        onProgress({ stage: 'tile', name: entry.name, done: i + 1, total: tiles.length });
      }
    }
  }

  // Detections from different passes and models overlap freely; the mask is a
  // union, so merging here only exists to give an honest count for the UI.
  const merged = nms(detections, 0.6).map(({ label, score, box, model: m }) => ({
    label,
    score,
    box,
    model: m,
  }));

  return {
    width,
    height,
    detections: merged,
    mask: ctx.mask,
    labelMasks: ctx.labelMasks || null,
  };
}

module.exports = {
  detect,
  setModelsDir,
  setUserModelsDir,
  availableModels,
  unloadModel,
  MODELS,
  DEFAULT_ENSEMBLE,
};
