'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const os = require('os');
const path = require('path');
const { spawn, execFile } = require('child_process');
const { CLASSES } = require('./labels');

/**
 * Fine-tuning runs in the user's own Python, not in the app: ultralytics pulls
 * in torch, which is far too large to bundle. The app's job is to write the
 * config, stream the output, and install whatever comes out as a selectable
 * model.
 */

/**
 * Base weight presets. Prefer pre-trained anime/NSFW weights over generic COCO
 * YOLO weights for much better fine-tuning results.
 */
const BASE_PRESETS = [
  {
    key: 'anime_medium',
    label: 'anime-medium (권장 · 1280px 애니 7클래스 기본)',
    file: 'nsfw-anime-medium-x1280.pt',
    default: true,
  },
  {
    key: 'anime_nano',
    label: 'anime-nano (640px · 빠른 학습용)',
    file: 'nsfw-anime-nano-x640.pt',
  },
  {
    key: 'anime_xl',
    label: 'anime-xl (1280px · 초대형 최고성능)',
    file: 'nsfw-anime-xl-x1280.pt',
  },
  {
    key: 'yolo11n-seg',
    label: 'yolo11n-seg (초경량 YOLO COCO 기본)',
    file: 'yolo11n-seg.pt',
    isStandard: true,
  },
  {
    key: 'yolo11s-seg',
    label: 'yolo11s-seg (표준 YOLO COCO 기본)',
    file: 'yolo11s-seg.pt',
    isStandard: true,
  },
];

function resolveBaseWeight(base, userModelsDir) {
  if (!base) return 'nsfw-anime-medium-x1280.pt';

  // Check if it matches a preset key or filename
  const preset = BASE_PRESETS.find((p) => p.key === base || p.file === base);
  const targetFile = preset ? preset.file : base;

  // If already absolute or relative existing file
  if (fs.existsSync(targetFile)) return path.resolve(targetFile);

  // Search candidate directories for the .pt file
  const candidates = [
    userModelsDir ? path.join(userModelsDir, targetFile) : null,
    path.join(__dirname, '..', 'tools', 'src-models', targetFile),
    path.join(__dirname, '..', 'models', targetFile),
    process.resourcesPath ? path.join(process.resourcesPath, 'models', targetFile) : null,
    process.resourcesPath ? path.join(process.resourcesPath, 'src-models', targetFile) : null,
  ].filter(Boolean);

  for (const c of candidates) {
    if (fs.existsSync(c)) return path.resolve(c);
  }

  if (preset && preset.isStandard) {
    return targetFile;
  }

  return targetFile;
}

const PY_SCRIPT = `
import json, sys
from pathlib import Path
from ultralytics import YOLO

cfg = json.loads(Path(sys.argv[1]).read_text())
patience = int(cfg.get("patience", 25))
model = YOLO(cfg["base"])
model.train(
    data=cfg["data"],
    epochs=cfg["epochs"],
    imgsz=cfg["imgsz"],
    batch=cfg["batch"],
    device=cfg["device"],
    project=cfg["project"],
    name=cfg["name"],
    patience=patience,
    save=True,
    exist_ok=True,
    verbose=True,
    plots=False,
)

best = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
trained = YOLO(str(best))
onnx = trained.export(format="onnx", imgsz=cfg["imgsz"], opset=12, dynamic=False, simplify=False)
print("AUTOCENSOR_ONNX::" + str(onnx), flush=True)
print("AUTOCENSOR_NAMES::" + json.dumps(trained.names, ensure_ascii=False), flush=True)
print("AUTOCENSOR_TASK::" + str(trained.task), flush=True)
`;

function run(bin, args, timeout = 20000) {
  return new Promise((resolve, reject) => {
    execFile(bin, args, { timeout }, (err, stdout, stderr) => {
      if (err) reject(new Error(stderr || err.message));
      else resolve(stdout);
    });
  });
}

/** @returns {{bin, version, torch}|null} */
async function findPython(extraPaths = []) {
  const probe = [
    ...extraPaths,
    process.env.AUTOCENSOR_PYTHON,
    'python3',
    'python',
    '/opt/homebrew/bin/python3',
    '/usr/local/bin/python3',
  ].filter(Boolean);

  for (const bin of probe) {
    try {
      const out = await run(bin, [
        '-c',
        'import ultralytics, torch; print(ultralytics.__version__); print(torch.__version__); print(torch.cuda.is_available()); print(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())',
      ]);
      const [version, torchVer, cudaAvail, mpsAvail] = out.trim().split('\n');
      const hasCuda = cudaAvail === 'True';
      const hasMps = mpsAvail === 'True';
      return { bin, version, torch: torchVer, hasCuda, hasMps };
    } catch {
      // Try the next candidate.
    }
  }
  return null;
}

function defaultDevice(pyEnv) {
  if (pyEnv && pyEnv.hasCuda) return '0';
  if (pyEnv && pyEnv.hasMps) return 'mps';
  if (process.platform === 'darwin' && process.arch === 'arm64') return 'mps';
  return 'cpu';
}

async function datasetReady(datasetDir) {
  const yaml = path.join(datasetDir, 'data.yaml');
  const images = path.join(datasetDir, 'images', 'train');
  if (!fs.existsSync(yaml) || !fs.existsSync(images)) {
    return { ok: false, reason: 'data.yaml 또는 images/train 이 없습니다.' };
  }
  const files = (await fsp.readdir(images)).filter((f) => /\.(png|jpg|jpeg)$/i.test(f));
  if (files.length === 0) return { ok: false, reason: '학습 이미지가 없습니다.' };
  return { ok: true, samples: files.length };
}

let child = null;

function isRunning() {
  return !!child;
}

function cancel() {
  if (!child) return false;
  child.kill('SIGTERM');
  return true;
}

/**
 * @param onEvent receives {type:'log'|'done'|'error', ...}
 */
async function startTraining(opts, onEvent) {
  if (child) throw new Error('학습이 이미 진행 중입니다.');

  const {
    python,
    datasetDir,
    base = 'anime_medium',
    epochs = 100,
    imgsz = 640,
    batch = 8,
    patience = 25,
    device = defaultDevice(),
    runsDir,
    userModelsDir,
    label,
  } = opts;

  const ready = await datasetReady(datasetDir);
  if (!ready.ok) throw new Error(ready.reason);

  const baseWeight = resolveBaseWeight(base, userModelsDir);
  const name = `autocensor_${new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14)}`;
  const workDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'autocensor-train-'));
  const scriptFile = path.join(workDir, 'train.py');
  const cfgFile = path.join(workDir, 'cfg.json');

  await fsp.writeFile(scriptFile, PY_SCRIPT);
  await fsp.writeFile(
    cfgFile,
    JSON.stringify({
      base: baseWeight,
      data: path.join(datasetDir, 'data.yaml'),
      epochs,
      imgsz,
      batch,
      patience,
      device,
      project: runsDir,
      name,
    })
  );

  onEvent({
    type: 'log',
    text: `데이터셋 ${ready.samples}장 · base=${path.basename(baseWeight)} · epochs=${epochs} · imgsz=${imgsz} · batch=${batch} · patience=${patience} · device=${device}\n`,
  });

  return new Promise((resolve) => {
    // -u so ultralytics' progress lines arrive while the run is going, not at the end.
    child = spawn(python, ['-u', scriptFile, cfgFile], {
      cwd: workDir,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let onnxPath = null;
    let names = null;
    let task = 'segment';
    let tail = '';

    const handle = (buf) => {
      const text = buf.toString();
      tail = (tail + text).slice(-4000);
      for (const line of text.split('\n')) {
        if (line.startsWith('AUTOCENSOR_ONNX::')) onnxPath = line.slice(17).trim();
        else if (line.startsWith('AUTOCENSOR_NAMES::')) {
          try {
            names = JSON.parse(line.slice(18));
          } catch {
            names = null;
          }
        } else if (line.startsWith('AUTOCENSOR_TASK::')) task = line.slice(17).trim();
      }
      onEvent({ type: 'log', text });
    };

    child.stdout.on('data', handle);
    child.stderr.on('data', handle);

    child.on('error', (err) => {
      child = null;
      onEvent({ type: 'error', message: err.message });
      resolve({ ok: false, message: err.message });
    });

    child.on('close', async (code, signal) => {
      child = null;
      if (signal) {
        onEvent({ type: 'error', message: `중지됨 (${signal})` });
        resolve({ ok: false, cancelled: true });
        return;
      }
      if (code !== 0 || !onnxPath) {
        const message = code !== 0 ? `python 종료 코드 ${code}` : 'ONNX 내보내기 결과를 찾지 못했습니다.';
        onEvent({ type: 'error', message, tail });
        resolve({ ok: false, message });
        return;
      }
      try {
        const installed = await installModel({
          onnxPath,
          names,
          task,
          imgsz,
          userModelsDir,
          name,
          label,
          baseWeight,
          epochs,
        });
        onEvent({ type: 'done', model: installed });
        resolve({ ok: true, model: installed });
      } catch (err) {
        onEvent({ type: 'error', message: err.message });
        resolve({ ok: false, message: err.message });
      }
    });
  });
}

/** Copy the exported ONNX into the user model folder and describe it. */
async function installModel(opts) {
  const { onnxPath, names, task, imgsz, userModelsDir, name, label, baseWeight, epochs } = opts;
  await fsp.mkdir(userModelsDir, { recursive: true });

  const fileName = `${name}.onnx`;
  await fsp.copyFile(onnxPath, path.join(userModelsDir, fileName));

  // Training used our own data.yaml, so the model's class order is the canonical
  // one; fall back to it if ultralytics did not report names.
  const ordered = names
    ? Object.keys(names)
        .map(Number)
        .sort((a, b) => a - b)
        .map((i) => names[i])
    : CLASSES.slice();

  const map = {};
  for (const c of ordered) if (CLASSES.includes(c)) map[c] = c;

  const baseTag = baseWeight ? path.basename(baseWeight, '.pt') : 'model';
  const spec = {
    key: name,
    label: label || `${baseTag} 파인튜닝 ${name.slice(-6)} (ep${epochs})`,
    file: fileName,
    size: imgsz,
    task: task === 'detect' ? 'detect' : 'segment',
    classes: ordered,
    map,
    trainedAt: new Date().toISOString(),
  };
  await fsp.writeFile(
    path.join(userModelsDir, `${name}.json`),
    JSON.stringify(spec, null, 2)
  );
  return spec;
}

module.exports = {
  findPython,
  startTraining,
  cancel,
  isRunning,
  datasetReady,
  defaultDevice,
  BASE_PRESETS,
  resolveBaseWeight,
};
