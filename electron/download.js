'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const http = require('http');
const https = require('https');
const { URL } = require('url');

const DOWNLOAD_REGISTRY = {
  // Detection ONNX models (Pre-converted)
  'anime-medium': {
    file: 'nsfw-anime-medium-x1280.onnx',
    type: 'onnx',
    label: 'anime-medium (1280px, 정밀 · 추천)',
    url: 'https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/main/nsfw-anime-medium-x1280.onnx',
    sizeBytes: 47600269,
  },
  'anime-xl': {
    file: 'nsfw-anime-xl-x1280.onnx',
    type: 'onnx',
    label: 'anime-xl (1280px, 초대형 최고성능)',
    url: 'https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/main/nsfw-anime-xl-x1280.onnx',
    sizeBytes: 126350117,
  },
  'anime-nano': {
    file: 'nsfw-anime-nano-x640.onnx',
    type: 'onnx',
    label: 'anime-nano (640px, 빠름)',
    url: 'https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/main/nsfw-anime-nano-x640.onnx',
    sizeBytes: 5902685,
  },

  // Fine-tuning PyTorch base weights (.pt)
  anime_medium: {
    file: 'nsfw-anime-medium-x1280.pt',
    type: 'pt',
    label: 'anime-medium 가중치 (1280px .pt)',
    url: 'https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/main/nsfw-anime-medium-x1280.pt',
    sizeBytes: 47600269,
  },
  anime_nano: {
    file: 'nsfw-anime-nano-x640.pt',
    type: 'pt',
    label: 'anime-nano 가중치 (640px .pt)',
    url: 'https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/main/nsfw-anime-nano-x640.pt',
    sizeBytes: 5902685,
  },
  anime_xl: {
    file: 'nsfw-anime-xl-x1280.pt',
    type: 'pt',
    label: 'anime-xl 가중치 (1280px 초대형 .pt)',
    url: 'https://huggingface.co/01miku/anime-nsfw-segm-yolo26/resolve/main/nsfw-anime-xl-x1280.pt',
    sizeBytes: 141836677,
  },
  'yolo11n-seg': {
    file: 'yolo11n-seg.pt',
    type: 'pt',
    label: 'yolo11n-seg 가중치 (.pt)',
    url: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt',
    sizeBytes: 6000000,
  },
  'yolo11s-seg': {
    file: 'yolo11s-seg.pt',
    type: 'pt',
    label: 'yolo11s-seg 가중치 (.pt)',
    url: 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt',
    sizeBytes: 20000000,
  },
};

let activeDownload = null;

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/**
 * Downloads a file from url to destPath following redirects.
 * @param {string} url
 * @param {string} destPath
 * @param {object} opts { token, headers }
 * @param {function} onProgress ({ downloaded, total, percent, speed, key })
 */
function downloadFile(url, destPath, opts = {}, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const tempPath = `${destPath}.download_${Date.now()}`;
    let fileStream = null;
    let req = null;
    let aborted = false;

    const cleanup = async () => {
      if (fileStream) {
        fileStream.destroy();
        fileStream = null;
      }
      try {
        if (fs.existsSync(tempPath)) await fsp.unlink(tempPath);
      } catch {
        // ignore cleanup error
      }
    };

    activeDownload = {
      abort: async () => {
        aborted = true;
        if (req) req.destroy();
        await cleanup();
        reject(new Error('다운로드가 취소되었습니다.'));
      },
    };

    const fetchWithRedirects = (currentUrl, redirectCount = 0) => {
      if (aborted) return;
      if (redirectCount > 10) {
        cleanup().then(() => reject(new Error('리다이렉트 횟수가 너무 많습니다.')));
        return;
      }

      const parsed = new URL(currentUrl);
      const isHttps = parsed.protocol === 'https:';
      const client = isHttps ? https : http;

      const headers = {
        'User-Agent': 'AutoCensor/1.0',
        ...(opts.headers || {}),
      };

      const token = opts.token || process.env.HF_TOKEN;
      if (token && parsed.hostname === 'huggingface.co') {
        headers['Authorization'] = `Bearer ${token}`;
      } else {
        delete headers['Authorization'];
      }

      req = client.get(currentUrl, { headers }, (res) => {
        if (aborted) return;

        // Handle redirects (301, 302, 303, 307, 308)
        if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
          res.resume();
          const redirectUrl = new URL(res.headers.location, currentUrl).toString();
          fetchWithRedirects(redirectUrl, redirectCount + 1);
          return;
        }

        if (res.statusCode < 200 || res.statusCode >= 300) {
          res.resume();
          const msg =
            res.statusCode === 401 || res.statusCode === 403
              ? `접근 권한 오류 (${res.statusCode}): HuggingFace 토큰이 필요하거나 비공개 모델입니다.`
              : `다운로드 실패 (HTTP ${res.statusCode})`;
          cleanup().then(() => reject(new Error(msg)));
          return;
        }

        const totalBytes = parseInt(res.headers['content-length'] || '0', 10);
        let downloadedBytes = 0;
        let lastTime = Date.now();
        let lastBytes = 0;

        fileStream = fs.createWriteStream(tempPath);

        res.on('data', (chunk) => {
          if (aborted) return;
          downloadedBytes += chunk.length;
          const now = Date.now();
          if (now - lastTime >= 200 || downloadedBytes === totalBytes) {
            const timeDiff = (now - lastTime) / 1000;
            const bytesDiff = downloadedBytes - lastBytes;
            const speed = timeDiff > 0 ? bytesDiff / timeDiff : 0;
            const percent = totalBytes > 0 ? (downloadedBytes / totalBytes) * 100 : 0;

            onProgress({
              downloaded: downloadedBytes,
              total: totalBytes,
              percent: Math.min(100, percent),
              speed,
              formattedDownloaded: formatBytes(downloadedBytes),
              formattedTotal: formatBytes(totalBytes),
              formattedSpeed: `${formatBytes(speed)}/s`,
            });

            lastTime = now;
            lastBytes = downloadedBytes;
          }
        });

        res.pipe(fileStream);

        fileStream.on('finish', async () => {
          if (aborted) return;
          try {
            fileStream.close();
            // Ensure parent directory exists
            await fsp.mkdir(path.dirname(destPath), { recursive: true });
            // Move temp file to destination
            if (fs.existsSync(destPath)) await fsp.unlink(destPath);
            await fsp.rename(tempPath, destPath);
            activeDownload = null;
            resolve({ destPath, size: downloadedBytes });
          } catch (err) {
            await cleanup();
            activeDownload = null;
            reject(err);
          }
        });

        fileStream.on('error', async (err) => {
          await cleanup();
          activeDownload = null;
          reject(err);
        });
      });

      req.on('error', async (err) => {
        if (aborted) return;
        await cleanup();
        activeDownload = null;
        reject(new Error(`네트워크 오류: ${err.message}`));
      });
    };

    fetchWithRedirects(url);
  });
}

function cancelDownload() {
  if (activeDownload) {
    activeDownload.abort();
    activeDownload = null;
    return true;
  }
  return false;
}

function isDownloading() {
  return !!activeDownload;
}

/**
 * Returns installation status for all models in DOWNLOAD_REGISTRY.
 */
function getModelsStatus(dirs = {}) {
  const { modelsDir, userModelsDir, srcModelsDir } = dirs;
  const list = [];

  for (const [key, item] of Object.entries(DOWNLOAD_REGISTRY)) {
    const candidates = [
      modelsDir ? path.join(modelsDir, item.file) : null,
      userModelsDir ? path.join(userModelsDir, item.file) : null,
      srcModelsDir ? path.join(srcModelsDir, item.file) : null,
    ].filter(Boolean);

    let installedPath = null;
    let installed = false;
    let size = 0;

    for (const c of candidates) {
      if (fs.existsSync(c)) {
        installed = true;
        installedPath = c;
        try {
          size = fs.statSync(c).size;
        } catch {
          // ignore
        }
        break;
      }
    }

    list.push({
      key,
      file: item.file,
      type: item.type,
      label: item.label,
      url: item.url,
      expectedSize: item.sizeBytes,
      installed,
      installedPath,
      size,
      formattedSize: formatBytes(size || item.sizeBytes),
    });
  }

  return list;
}

/**
 * Download a specific model from registry.
 */
async function downloadModel(key, dirs = {}, opts = {}, onProgress = () => {}) {
  const item = DOWNLOAD_REGISTRY[key];
  if (!item) throw new Error(`알 수 없는 모델 키: ${key}`);

  const targetDir =
    item.type === 'pt'
      ? dirs.userModelsDir || dirs.srcModelsDir || dirs.modelsDir
      : dirs.userModelsDir || dirs.modelsDir;

  if (!targetDir) throw new Error('저장 대상 폴더를 찾을 수 없습니다.');
  await fsp.mkdir(targetDir, { recursive: true });

  const destPath = path.join(targetDir, item.file);

  onProgress({
    key,
    stage: 'start',
    label: item.label,
    percent: 0,
  });

  const result = await downloadFile(item.url, destPath, opts, (p) => {
    onProgress({
      key,
      stage: 'progress',
      label: item.label,
      ...p,
    });
  });

  onProgress({
    key,
    stage: 'done',
    label: item.label,
    percent: 100,
    destPath: result.destPath,
  });

  return { key, ...item, destPath: result.destPath };
}

module.exports = {
  DOWNLOAD_REGISTRY,
  downloadFile,
  downloadModel,
  cancelDownload,
  isDownloading,
  getModelsStatus,
  formatBytes,
};
