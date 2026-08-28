'use strict';

/**
 * Review grid + brush editor.
 *
 * A stroke does double duty: it fixes the mask for this image, and it is the
 * exact per-class annotation the detector needed, so the same canvases are what
 * gets written to the fine-tuning dataset.
 */
(() => {
  const $ = (id) => document.getElementById(id);
  let meta = null;
  let manifest = null;
  let entries = [];

  // Editor state
  let current = null; // manifest entry being edited
  let image = null; // HTMLImageElement of the original
  let maskCanvas = null; // the mask that will be censored (opaque black/white)
  let alphaCanvas = null; // same coverage with alpha set, for the screen overlay
  let tintCanvas = null; // reused so a redraw is two GPU ops, not a fresh alloc
  let classCanvas = {}; // label -> canvas, the training annotation
  let censoredUrl = null; // the saved output, loaded only if the user asks
  let resultImage = null;
  let viewResult = false;
  let undoStack = [];
  let tool = 'paint';
  let showMask = true;
  let drawing = false;
  let lastPt = null;

  const OVERLAY = 'rgba(255, 64, 96, 0.45)';

  /**
   * Mask canvases are opaque black/white, because that is what survives the trip
   * through PNG and sharp without an alpha channel to misread. That makes them
   * useless as a compositing source: `destination-in` clips by ALPHA, and every
   * pixel of an opaque mask has alpha 255, so nothing gets clipped and the tint
   * covers the whole image. Rewrite the pixels so coverage lives in the alpha
   * channel, and keep that copy purely for display.
   */
  function maskPixelsToAlpha(px) {
    for (let i = 0; i < px.length; i += 4) {
      const on = px[i] > 127;
      px[i] = 255;
      px[i + 1] = 255;
      px[i + 2] = 255;
      px[i + 3] = on ? 255 : 0;
    }
    return px;
  }

  function canvasOf(w, h) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    return c;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('image decode failed'));
      img.src = src;
    });
  }

  /** A mask PNG is black/white; draw it as an opaque white-on-black canvas. */
  async function maskCanvasFrom(dataUrl, w, h) {
    const c = canvasOf(w, h);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    if (dataUrl) {
      const img = await loadImage(dataUrl);
      ctx.drawImage(img, 0, 0, w, h);
    }
    return c;
  }

  /** Display-only copy of `maskCanvas` with coverage moved into alpha. */
  function syncAlpha() {
    const w = maskCanvas.width;
    const h = maskCanvas.height;
    const src = maskCanvas.getContext('2d', { willReadFrequently: true });
    const data = src.getImageData(0, 0, w, h);
    maskPixelsToAlpha(data.data);
    alphaCanvas.getContext('2d').putImageData(data, 0, 0);
  }

  function maskCoverage() {
    const w = maskCanvas.width;
    const h = maskCanvas.height;
    const d = maskCanvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h)
      .data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] > 127) n++;
    return n;
  }

  // ------------------------------------------------------------------ grid

  async function loadReview() {
    const dir = $('reviewDir').value;
    if (!dir) return;
    $('reviewStat').textContent = '불러오는 중...';
    manifest = await window.api.reviewLoad(dir);
    if (!manifest) {
      $('grid').innerHTML = '';
      $('reviewStat').textContent =
        '이 폴더에 _autocensor/manifest.json 이 없습니다. 마스크 저장을 켜고 배치를 돌리세요.';
      return;
    }
    entries = manifest.entries;
    renderGrid();
  }

  function passesFilter(en) {
    const f = $('reviewFilter').value;
    if (f === 'censored') return en.censored;
    if (f === 'clean') return !en.censored;
    if (f === 'corrected') return !!en.corrected;
    return true;
  }

  function baseName(p) {
    return p.split(/[\\/]/).pop();
  }

  async function renderGrid() {
    const grid = $('grid');
    grid.innerHTML = '';
    grid.style.setProperty('--thumb', `${$('thumbSize').value}px`);
    const shown = entries.filter(passesFilter);
    $('reviewStat').textContent = `${shown.length} / ${entries.length}개`;

    for (const en of shown) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      const tags = en.detections.length
        ? `<span class="hit">${en.detections
            .map((d) => meta.ko[d.label] || d.label)
            .filter((v, i, a) => a.indexOf(v) === i)
            .join(', ')}</span>`
        : '<span class="none">검출 없음</span>';
      cell.innerHTML = `
        <img alt="" />
        <div class="meta">
          <div class="name">${baseName(en.source)}</div>
          <div class="tags">${tags}${en.corrected ? ' <span class="fix">· 보정됨</span>' : ''}</div>
        </div>`;
      cell.onclick = () => openEditor(en);
      grid.appendChild(cell);

      window.api
        .reviewThumb({ file: en.dest, size: Number($('thumbSize').value) * 2 })
        .then((url) => {
          cell.querySelector('img').src = url;
        })
        .catch(() => {});
    }
  }

  // ---------------------------------------------------------------- editor

  async function openEditor(entry) {
    current = entry;
    $('editorName').textContent = `${baseName(entry.source)} (${entry.width}x${entry.height})`;
    $('editorStat').textContent = '불러오는 중...';
    $('editor').classList.remove('hidden');

    const data = await window.api.reviewOpen({ outputDir: $('reviewDir').value, entry });
    image = await loadImage(data.original);
    maskCanvas = await maskCanvasFrom(data.union, entry.width, entry.height);
    alphaCanvas = canvasOf(entry.width, entry.height);
    tintCanvas = canvasOf(entry.width, entry.height);
    syncAlpha();
    censoredUrl = null;
    resultImage = null;
    viewResult = false;
    $('viewResult').textContent = '결과 보기';

    classCanvas = {};
    for (const label of meta.classes) {
      classCanvas[label] = await maskCanvasFrom(data.labelMasks[label] || null, entry.width, entry.height);
    }

    undoStack = [];
    const stage = $('stage');
    stage.width = entry.width;
    stage.height = entry.height;
    redraw();
    const covered = maskCoverage();
    $('editorStat').textContent = `마스크 ${covered.toLocaleString()}px · 자동 검출: ${
      entry.detections.map((d) => `${meta.ko[d.label] || d.label} ${d.score.toFixed(2)}`).join(', ') ||
      '없음'
    }`;
  }

  function redraw() {
    const stage = $('stage');
    const ctx = stage.getContext('2d');
    const w = stage.width;
    const h = stage.height;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(viewResult && resultImage ? resultImage : image, 0, 0, w, h);
    if (viewResult || !showMask) return;

    // Tint the covered area rather than filling it, so the artwork underneath
    // stays visible while painting.
    const tctx = tintCanvas.getContext('2d');
    tctx.globalCompositeOperation = 'source-over';
    tctx.clearRect(0, 0, w, h);
    tctx.fillStyle = OVERLAY;
    tctx.fillRect(0, 0, w, h);
    tctx.globalCompositeOperation = 'destination-in';
    tctx.drawImage(alphaCanvas, 0, 0);
    ctx.drawImage(tintCanvas, 0, 0);
  }

  function stagePoint(ev) {
    const stage = $('stage');
    const r = stage.getBoundingClientRect();
    return {
      x: ((ev.clientX - r.left) / r.width) * stage.width,
      y: ((ev.clientY - r.top) / r.height) * stage.height,
    };
  }

  function strokeOn(canvas, from, to, size, erase) {
    const ctx = canvas.getContext('2d');
    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = erase ? '#000' : '#fff';
    ctx.fillStyle = erase ? '#000' : '#fff';
    ctx.lineWidth = size;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.restore();
  }

  /** The display copy carries coverage in alpha, so erasing must clear alpha. */
  function strokeOnAlpha(canvas, from, to, size, erase) {
    const ctx = canvas.getContext('2d');
    ctx.save();
    ctx.globalCompositeOperation = erase ? 'destination-out' : 'source-over';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = size;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.restore();
  }

  function pushUndo() {
    const snap = { mask: canvasOf(maskCanvas.width, maskCanvas.height), classes: {} };
    snap.mask.getContext('2d').drawImage(maskCanvas, 0, 0);
    for (const [label, c] of Object.entries(classCanvas)) {
      const s = canvasOf(c.width, c.height);
      s.getContext('2d').drawImage(c, 0, 0);
      snap.classes[label] = s;
    }
    undoStack.push(snap);
    if (undoStack.length > 12) undoStack.shift();
  }

  function applyStroke(from, to) {
    const size = Number($('brushSize').value);
    const erase = tool === 'erase';
    strokeOn(maskCanvas, from, to, size, erase);
    strokeOnAlpha(alphaCanvas, from, to, size, erase);
    if (erase) {
      // Erasing has to clear every class, otherwise a stale annotation would be
      // exported for a region the user just uncovered.
      for (const c of Object.values(classCanvas)) strokeOn(c, from, to, size, true);
    } else {
      strokeOn(classCanvas[$('brushClass').value], from, to, size, false);
    }
    redraw();
  }

  function bindStage() {
    const stage = $('stage');
    stage.onpointerdown = (ev) => {
      if (!image) return;
      stage.setPointerCapture(ev.pointerId);
      pushUndo();
      drawing = true;
      lastPt = stagePoint(ev);
      applyStroke(lastPt, lastPt);
    };
    stage.onpointermove = (ev) => {
      if (!drawing) return;
      const p = stagePoint(ev);
      applyStroke(lastPt, p);
      lastPt = p;
    };
    const end = () => {
      drawing = false;
      lastPt = null;
    };
    stage.onpointerup = end;
    stage.onpointercancel = end;
    stage.onpointerleave = end;
  }

  function classMaskUrls() {
    const out = {};
    for (const [label, c] of Object.entries(classCanvas)) {
      const ctx = c.getContext('2d', { willReadFrequently: true });
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      let any = false;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 127) {
          any = true;
          break;
        }
      }
      if (any) out[label] = c.toDataURL('image/png');
    }
    return out;
  }

  async function applyEdit() {
    if (!current) return;
    $('editorStat').textContent = '적용 중...';
    try {
      const r = await window.api.reviewApply({
        outputDir: $('reviewDir').value,
        entry: current,
        unionMask: maskCanvas.toDataURL('image/png'),
        classMasks: classMaskUrls(),
        censorOptions: window.censorOptions(),
      });
      current.corrected = true;
      current.labels = r.labels;
      $('editorStat').textContent = `저장됨 — 마스크 ${r.coveredPixels}px, 라벨 ${
        r.labels.join(', ') || '없음'
      }`;
      renderGrid();
    } catch (err) {
      $('editorStat').textContent = `오류: ${err.message}`;
    }
  }

  async function addToDataset() {
    const dir = $('datasetDir').value;
    if (!dir) {
      alert('사이드바에서 데이터셋 폴더를 먼저 지정하세요.');
      return;
    }
    $('editorStat').textContent = '데이터셋에 쓰는 중...';
    try {
      const r = await window.api.datasetAdd({
        datasetDir: dir,
        entry: current,
        classMasks: classMaskUrls(),
      });
      $('editorStat').textContent = `데이터셋: ${r.name} (폴리곤 ${r.polygons}개)`;
      showDatasetStats(r.stats);
    } catch (err) {
      $('editorStat').textContent = `오류: ${err.message}`;
    }
  }

  function showDatasetStats(stats) {
    if (!stats) return;
    const parts = Object.entries(stats.classes).map(([k, v]) => `${meta.ko[k] || k} ${v}`);
    $('datasetStat').textContent = `${stats.samples}장 / 폴리곤 ${stats.polygons}${
      parts.length ? ` · ${parts.join(', ')}` : ''
    }`;
  }

  // ------------------------------------------------------------------ init

  window.initReview = (m) => {
    meta = m;
    const sel = $('brushClass');
    sel.innerHTML = meta.classes
      .map((c) => `<option value="${c}">${meta.ko[c] || c}</option>`)
      .join('');
    sel.value = 'nipple';

    $('pickReview').onclick = async () => {
      const d = await window.api.pickFolder('결과 폴더 선택');
      if (d) {
        $('reviewDir').value = d;
        loadReview();
      }
    };
    $('reloadReview').onclick = loadReview;
    $('reviewFilter').onchange = renderGrid;
    $('thumbSize').oninput = () => {
      $('grid').style.setProperty('--thumb', `${$('thumbSize').value}px`);
    };
    $('thumbSize').onchange = renderGrid;

    $('brushSize').oninput = () => ($('brushVal').textContent = $('brushSize').value);
    $('toolPaint').onclick = () => {
      tool = 'paint';
      $('toolPaint').classList.add('primary');
      $('toolErase').classList.remove('primary');
    };
    $('toolErase').onclick = () => {
      tool = 'erase';
      $('toolErase').classList.add('primary');
      $('toolPaint').classList.remove('primary');
    };
    $('toggleMask').onclick = () => {
      showMask = !showMask;
      $('toggleMask').textContent = showMask ? '마스크 보기' : '마스크 숨김';
      redraw();
    };
    $('undoStroke').onclick = () => {
      const snap = undoStack.pop();
      if (!snap) return;
      maskCanvas.getContext('2d').drawImage(snap.mask, 0, 0);
      for (const [label, c] of Object.entries(snap.classes)) {
        const ctx = classCanvas[label].getContext('2d');
        ctx.clearRect(0, 0, c.width, c.height);
        ctx.drawImage(c, 0, 0);
      }
      syncAlpha();
      redraw();
    };
    $('viewResult').onclick = async () => {
      viewResult = !viewResult;
      $('viewResult').textContent = viewResult ? '원본 보기' : '결과 보기';
      if (viewResult && !resultImage) {
        censoredUrl = await window.api.reviewDest(current.dest);
        resultImage = await loadImage(censoredUrl);
      }
      redraw();
    };
    $('applyEdit').onclick = applyEdit;
    $('addDataset').onclick = addToDataset;
    $('closeEditor').onclick = () => {
      $('editor').classList.add('hidden');
      current = null;
    };

    $('pickDataset').onclick = async () => {
      const d = await window.api.pickFolder('데이터셋 폴더 선택');
      if (!d) return;
      $('datasetDir').value = d;
      showDatasetStats(await window.api.datasetStats(d));
    };

    document.addEventListener('keydown', (ev) => {
      if ($('editor').classList.contains('hidden')) return;
      if (ev.key === 'Escape') $('closeEditor').click();
      if (ev.key === 'e') $('toolErase').click();
      if (ev.key === 'b') $('toolPaint').click();
      if ((ev.metaKey || ev.ctrlKey) && ev.key === 'z') $('undoStroke').click();
    });

    bindStage();
  };

  // Exposed for tests: the pixel rewrite is the whole fix.
  window.reviewInternals = { maskPixelsToAlpha };

  window.reviewSetDir = (dir) => {
    if (!$('reviewDir').value) $('reviewDir').value = dir;
  };
})();
