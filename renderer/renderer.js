'use strict';

const $ = (id) => document.getElementById(id);
let meta = null;
let labelConfig = null;
let previewFile = null;
let running = false;

const modelStrengths = {};

function detectOptions() {
  const configs = {};
  for (const k of selectedModels()) {
    configs[k] = { strength: modelStrengths[k] !== undefined ? modelStrengths[k] : 1.0 };
  }
  return {
    models: selectedModels(),
    modelConfigs: configs,
    tiling: $('tiling').value,
    labelConfig,
  };
}

function selectedModels() {
  return [...document.querySelectorAll('#models input[type=checkbox]:checked')].map((el) => el.value);
}

let downloadingModelKey = null;

// A freshly trained model or newly downloaded model has to show up without restarting the app.
window.refreshModels = async () => {
  const keep = new Set(selectedModels());
  meta.models = await window.api.modelsList();
  renderModels();
  for (const el of document.querySelectorAll('#models input[type=checkbox]')) {
    if (!el.disabled) {
      el.checked = keep.has(el.value) || (keep.size === 0 && meta.defaultModels.includes(el.value));
    }
  }
};

function renderModels() {
  const box = $('models');
  box.innerHTML = '';
  let missingCount = 0;

  for (const m of meta.models) {
    const row = document.createElement('div');
    row.className = 'model-row';
    const isInstalled = m.installed !== false;
    if (!isInstalled) missingCount++;

    const on = isInstalled && meta.defaultModels.includes(m.key);
    const strVal = modelStrengths[m.key] !== undefined ? modelStrengths[m.key] : 1.0;

    const checkHtml = isInstalled
      ? `<input type="checkbox" value="${m.key}" ${on ? 'checked' : ''} />`
      : `<input type="checkbox" value="${m.key}" disabled title="다운로드 필요" />`;

    const dlBtnHtml = !isInstalled
      ? `<button class="btn-sm btn-dl primary" data-key="${m.key}" title="Hugging Face에서 모델 가중치 다운로드">다운로드</button>`
      : '';

    const delBtnHtml = isInstalled
      ? `<button class="btn-sm btn-del" data-key="${m.key}" title="${m.custom ? '학습된 모델 삭제' : '로컬 모델 파일 삭제'}">삭제</button>`
      : '';

    const strengthHtml = isInstalled
      ? `<div class="model-strength-wrap" title="모델 민감도/강도 (0.5x~2.0x)">
           <input type="range" class="model-strength-slider" min="0.5" max="2.0" step="0.1" value="${strVal}" />
           <span class="model-strength-val">${strVal.toFixed(1)}x</span>
         </div>`
      : '';

    row.innerHTML = `
      <label class="model-label">
        ${checkHtml} <span>${m.label}</span>
      </label>
      ${m.custom ? '<span class="badge">학습됨</span>' : ''}
      ${strengthHtml}
      ${delBtnHtml}
      ${dlBtnHtml}
    `;

    const slider = row.querySelector('.model-strength-slider');
    const valSpan = row.querySelector('.model-strength-val');
    if (slider && valSpan) {
      slider.oninput = (e) => {
        const val = Number(e.target.value);
        modelStrengths[m.key] = val;
        valSpan.textContent = `${val.toFixed(1)}x`;
      };
    }

    const dlBtn = row.querySelector('.btn-dl');
    if (dlBtn) {
      dlBtn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        await startModelDownload(m.key, dlBtn);
      };
    }

    const delBtn = row.querySelector('.btn-del');
    if (delBtn) {
      delBtn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const msg = m.custom
          ? `'${m.label}' 학습 모델을 정말 삭제할까요?`
          : `'${m.label}' 모델 파일을 로컬에서 삭제할까요? (언제든 다시 다운로드할 수 있습니다)`;
        if (confirm(msg)) {
          delBtn.disabled = true;
          try {
            await window.api.modelDelete(m.key);
            await window.refreshModels();
          } catch (err) {
            alert(`모델 삭제 실패: ${err.message}`);
          }
        }
      };
    }

    box.appendChild(row);
  }

  // Quick banner at the top of the app
  const quickBanner = $('quickSetupBanner');
  if (quickBanner) {
    if (missingCount > 0) {
      quickBanner.classList.remove('hidden');
      $('btnQuickSetup').onclick = downloadAllMissing;
    } else {
      quickBanner.classList.add('hidden');
    }
  }

  if (missingCount > 0) {
    const allRow = document.createElement('div');
    allRow.className = 'model-all-row';
    allRow.innerHTML = `<button id="dlAllModels" class="btn-sm primary">누락된 모델 다운로드 (${missingCount}개)</button>`;
    box.appendChild(allRow);
    allRow.querySelector('#dlAllModels').onclick = async () => {
      await downloadAllMissing();
    };
  }

  const progBox = document.createElement('div');
  progBox.id = 'modelDlProg';
  progBox.className = 'model-dl-prog hidden';
  progBox.innerHTML = `
    <div class="dl-info"><span id="dlModelName"></span><span id="dlModelSpeed" class="muted"></span></div>
    <div class="progress"><div id="dlBar"></div></div>
  `;
  box.appendChild(progBox);
}

function showDlProgress(show, label) {
  const el = $('modelDlProg');
  if (!el) return;
  if (show) {
    el.classList.remove('hidden');
    if (label) $('dlModelName').textContent = label;
    $('dlBar').style.width = '0%';
    $('dlModelSpeed').textContent = '';
  } else {
    el.classList.add('hidden');
  }
}

async function startModelDownload(key, btn) {
  if (downloadingModelKey) return;
  downloadingModelKey = key;
  if (btn) btn.disabled = true;
  showDlProgress(true, `${key} 다운로드 준비 중...`);

  try {
    await window.api.modelDownload({ key });
    await window.refreshModels();
  } catch (err) {
    alert(`모델 다운로드 실패: ${err.message}`);
  } finally {
    downloadingModelKey = null;
    showDlProgress(false);
  }
}

async function downloadAllMissing() {
  if (downloadingModelKey) return;
  const missing = (meta.models || []).filter((m) => m.installed === false);
  if (missing.length === 0) return;

  for (const m of missing) {
    downloadingModelKey = m.key;
    showDlProgress(true, `${m.label} 다운로드 중...`);
    try {
      await window.api.modelDownload({ key: m.key });
    } catch (err) {
      alert(`${m.label} 다운로드 실패: ${err.message}`);
      break;
    }
  }
  downloadingModelKey = null;
  showDlProgress(false);
  await window.refreshModels();
}

function onModelDownloadProgress(ev) {
  if (ev.stage === 'start') {
    showDlProgress(true, `${ev.label || ev.key} 다운로드 시작...`);
  } else if (ev.stage === 'progress') {
    $('dlModelName').textContent = `${ev.label || ev.key} (${ev.formattedDownloaded || ''} / ${ev.formattedTotal || ''})`;
    $('dlModelSpeed').textContent = ev.formattedSpeed || '';
    $('dlBar').style.width = `${ev.percent || 0}%`;
  } else if (ev.stage === 'done') {
    $('dlBar').style.width = '100%';
    $('dlModelSpeed').textContent = '완료';
  } else if (ev.stage === 'error') {
    showDlProgress(false);
  }
}

function resetToRecommendedDefaults() {
  for (const m of meta.models || []) {
    modelStrengths[m.key] = 1.0;
  }
  for (const el of document.querySelectorAll('#models input[type=checkbox]')) {
    if (!el.disabled) {
      el.checked = meta.defaultModels.includes(el.value);
    }
  }
  for (const el of document.querySelectorAll('.model-strength-slider')) {
    el.value = '1.0';
  }
  for (const el of document.querySelectorAll('.model-strength-val')) {
    el.textContent = '1.0x';
  }

  $('tiling').value = 'auto';
  $('mode').value = 'white';
  $('areolaMethod').value = 'off';
  $('shape').value = 'contour';
  $('dilate').value = '4';
  $('dilateVal').textContent = '4';
  $('feather').value = '10';
  $('featherVal').textContent = '10';
  $('gamma').value = '0.7';
  $('gammaVal').textContent = '0.70';
  $('scaleRes').checked = true;
  $('strength').value = '1';
  $('strengthVal').textContent = '1.0';
  $('cgTol').value = '0.35';
  $('cgTolVal').textContent = '0.35';
  $('cgMax').value = '5';
  $('cgMaxVal').textContent = '5.0';
  $('cgFb').value = '2.6';
  $('cgFbVal').textContent = '2.6';
  $('mode').onchange();

  labelConfig = JSON.parse(JSON.stringify(meta.labelConfig));
  renderLabels();
  alert('✨ 추천 최적 설정으로 복원되었습니다.');
}

/**
 * renderer.js is a classic script, so a top-level `function foo()` IS
 * `window.foo`. Exposing it as `window.foo = () => foo()` would overwrite that
 * binding with the wrapper, and the wrapper's own call would then resolve to
 * itself - infinite recursion. Export the function by reference under a
 * distinct name instead.
 */
function currentCensorOptions() {
  return {
    mode: $('mode').value,
    shape: $('shape').value,
    dilateRadius: Number($('dilate').value),
    featherRadius: Number($('feather').value),
    edgeGamma: Number($('gamma').value),
    scaleWithResolution: $('scaleRes').checked,
    strength: Number($('strength').value),
  };
}

// The review editor re-censors with the same settings the batch used.
window.censorOptions = currentCensorOptions;

function renderLabels() {
  const box = $('labels');
  box.innerHTML =
    '<div class="label-row label-head"><span>부위</span><span>임계</span><span>확장</span></div>';
  for (const cls of meta.classes) {
    const cfg = labelConfig[cls];
    const row = document.createElement('div');
    row.className = 'label-row';
    row.innerHTML = `
      <label><input type="checkbox" ${cfg.enabled ? 'checked' : ''} /> ${meta.ko[cls] || cls}</label>
      <input type="number" min="0.05" max="0.95" step="0.05" value="${cfg.threshold}" />
      <input type="number" min="1" max="5" step="0.05" value="${cfg.expand}" />`;
    const [chk, thr, exp] = row.querySelectorAll('input');
    chk.onchange = () => (cfg.enabled = chk.checked);
    thr.onchange = () => (cfg.threshold = Number(thr.value));
    exp.onchange = () => (cfg.expand = Number(exp.value));
    box.appendChild(row);
  }
}

function log(cls, text) {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  $('log').appendChild(line);
  $('log').scrollTop = $('log').scrollHeight;
}

function baseName(p) {
  return p.split(/[\\/]/).pop();
}

async function init() {
  meta = await window.api.meta();
  labelConfig = meta.labelConfig;
  renderModels();
  renderLabels();
  if (window.initReview) window.initReview(meta);
  if (window.initTrain) window.initTrain(meta);

  if ($('btnResetDefaults')) {
    $('btnResetDefaults').onclick = resetToRecommendedDefaults;
  }

  document.querySelectorAll('.tab').forEach((t) => {
    t.onclick = () => {
      document.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((x) => x.classList.remove('active'));
      t.classList.add('active');
      $(`tab-${t.dataset.tab}`).classList.add('active');
      if (t.dataset.tab === 'train' && window.drawTrainCharts) {
        setTimeout(window.drawTrainCharts, 20);
      }
    };
  });

  for (const [slider, out, fmt] of [
    ['dilate', 'dilateVal', (v) => v],
    ['feather', 'featherVal', (v) => v],
    ['gamma', 'gammaVal', (v) => Number(v).toFixed(2)],
    ['strength', 'strengthVal', (v) => Number(v).toFixed(1)],
  ]) {
    $(slider).oninput = () => ($(out).textContent = fmt($(slider).value));
  }

  const cg = labelConfig.nipple.areola;
  if (cg && cg.method) {
    $('areolaMethod').value = cg.method;
  }
  $('areolaMethod').onchange = () => (cg.method = $('areolaMethod').value);
  const bindGrow = (id, out, key, fmt) => {
    const el = $(id);
    el.oninput = () => {
      cg[key] = Number(el.value);
      $(out).textContent = fmt(el.value);
    };
  };
  bindGrow('cgTol', 'cgTolVal', 'tolerance', (v) => Number(v).toFixed(2));
  bindGrow('cgMax', 'cgMaxVal', 'maxScale', (v) => Number(v).toFixed(1));
  bindGrow('cgFb', 'cgFbVal', 'fallbackExpand', (v) => Number(v).toFixed(1));

  $('mode').onchange = () => {
    const needsStrength = ['mosaic', 'blur'].includes($('mode').value);
    $('strengthField').style.display = needsStrength ? '' : 'none';
  };
  $('mode').onchange();

  $('inputDir').onchange = () => {
    const val = $('inputDir').value.trim();
    if (val && !$('outputDir').value) {
      $('outputDir').value = `${val}_censored`;
    }
  };

  const updateSidebarDataset = async () => {
    const dir = $('datasetDir').value.trim();
    if (!dir) {
      $('datasetStat').textContent = '미설정';
      return;
    }
    try {
      const ready = await window.api.trainDatasetReady(dir);
      if (ready.ok) {
        const stats = await window.api.datasetStats(dir);
        const parts = Object.entries(stats.classes).map(([k, v]) => `${meta.ko[k] || k} ${v}`);
        $('datasetStat').textContent = `${ready.samples}장 · 폴리곤 ${stats.polygons}${
          parts.length ? ` · ${parts.join(', ')}` : ''
        }`;
      } else {
        $('datasetStat').textContent = `사용 불가 — ${ready.reason}`;
      }
    } catch {
      $('datasetStat').textContent = '폴더 확인 불가';
    }
  };

  $('datasetDir').onchange = updateSidebarDataset;
  $('datasetDir').oninput = updateSidebarDataset;

  $('pickInput').onclick = async () => {
    const d = await window.api.pickFolder('입력 폴더 선택');
    if (d) {
      $('inputDir').value = d;
      if (!$('outputDir').value) $('outputDir').value = `${d}_censored`;
    }
  };
  $('pickOutput').onclick = async () => {
    const d = await window.api.pickFolder('출력 폴더 선택');
    if (d) $('outputDir').value = d;
  };
  $('pickDataset').onclick = async () => {
    const d = await window.api.pickFolder('데이터셋 저장 폴더 선택');
    if (d) {
      $('datasetDir').value = d;
      await updateSidebarDataset();
    }
  };
  $('openOut').onclick = () => window.api.openPath($('outputDir').value);

  $('start').onclick = startBatch;
  $('cancel').onclick = () => window.api.batchCancel();

  $('pickImage').onclick = async () => {
    const f = await window.api.pickImage();
    if (!f) return;
    previewFile = f;
    $('previewPath').textContent = f;
    $('runPreview').disabled = false;
  };
  $('runPreview').onclick = runPreview;

  if ($('btnRefreshModels')) {
    $('btnRefreshModels').onclick = async () => {
      $('btnRefreshModels').disabled = true;
      try {
        await window.refreshModels();
      } finally {
        $('btnRefreshModels').disabled = false;
      }
    };
  }

  window.addEventListener('focus', () => {
    if (window.refreshModels) window.refreshModels();
  });

  window.api.onBatchEvent(onBatchEvent);
  window.api.onModelDownloadProgress(onModelDownloadProgress);
}

async function startBatch() {
  const inputDir = $('inputDir').value;
  const outputDir = $('outputDir').value;
  if (!inputDir || !outputDir) {
    alert('입력 폴더와 출력 폴더를 모두 지정하세요.');
    return;
  }
  if (selectedModels().length === 0) {
    alert('검출 모델을 하나 이상 선택하세요.');
    return;
  }
  if (inputDir === outputDir && !confirm('입력 폴더와 출력 폴더가 같습니다. 원본을 덮어씁니다. 계속할까요?')) {
    return;
  }
  $('log').innerHTML = '';
  running = true;
  $('start').disabled = true;
  $('cancel').disabled = false;

  try {
    await window.api.batchStart({
      inputDir,
      outputDir,
      recursive: $('recursive').checked,
      overwrite: $('overwrite').checked,
      copyUnchanged: $('copyUnchanged').checked,
      suffix: $('suffix').value,
      format: $('format').value,
      quality: Number($('quality').value),
      saveMasks: $('saveMasks').checked,
      detectOptions: detectOptions(),
      censorOptions: currentCensorOptions(),
    });
  } catch (err) {
    // The stack is the only thing that identifies where a renderer-side failure
    // came from; without it a message like "Maximum call stack size exceeded"
    // says nothing about which call recursed.
    log('err', `실패: ${err.message}\n${err.stack || ''}`);
  } finally {
    running = false;
    $('start').disabled = false;
    $('cancel').disabled = true;
    $('openOut').disabled = false;
  }
}

function onBatchEvent(ev) {
  if (ev.type === 'start') {
    $('stat').textContent = `${ev.total}개 이미지 발견`;
    $('bar').style.width = '0%';
  } else if (ev.type === 'progress') {
    const pct = (ev.processed / Math.max(1, ev.total)) * 100;
    $('bar').style.width = `${pct}%`;
    $('stat').textContent = `${ev.processed} / ${ev.total}`;
    const name = baseName(ev.file);
    if (ev.status === 'censored') {
      const parts = ev.detections.map(
        (d) => `${meta.ko[d.label] || d.label} ${d.score.toFixed(2)}${d.model ? `/${d.model}` : ''}`
      );
      log('ok', `[검열] ${name} — ${parts.join(', ')}`);
    } else if (ev.status === 'skipped') {
      log('skip', `[건너뜀] ${name} (결과 파일 이미 존재)`);
    } else {
      log('clean', `[검출없음] ${name}`);
    }
  } else if (ev.type === 'error') {
    log('err', `[오류] ${baseName(ev.file)} — ${ev.message}`);
  } else if (ev.type === 'done') {
    $('bar').style.width = '100%';
    $('stat').textContent = `완료 — 검열 ${ev.censored}, 건너뜀 ${ev.skipped}, 실패 ${ev.failed}, 총 ${ev.processed}`;
    if (ev.outputDir && window.reviewSetDir) window.reviewSetDir(ev.outputDir);
  } else if (ev.type === 'cancelled') {
    $('stat').textContent = `중지됨 — ${ev.processed}개 처리`;
  } else if (ev.type === 'fatal') {
    log('err', `치명적 오류: ${ev.message}\n${ev.stack || ''}`);
  }
}

// Anything that escapes an event handler lands in the batch log rather than
// only in devtools, which the user has no reason to have open.
window.addEventListener('error', (ev) => {
  log('err', `오류: ${ev.message}\n${(ev.error && ev.error.stack) || ''}`);
});
window.addEventListener('unhandledrejection', (ev) => {
  const r = ev.reason;
  log('err', `처리되지 않은 오류: ${(r && r.message) || r}\n${(r && r.stack) || ''}`);
});

async function runPreview() {
  if (!previewFile) return;
  $('runPreview').disabled = true;
  $('previewInfo').textContent = '처리 중...';
  try {
    const r = await window.api.preview({
      file: previewFile,
      detectOptions: detectOptions(),
      censorOptions: currentCensorOptions(),
    });
    $('imgBefore').src = r.original;
    $('imgAfter').src = r.censored || r.original;
    $('previewInfo').textContent = r.detections.length
      ? `${r.width}x${r.height} — ` +
        r.detections
          .map((d) => `${meta.ko[d.label] || d.label} ${d.score.toFixed(2)}${d.model ? `/${d.model}` : ''}`)
          .join(', ')
      : `${r.width}x${r.height} — 검출 없음`;
  } catch (err) {
    $('previewInfo').textContent = `오류: ${err.message}`;
    log('err', `미리보기 실패: ${err.message}\n${err.stack || ''}`);
  } finally {
    $('runPreview').disabled = false;
  }
}

init();
