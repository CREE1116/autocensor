'use strict';

/**
 * Fine-tuning tab. Training itself runs in the user's Python (ultralytics pulls
 * in torch, far too big to ship), so the tab's job is to check that Python is
 * usable, hand over the config, stream the output, and pick up the model that
 * comes out the other end.
 */
(() => {
  const $ = (id) => document.getElementById(id);
  let meta = null;
  let env = null;
  let running = false;

  function log(text, cls) {
    const el = $('trainLog');
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    const node = document.createElement('span');
    if (cls) node.className = cls;
    node.textContent = text;
    el.appendChild(node);
    // Ultralytics is chatty; keep the panel from growing without bound.
    while (el.childNodes.length > 4000) el.removeChild(el.firstChild);
    if (atBottom) el.scrollTop = el.scrollHeight;
  }

  async function checkPython() {
    env = await window.api.trainCheck([]);
    const box = $('pyStatus');
    if (env.python) {
      box.className = 'notice ok';
      box.innerHTML = `Python 준비됨 — <code>${env.python.bin}</code> · ultralytics ${env.python.version} · torch ${env.python.torch}`;
      $('trainDevice').value = env.device;
    } else {
      box.className = 'notice bad';
      box.innerHTML =
        'ultralytics를 찾지 못했습니다. 터미널에서 <code>pip install ultralytics</code> 후 앱을 다시 여세요. ' +
        '특정 파이썬을 쓰려면 <code>AUTOCENSOR_PYTHON=/경로/python3</code> 환경변수를 지정하세요.';
      $('trainStart').disabled = true;
    }
  }

  async function refreshDataset() {
    const dir = $('trainDataset').value;
    if (!dir) {
      $('trainDatasetStat').textContent = '';
      return;
    }
    const ready = await window.api.trainDatasetReady(dir);
    if (ready.ok) {
      const stats = await window.api.datasetStats(dir);
      const parts = Object.entries(stats.classes).map(([k, v]) => `${meta.ko[k] || k} ${v}`);
      $('trainDatasetStat').textContent = `${ready.samples}장 · 폴리곤 ${stats.polygons}${
        parts.length ? ` · ${parts.join(', ')}` : ''
      }`;
      // A handful of images will overfit instantly; say so rather than let the
      // run look successful.
      if (ready.samples < 20) {
        $('trainDatasetStat').textContent += '  ⚠ 20장 미만은 과적합됩니다';
      }
    } else {
      $('trainDatasetStat').textContent = `사용 불가 — ${ready.reason}`;
    }
  }

  function baseValue() {
    const sel = $('trainBase').value;
    return sel === '__local__' ? $('trainBasePath').value : sel;
  }

  async function start() {
    const datasetDir = $('trainDataset').value;
    if (!datasetDir) {
      alert('데이터셋 폴더를 지정하세요.');
      return;
    }
    const base = baseValue();
    if (!base) {
      alert('베이스 가중치를 선택하세요.');
      return;
    }

    running = true;
    $('trainStart').disabled = true;
    $('trainCancel').disabled = false;
    $('trainLog').innerHTML = '';

    // If preset base weight is not installed yet, auto-download it seamlessly!
    if (base !== '__local__') {
      const statuses = await window.api.modelsStatus();
      const found = statuses.find((s) => s.key === base);
      if (found && !found.installed) {
        $('trainStat').textContent = `베이스 가중치 자동 다운로드 중 (${found.formattedSize})...`;
        log(`[준비] 베이스 가중치(${found.label}) 자동 다운로드 중...\n`);
        try {
          await window.api.modelDownload({ key: base });
          log(`[준비] 베이스 가중치 다운로드 완료!\n`);
        } catch (err) {
          running = false;
          $('trainStart').disabled = false;
          $('trainCancel').disabled = true;
          $('trainStat').textContent = `가중치 다운로드 실패: ${err.message}`;
          alert(`베이스 가중치 자동 다운로드 실패: ${err.message}`);
          return;
        }
      }
    }

    $('trainStat').textContent = '학습 중...';

    const r = await window.api.trainStart({
      python: env.python.bin,
      datasetDir,
      base,
      epochs: Number($('trainEpochs').value),
      imgsz: Number($('trainImgsz').value),
      batch: Number($('trainBatch').value),
      device: $('trainDevice').value,
      label: $('trainLabel').value.trim() || undefined,
    });

    running = false;
    $('trainStart').disabled = false;
    $('trainCancel').disabled = true;
    if (!r.ok && r.message) $('trainStat').textContent = `실패 — ${r.message}`;
  }

  function onEvent(ev) {
    if (ev.type === 'log') {
      log(ev.text);
    } else if (ev.type === 'error') {
      log(`\n${ev.message}\n`, 'err');
      $('trainStat').textContent = `실패 — ${ev.message}`;
    } else if (ev.type === 'done') {
      log(`\n모델 등록됨: ${ev.model.key}\n`, 'ok');
      $('trainStat').textContent = `완료 — ${ev.model.label}. 사이드바 모델 목록에 추가됨.`;
      if (window.refreshModels) window.refreshModels();
    }
  }

  window.initTrain = (m) => {
    meta = m;
    checkPython();

    $('useDatasetDir').onclick = () => {
      $('trainDataset').value = $('datasetDir').value;
      refreshDataset();
    };
    async function checkBaseStatus() {
      const sel = $('trainBase').value;
      const dlBtn = $('trainDlBase');
      if (sel === '__local__') {
        dlBtn.style.display = 'none';
        $('trainBasePath').style.display = '';
        return;
      }
      $('trainBasePath').style.display = 'none';
      $('trainBasePath').value = '';

      const statuses = await window.api.modelsStatus();
      const found = statuses.find((s) => s.key === sel);
      if (found && !found.installed) {
        dlBtn.style.display = '';
        dlBtn.textContent = `가중치 다운로드 (${found.formattedSize})`;
        dlBtn.onclick = async () => {
          dlBtn.disabled = true;
          dlBtn.textContent = '다운로드 중...';
          try {
            await window.api.modelDownload({ key: sel });
            dlBtn.style.display = 'none';
          } catch (err) {
            alert(`다운로드 실패: ${err.message}`);
            dlBtn.textContent = `가중치 다운로드 (${found.formattedSize})`;
          } finally {
            dlBtn.disabled = false;
          }
        };
      } else {
        dlBtn.style.display = 'none';
      }
    }

    $('trainBase').onchange = async () => {
      if ($('trainBase').value === '__local__') {
        $('trainBasePath').style.display = '';
        const f = await window.api.pickWeights();
        if (f) {
          $('trainBasePath').value = f;
        } else if (!$('trainBasePath').value) {
          $('trainBase').value = 'anime_medium';
          $('trainBasePath').style.display = 'none';
        }
      }
      await checkBaseStatus();
    };
    checkBaseStatus();
    $('trainStart').onclick = start;
    $('trainCancel').onclick = () => window.api.trainCancel();

    window.api.onTrainEvent(onEvent);

    // Keep the tab in step with the dataset picked in the sidebar.
    const pick = $('pickDataset');
    const prev = pick.onclick;
    pick.onclick = async (ev) => {
      if (prev) await prev(ev);
      if ($('datasetDir').value && !$('trainDataset').value) {
        $('trainDataset').value = $('datasetDir').value;
        refreshDataset();
      }
    };
  };

  window.trainIsRunning = () => running;
})();
