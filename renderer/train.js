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
  let timerInterval = null;
  let startTime = 0;

  let totalEpochs = 100;
  let currentEpoch = 0;
  let currentBatch = 0;
  let totalBatches = 0;
  let bestMaskMap = 0;
  let bestEpoch = 0;
  let stagnantEpochs = 0;

  function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0 || !isFinite(seconds)) return '--:--';
    const s = Math.floor(seconds % 60);
    const m = Math.floor((seconds / 60) % 60);
    const h = Math.floor(seconds / 3600);
    if (h > 0) {
      return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  function stripAnsi(str) {
    return str
      .replace(/\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, '')
      .replace(/\[K/g, '')
      .replace(/\r/g, '\n');
  }

  function log(text, cls) {
    const el = $('trainLog');
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    const clean = stripAnsi(text);
    const node = document.createElement('span');
    if (cls) node.className = cls;
    node.textContent = clean;
    el.appendChild(node);
    while (el.childNodes.length > 4000) el.removeChild(el.firstChild);
    if (atBottom) el.scrollTop = el.scrollHeight;
  }

  function updateDashboardMetrics(rawText) {
    const text = stripAnsi(rawText);
    const lines = text.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();

      // Early stopping check
      if (trimmed.includes('EarlyStopping') || trimmed.includes('Training stopped early')) {
        if ($('metricOverfit')) {
          $('metricOverfit').textContent = '🛑 과적합 방지 조기종료';
          $('metricOverfit').style.color = '#58a6ff';
        }
        if ($('dashBadge')) {
          $('dashBadge').className = 'dash-stage-badge done';
          $('dashBadge').textContent = `🛑 조기종료 (ep${bestEpoch || currentEpoch} 최적 모델)`;
        }
      }

      // 1. Loss line: "1/100 11.7G 2.416 3.942 5.63 0.007809 7.056 31 640: 90% 19/21 1.8s/it"
      const lossMatch = trimmed.match(
        /(\d+)\s*\/\s*(\d+)\s+([\d\.]+[GMK]?)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)/
      );
      if (lossMatch) {
        currentEpoch = parseInt(lossMatch[1], 10);
        totalEpochs = parseInt(lossMatch[2], 10) || totalEpochs;
        const gpuMem = lossMatch[3];
        const boxLoss = lossMatch[4];
        const segLoss = lossMatch[5];
        const clsLoss = lossMatch[6];

        if ($('metricGpuMem')) $('metricGpuMem').textContent = gpuMem;
        if ($('metricBoxLoss')) $('metricBoxLoss').textContent = parseFloat(boxLoss).toFixed(3);
        if ($('metricSegLoss')) $('metricSegLoss').textContent = parseFloat(segLoss).toFixed(3);
        if ($('metricClsLoss')) $('metricClsLoss').textContent = parseFloat(clsLoss).toFixed(3);
      }

      // 2. Batch & Speed line: "90% ━━━━━━━━━━╸─ 19/21 1.8s/it" or "19/21 1.8s/it"
      const batchMatch = trimmed.match(/(\d+)\s*\/\s*(\d+)\s+([\d\.]+(?:s\/it|it\/s))/);
      if (batchMatch) {
        currentBatch = parseInt(batchMatch[1], 10);
        totalBatches = parseInt(batchMatch[2], 10);
        const speed = batchMatch[3];

        if ($('dashBatchText')) $('dashBatchText').textContent = `${currentBatch} / ${totalBatches}`;
        if ($('dashBatchSpeed')) $('dashBatchSpeed').textContent = speed;

        const batchPct = totalBatches > 0 ? (currentBatch / totalBatches) * 100 : 0;
        if ($('dashBatchBar')) $('dashBatchBar').style.width = `${Math.min(100, Math.max(0, batchPct))}%`;
      }

      // 3. Validation results: "all 165 546 0.515 0.525 0.555 0.341 0.52 0.504 0.546 0.263"
      const valMatch = trimmed.match(
        /^all\s+\d+\s+\d+\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)/
      );
      if (valMatch) {
        const boxMap50 = parseFloat(valMatch[3]);
        const maskMap50 = parseFloat(valMatch[7]);
        if ($('metricBoxMap')) $('metricBoxMap').textContent = `${(boxMap50 * 100).toFixed(1)}%`;
        if ($('metricMaskMap')) $('metricMaskMap').textContent = `${(maskMap50 * 100).toFixed(1)}%`;

        if (maskMap50 > bestMaskMap) {
          bestMaskMap = maskMap50;
          bestEpoch = currentEpoch;
          stagnantEpochs = 0;
          if ($('metricBestMap')) {
            $('metricBestMap').textContent = `${(bestMaskMap * 100).toFixed(1)}% (ep${bestEpoch})`;
          }
          if ($('metricOverfit')) {
            $('metricOverfit').textContent = '✨ 최고점 갱신 (개선 중)';
            $('metricOverfit').style.color = '#7ee787';
          }
          if ($('dashBadge')) {
            $('dashBadge').className = 'dash-stage-badge';
            $('dashBadge').textContent = '✨ 성능 개선 중';
          }
        } else {
          stagnantEpochs++;
          if (stagnantEpochs >= 5) {
            if ($('metricOverfit')) {
              $('metricOverfit').textContent = `⚠️ 과적합 조짐 (${stagnantEpochs}ep 정체)`;
              $('metricOverfit').style.color = '#e3b341';
            }
            if ($('dashBadge')) {
              $('dashBadge').className = 'dash-stage-badge err';
              $('dashBadge').textContent = `⚠️ 성능 정체 (${stagnantEpochs}ep)`;
            }
          }
        }
      }
    }

    // Update Epoch Overall Progress
    if (totalEpochs > 0 && currentEpoch > 0) {
      const batchRatio = totalBatches > 0 ? currentBatch / totalBatches : 0;
      const overallPct = Math.min(100, ((currentEpoch - 1 + batchRatio) / totalEpochs) * 100);

      if ($('dashEpochText')) {
        $('dashEpochText').textContent = `Epoch ${currentEpoch} / ${totalEpochs} (${overallPct.toFixed(1)}%)`;
      }
      if ($('dashTotalPct')) {
        $('dashTotalPct').textContent = `${overallPct.toFixed(1)}%`;
      }
      if ($('dashTotalBar')) {
        $('dashTotalBar').style.width = `${overallPct}%`;
      }
    }
  }

  function startTimer() {
    stopTimer();
    startTime = Date.now();
    timerInterval = setInterval(() => {
      if (!running) {
        stopTimer();
        return;
      }
      const elapsedSec = (Date.now() - startTime) / 1000;
      if ($('dashElapsed')) $('dashElapsed').textContent = formatTime(elapsedSec);

      // Estimate remaining time
      if (totalEpochs > 0 && (currentEpoch > 0 || currentBatch > 0)) {
        const batchRatio = totalBatches > 0 ? currentBatch / totalBatches : 0;
        const progress = (Math.max(0, currentEpoch - 1) + batchRatio) / totalEpochs;

        if (progress > 0.005) {
          const totalSec = elapsedSec / progress;
          const remainingSec = Math.max(0, totalSec - elapsedSec);
          if ($('dashEta')) $('dashEta').textContent = `~${formatTime(remainingSec)}`;
        }
      }
    }, 1000);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
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
    const dir = $('trainDataset').value.trim();
    if (!dir) {
      $('trainDatasetStat').textContent = '';
      return;
    }
    try {
      const ready = await window.api.trainDatasetReady(dir);
      if (ready.ok) {
        const stats = await window.api.datasetStats(dir);
        const parts = Object.entries(stats.classes).map(([k, v]) => `${meta.ko[k] || k} ${v}`);
        $('trainDatasetStat').textContent = `${ready.samples}장 · 폴리곤 ${stats.polygons}${
          parts.length ? ` · ${parts.join(', ')}` : ''
        }`;
        if (ready.samples < 20) {
          $('trainDatasetStat').textContent += '  ⚠ 20장 미만은 과적합됩니다';
        }
      } else {
        $('trainDatasetStat').textContent = `사용 불가 — ${ready.reason}`;
      }
    } catch {
      $('trainDatasetStat').textContent = '폴더 확인 불가';
    }
  }

  function baseValue() {
    const sel = $('trainBase').value;
    return sel === '__local__' ? $('trainBasePath').value : sel;
  }

  async function start() {
    const datasetDir = $('trainDataset').value.trim();
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
    currentEpoch = 0;
    currentBatch = 0;
    totalEpochs = Number($('trainEpochs').value) || 100;
    totalBatches = 0;
    bestMaskMap = 0;
    bestEpoch = 0;
    stagnantEpochs = 0;

    $('trainStart').disabled = true;
    $('trainCancel').disabled = false;
    $('trainLog').innerHTML = '';

    // Show and reset dashboard
    $('trainDashboard').classList.remove('hidden');
    $('dashBadge').className = 'dash-stage-badge';
    $('dashBadge').textContent = '학습 진행 중';
    $('dashEpochText').textContent = `Epoch 0 / ${totalEpochs} (0%)`;
    $('dashTotalPct').textContent = '0%';
    $('dashTotalBar').style.width = '0%';
    $('dashBatchBar').style.width = '0%';
    $('dashBatchText').textContent = '0 / 0';
    $('dashBatchSpeed').textContent = '--';
    $('dashElapsed').textContent = '00:00';
    $('dashEta').textContent = '계산 중...';
    $('metricBoxLoss').textContent = '-';
    $('metricSegLoss').textContent = '-';
    $('metricClsLoss').textContent = '-';
    $('metricMaskMap').textContent = '-';
    $('metricBestMap').textContent = '-';
    if ($('metricOverfit')) {
      $('metricOverfit').textContent = '정상 (학습 중)';
      $('metricOverfit').style.color = '#7ee787';
    }
    $('metricGpuMem').textContent = '-';

    startTimer();

    // If preset base weight is not installed yet, auto-download it seamlessly!
    if (base !== '__local__') {
      const statuses = await window.api.modelsStatus();
      const found = statuses.find((s) => s.key === base);
      if (found && !found.installed) {
        $('dashBadge').textContent = '가중치 다운로드 중';
        $('trainStat').textContent = `베이스 가중치 자동 다운로드 중 (${found.formattedSize})...`;
        log(`[준비] 베이스 가중치(${found.label}) 자동 다운로드 중...\n`);
        try {
          await window.api.modelDownload({ key: base });
          log(`[준비] 베이스 가중치 다운로드 완료!\n`);
          $('dashBadge').textContent = '학습 진행 중';
        } catch (err) {
          running = false;
          stopTimer();
          $('trainStart').disabled = false;
          $('trainCancel').disabled = true;
          $('trainStat').textContent = `가중치 다운로드 실패: ${err.message}`;
          $('dashBadge').className = 'dash-stage-badge err';
          $('dashBadge').textContent = '다운로드 실패';
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
      epochs: totalEpochs,
      imgsz: Number($('trainImgsz').value),
      batch: Number($('trainBatch').value),
      patience: Number($('trainPatience').value),
      device: $('trainDevice').value,
      label: $('trainLabel').value.trim() || undefined,
    });

    running = false;
    stopTimer();
    $('trainStart').disabled = false;
    $('trainCancel').disabled = true;
    if (!r.ok && r.message) {
      $('trainStat').textContent = `실패 — ${r.message}`;
      $('dashBadge').className = 'dash-stage-badge err';
      $('dashBadge').textContent = '학습 실패';
    }
  }

  function onEvent(ev) {
    if (ev.type === 'log') {
      log(ev.text);
      updateDashboardMetrics(ev.text);
    } else if (ev.type === 'error') {
      log(`\n${ev.message}\n`, 'err');
      $('trainStat').textContent = `실패 — ${ev.message}`;
      $('dashBadge').className = 'dash-stage-badge err';
      $('dashBadge').textContent = '오류';
    } else if (ev.type === 'done') {
      log(`\n🎉 모델 등록 완료: ${ev.model.key}\n`, 'ok');
      $('trainStat').textContent = `완료 — ${ev.model.label}. 사이드바 모델 목록에 추가됨.`;
      $('dashBadge').className = 'dash-stage-badge done';
      $('dashBadge').textContent = '학습 완료';
      $('dashTotalBar').style.width = '100%';
      $('dashTotalPct').textContent = '100%';
      $('dashEpochText').textContent = `Epoch ${totalEpochs} / ${totalEpochs} (100%)`;
      if (window.refreshModels) window.refreshModels();
    }
  }

  window.initTrain = (m) => {
    meta = m;
    checkPython();

    $('trainDataset').onchange = refreshDataset;
    $('trainDataset').oninput = refreshDataset;

    $('pickTrainDataset').onclick = async () => {
      const d = await window.api.pickFolder('데이터셋 폴더 선택');
      if (d) {
        $('trainDataset').value = d;
        await refreshDataset();
      }
    };

    $('useDatasetDir').onclick = async () => {
      $('trainDataset').value = $('datasetDir').value;
      await refreshDataset();
    };

    $('btnToggleTrainLog').onclick = () => {
      const logEl = $('trainLog');
      logEl.classList.toggle('hidden');
      $('btnToggleTrainLog').textContent = logEl.classList.contains('hidden')
        ? '터미널 로그 보기'
        : '터미널 로그 숨기기';
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

