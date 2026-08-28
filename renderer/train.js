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
  let trainHistory = [];

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

  function setupCanvas(canvas) {
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || 320;
    const h = rect.height || 150;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.resetTransform?.();
    ctx.scale(dpr, dpr);
    return { ctx, width: w, height: h };
  }

  function drawLossChart() {
    const canvas = $('chartLoss');
    const info = setupCanvas(canvas);
    if (!info) return;
    const { ctx, width, height } = info;

    ctx.clearRect(0, 0, width, height);

    const padLeft = 36;
    const padRight = 14;
    const padTop = 15;
    const padBottom = 22;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Grid & axes
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    let maxLoss = 4.0;
    for (const h of trainHistory) {
      if (h.boxLoss !== undefined) maxLoss = Math.max(maxLoss, h.boxLoss);
      if (h.segLoss !== undefined) maxLoss = Math.max(maxLoss, h.segLoss);
      if (h.clsLoss !== undefined) maxLoss = Math.max(maxLoss, h.clsLoss);
    }
    maxLoss = Math.ceil(maxLoss * 1.1 * 10) / 10;

    for (let i = 0; i <= 3; i++) {
      const yVal = (maxLoss * (3 - i)) / 3;
      const yPos = padTop + (plotH * i) / 3;
      ctx.beginPath();
      ctx.moveTo(padLeft, yPos);
      ctx.lineTo(width - padRight, yPos);
      ctx.stroke();
      ctx.fillText(yVal.toFixed(1), padLeft - 6, yPos);
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const numTicks = Math.min(5, Math.max(1, totalEpochs));
    for (let i = 0; i <= numTicks; i++) {
      const ep = Math.round((totalEpochs * i) / numTicks);
      if (ep === 0 && numTicks > 1) continue;
      const xPos = padLeft + (plotW * (ep || 1)) / Math.max(1, totalEpochs);
      ctx.fillText(`ep${ep || 1}`, xPos, height - padBottom + 6);
    }

    if (trainHistory.length === 0) {
      ctx.fillStyle = '#484f58';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('학습 데이터 대기 중...', padLeft + plotW / 2, padTop + plotH / 2);
      return;
    }

    const toX = (ep) => padLeft + (plotW * (ep - 1)) / Math.max(1, totalEpochs - 1);
    const toY = (val) => padTop + plotH - (plotH * Math.min(maxLoss, Math.max(0, val))) / maxLoss;

    function plotLine(key, color) {
      const valid = trainHistory.filter((h) => h[key] !== undefined);
      if (valid.length === 0) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.lineJoin = 'round';
      valid.forEach((pt, idx) => {
        const x = toX(pt.epoch);
        const y = toY(pt[key]);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      const last = valid[valid.length - 1];
      const lx = toX(last.epoch);
      const ly = toY(last[key]);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(lx, ly, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    plotLine('boxLoss', '#f85149');
    plotLine('segLoss', '#d29922');
    plotLine('clsLoss', '#a371f7');
  }

  function drawMapChart() {
    const canvas = $('chartMap');
    const info = setupCanvas(canvas);
    if (!info) return;
    const { ctx, width, height } = info;

    ctx.clearRect(0, 0, width, height);

    const padLeft = 36;
    const padRight = 14;
    const padTop = 15;
    const padBottom = 22;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i <= 4; i++) {
      const pct = (4 - i) * 25;
      const yPos = padTop + (plotH * i) / 4;
      ctx.beginPath();
      ctx.moveTo(padLeft, yPos);
      ctx.lineTo(width - padRight, yPos);
      ctx.stroke();
      ctx.fillText(`${pct}%`, padLeft - 6, yPos);
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const numTicks = Math.min(5, Math.max(1, totalEpochs));
    for (let i = 0; i <= numTicks; i++) {
      const ep = Math.round((totalEpochs * i) / numTicks);
      if (ep === 0 && numTicks > 1) continue;
      const xPos = padLeft + (plotW * (ep || 1)) / Math.max(1, totalEpochs);
      ctx.fillText(`ep${ep || 1}`, xPos, height - padBottom + 6);
    }

    const valPoints = trainHistory.filter((h) => h.maskMap !== undefined || h.boxMap !== undefined);
    if (valPoints.length === 0) {
      ctx.fillStyle = '#484f58';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('에폭 검증 결과 대기 중...', padLeft + plotW / 2, padTop + plotH / 2);
      return;
    }

    const toX = (ep) => padLeft + (plotW * (ep - 1)) / Math.max(1, totalEpochs - 1);
    const toY = (val) => padTop + plotH - plotH * Math.min(1.0, Math.max(0, val));

    const maskPts = trainHistory.filter((h) => h.maskMap !== undefined);
    if (maskPts.length > 0) {
      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
      grad.addColorStop(0, 'rgba(63, 185, 80, 0.25)');
      grad.addColorStop(1, 'rgba(63, 185, 80, 0.0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(toX(maskPts[0].epoch), padTop + plotH);
      maskPts.forEach((pt) => ctx.lineTo(toX(pt.epoch), toY(pt.maskMap)));
      ctx.lineTo(toX(maskPts[maskPts.length - 1].epoch), padTop + plotH);
      ctx.closePath();
      ctx.fill();
    }

    function plotLine(key, color, isMain) {
      const valid = trainHistory.filter((h) => h[key] !== undefined);
      if (valid.length === 0) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = isMain ? 2.2 : 1.5;
      ctx.lineJoin = 'round';
      valid.forEach((pt, idx) => {
        const x = toX(pt.epoch);
        const y = toY(pt[key]);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      valid.forEach((pt) => {
        const x = toX(pt.epoch);
        const y = toY(pt[key]);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    plotLine('boxMap', '#58a6ff', false);
    plotLine('maskMap', '#3fb950', true);

    const bestPoint = maskPts.find((h) => h.isBest);
    if (bestPoint) {
      const bx = toX(bestPoint.epoch);
      const by = toY(bestPoint.maskMap);

      ctx.strokeStyle = '#f0883e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(bx, by, 6, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#f0883e';
      ctx.beginPath();
      ctx.arc(bx, by, 3.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#f0883e';
      ctx.font = 'bold 10px -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`★ ${(bestPoint.maskMap * 100).toFixed(1)}%`, bx, by - 8);
    }
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
        const boxLoss = parseFloat(lossMatch[4]);
        const segLoss = parseFloat(lossMatch[5]);
        const clsLoss = parseFloat(lossMatch[6]);

        if ($('metricGpuMem')) $('metricGpuMem').textContent = gpuMem;
        if ($('metricBoxLoss')) $('metricBoxLoss').textContent = boxLoss.toFixed(3);
        if ($('metricSegLoss')) $('metricSegLoss').textContent = segLoss.toFixed(3);
        if ($('metricClsLoss')) $('metricClsLoss').textContent = clsLoss.toFixed(3);

        let item = trainHistory.find((h) => h.epoch === currentEpoch);
        if (!item) {
          item = { epoch: currentEpoch };
          trainHistory.push(item);
        }
        item.boxLoss = boxLoss;
        item.segLoss = segLoss;
        item.clsLoss = clsLoss;
        drawLossChart();
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

        let item = trainHistory.find((h) => h.epoch === currentEpoch);
        if (!item) {
          item = { epoch: currentEpoch };
          trainHistory.push(item);
        }
        item.boxMap = boxMap50;
        item.maskMap = maskMap50;

        if (maskMap50 > bestMaskMap) {
          bestMaskMap = maskMap50;
          bestEpoch = currentEpoch;
          stagnantEpochs = 0;
          trainHistory.forEach((h) => (h.isBest = h.epoch === bestEpoch));
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
        drawMapChart();
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
    trainHistory = [];

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

    drawLossChart();
    drawMapChart();
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

    drawLossChart();
    drawMapChart();

    window.addEventListener('resize', () => {
      drawLossChart();
      drawMapChart();
    });

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

  window.drawTrainCharts = () => {
    drawLossChart();
    drawMapChart();
  };
  window.trainIsRunning = () => running;
})();

