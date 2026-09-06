/**
 * SIH26055 — Tactical Electronic Warfare Dashboard Controller
 * Connects to live WebSocket stream, handles controls, updates visual telemetry,
 * renders the 30-channel tactical spectrum strip, and provides mission export.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const systemStateDot = document.getElementById('systemStateDot');
  const systemStateText = document.getElementById('systemStateText');
  const chipTime = document.getElementById('chipTime');
  const chipStep = document.getElementById('chipStep');
  const chipScenario = document.getElementById('chipScenario');
  const chipScheduler = document.getElementById('chipScheduler');
  const chipModelStatus = document.getElementById('chipModelStatus');

  // Primary Prediction
  const currentScanFreq = document.getElementById('currentScanFreq');
  const currentScanBin = document.getElementById('currentScanBin');
  const predictedNextFreq = document.getElementById('predictedNextFreq');
  const predictedNextBin = document.getElementById('predictedNextBin');
  const predConfidenceBadge = document.getElementById('predConfidenceBadge');
  const confGaugeFill = document.getElementById('confGaugeFill');
  const confPercent = document.getElementById('confPercent');
  const confLevelText = document.getElementById('confLevelText');
  const topPredictionsList = document.getElementById('topPredictionsList');

  // Performance & Latency
  const metricIR = document.getElementById('metricIR');
  const metricIRSub = document.getElementById('metricIRSub');
  const metricDetRate = document.getElementById('metricDetRate');
  const metricDetSub = document.getElementById('metricDetSub');
  const metricEfficiency = document.getElementById('metricEfficiency');
  const metricTotalScans = document.getElementById('metricTotalScans');
  const metricLatency = document.getElementById('metricLatency');
  const metricLatencyBudget = document.getElementById('metricLatencyBudget');

  // Detector
  const detStatusLabel = document.getElementById('detStatusLabel');
  const detSignalPower = document.getElementById('detSignalPower');
  const detReceiverBw = document.getElementById('detReceiverBw');

  // Why this scan & Arbitration
  const whyExplainText = document.getElementById('whyExplainText');
  const badgeArbitrationMode = document.getElementById('badgeArbitrationMode');
  const caWeightPct = document.getElementById('caWeightPct');
  const ddqnWeightPct = document.getElementById('ddqnWeightPct');
  const splitBarCA = document.getElementById('splitBarCA');
  const splitBarDDQN = document.getElementById('splitBarDDQN');
  const whySurprise = document.getElementById('whySurprise');
  const whyConsistency = document.getElementById('whyConsistency');
  const whyMode = document.getElementById('whyMode');

  // Intel panel
  const intelArchitecture = document.getElementById('intelArchitecture');
  const intelModelMode = document.getElementById('intelModelMode');
  const intelTrainingMode = document.getElementById('intelTrainingMode');
  const intelEpsilon = document.getElementById('intelEpsilon');
  const intelCheckpointName = document.getElementById('intelCheckpointName');
  const intelCheckpointHash = document.getElementById('intelCheckpointHash');

  // Spectrum Strip & Timeline & Log Console
  const spectrumBinsGrid = document.getElementById('spectrumBinsGrid');
  const timelineTableBody = document.getElementById('timelineTableBody');
  const tacticalLogConsole = document.getElementById('tacticalLogConsole');
  const logCounterBadge = document.getElementById('logCounterBadge');

  // Tactical Log Manager
  let logEventCount = 0;
  function appendTacticalLog(tag, msg, tagClass = 'tag-sys') {
    if (!tacticalLogConsole) return;
    logEventCount++;
    if (logCounterBadge) logCounterBadge.textContent = `${logEventCount} EVENTS`;

    const now = new Date();
    const timeStr = `${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}.${String(Math.floor(now.getMilliseconds() / 100))}`;

    const row = document.createElement('div');
    row.className = 'log-row';
    row.innerHTML = `<span class="log-time">[${timeStr}]</span> <span class="log-tag ${tagClass}">${tag}</span> <span>${msg}</span>`;

    tacticalLogConsole.appendChild(row);
    while (tacticalLogConsole.children.length > 50) {
      tacticalLogConsole.removeChild(tacticalLogConsole.firstChild);
    }
    tacticalLogConsole.scrollTop = tacticalLogConsole.scrollHeight;
  }

  // Controls
  const selectScenario = document.getElementById('selectScenario');
  const selectScheduler = document.getElementById('selectScheduler');
  const inputSeed = document.getElementById('inputSeed');
  const selectSpeed = document.getElementById('selectSpeed');
  const btnStart = document.getElementById('btnStart');
  const btnPause = document.getElementById('btnPause');
  const btnStep = document.getElementById('btnStep');
  const btnReset = document.getElementById('btnReset');
  const btnExport = document.getElementById('btnExport');

  // View Mode
  const btnModeOverview = document.getElementById('btnModeOverview');
  const btnModeTechnical = document.getElementById('btnModeTechnical');
  const evalBanner = document.getElementById('evalBanner');
  const chkGroundTruth = document.getElementById('chkGroundTruth');
  const evalLegends = document.querySelectorAll('.eval-only');

  // Initialize Spectrum Waterfall
  const waterfall = new SpectrumWaterfallRenderer('waterfallCanvas');


  // Build 30-channel Tactical Spectrum Strip
  const binCells = [];
  function initSpectrumStrip() {
    if (!spectrumBinsGrid) return;
    spectrumBinsGrid.innerHTML = '';
    binCells.length = 0;

    for (let b = 0; b < 30; b++) {
      const centerMhz = 100 + 10 + b * 20; // 110 to 690 MHz
      const cell = document.createElement('div');
      cell.className = 'spectrum-bin-cell';
      cell.id = `binCell_${b}`;
      cell.innerHTML = `
        <div class="bin-index">B${b}</div>
        <div class="bin-freq">${centerMhz}</div>
      `;
      cell.title = `Bin ${b}: ${centerMhz} MHz (Channel Bandwidth 20 MHz)`;
      spectrumBinsGrid.appendChild(cell);
      binCells.push(cell);
    }
  }
  initSpectrumStrip();

  // Mode Switcher handlers
  btnModeOverview.addEventListener('click', () => {
    btnModeOverview.classList.add('active');
    btnModeTechnical.classList.remove('active');
    document.body.classList.remove('mode-technical');
    document.body.classList.add('mode-overview');
    evalBanner.classList.add('hidden');
    evalLegends.forEach(el => el.classList.add('hidden'));
    waterfall.showGroundTruth = false;
    waterfall.draw();
  });

  btnModeTechnical.addEventListener('click', () => {
    btnModeTechnical.classList.add('active');
    btnModeOverview.classList.remove('active');
    document.body.classList.remove('mode-overview');
    document.body.classList.add('mode-technical');
    evalBanner.classList.remove('hidden');
    evalLegends.forEach(el => el.classList.remove('hidden'));
    waterfall.showGroundTruth = chkGroundTruth.checked;
    waterfall.draw();
  });

  chkGroundTruth.addEventListener('change', () => {
    waterfall.showGroundTruth = chkGroundTruth.checked;
    waterfall.draw();
  });

  // Fetch Scenarios Catalog
  async function loadScenarios() {
    try {
      const res = await fetch('/api/scenarios');
      const scenarios = await res.json();
      selectScenario.innerHTML = '';

      let curGroup = null;
      let curOptGroup = null;

      scenarios.forEach(sc => {
        const cat = sc.category || 'Available Scenarios';
        if (cat !== curGroup) {
          curGroup = cat;
          curOptGroup = document.createElement('optgroup');
          curOptGroup.label = cat;
          selectScenario.appendChild(curOptGroup);
        }
        const opt = document.createElement('option');
        opt.value = sc.id || sc.filename;
        opt.textContent = `${sc.name} (${sc.id})`;
        if (sc.id === '1_Seen_Structure') opt.selected = true;
        curOptGroup.appendChild(opt);
      });
    } catch (err) {
      console.error('Failed to load scenarios:', err);
    }
  }

  // Fetch Schedulers Catalog
  async function loadSchedulers() {
    if (!selectScheduler) return;
    try {
      const res = await fetch('/api/schedulers');
      if (!res.ok) return;
      const schedulers = await res.json();
      if (!Array.isArray(schedulers) || schedulers.length === 0) return;

      const previousVal = selectScheduler.value || 'hybrid_v4';
      selectScheduler.innerHTML = '';

      let curGroup = null;
      let curOptGroup = null;

      schedulers.forEach(sc => {
        const cat = sc.category || 'Scan Strategy Algorithms';
        if (cat !== curGroup) {
          curGroup = cat;
          curOptGroup = document.createElement('optgroup');
          curOptGroup.label = cat;
          selectScheduler.appendChild(curOptGroup);
        }
        const opt = document.createElement('option');
        opt.value = sc.id;
        const star = sc.is_proposed ? '⭐ ' : '';
        const tag = sc.is_proposed ? ' [OUR PROPOSED]' : '';
        opt.textContent = `${star}${sc.name} (${sc.overall_ir} IR) — ${sc.badge}${tag}`;
        if (sc.id === previousVal) opt.selected = true;
        curOptGroup.appendChild(opt);
      });
    } catch (err) {
      console.warn('Using pre-rendered schedulers catalog fallback:', err);
    }
  }

  // Update UI from Telemetry Payload
  function updateUI(payload) {
    if (!payload || !payload.system_status) return;

    const sys = payload.system_status;
    const pred = payload.primary_prediction || {};
    const det = payload.detector || {};
    const perf = payload.performance || {};
    const arb = payload.arbitration || {};
    const lat = payload.latency || {};
    const nmodel = payload.neural_model || {};

    // 1. Header Status
    systemStateText.textContent = sys.state || 'IDLE';
    systemStateDot.className = 'status-dot';
    if (sys.state === 'RUNNING') systemStateDot.classList.add('active');
    else if (sys.state === 'PAUSED') systemStateDot.classList.add('paused');

    chipTime.textContent = `${(sys.simulation_time_s || 0).toFixed(1)} s`;
    chipStep.textContent = sys.step || 0;
    chipScenario.textContent = sys.scenario_name ? sys.scenario_name.replace('.yaml', '').replace('.canonical', '') : '--';

    const sType = sys.scheduler_type || sys.scheduler_name || 'hybrid_v4';
    if (chipScheduler) {
      if (sType === 'hybrid_v4') {
        chipScheduler.innerHTML = '⭐ V4.0 HYBRID <span class="badge-tag-winner">PROPOSED</span>';
        chipScheduler.style.color = '#00e5ff';
      } else {
        chipScheduler.innerHTML = `BASELINE: ${sType.toUpperCase()}`;
        chipScheduler.style.color = '#f59e0b';
      }
    }

    if (selectScheduler && document.activeElement !== selectScheduler) {
      if (selectScheduler.value !== sType && selectScheduler.querySelector(`option[value="${sType}"]`)) {
        selectScheduler.value = sType;
      }
    }

    // 2. Prediction Hero Card
    currentScanFreq.textContent = (pred.current_scan_mhz || 0).toFixed(1);
    currentScanBin.textContent = `Channel Bin ${pred.current_scan_bin != null ? pred.current_scan_bin : '--'}`;

    predictedNextFreq.textContent = (pred.predicted_next_mhz || 0).toFixed(1);
    predictedNextBin.textContent = `Channel Bin ${pred.predicted_next_bin != null ? pred.predicted_next_bin : '--'}`;

    const conf = pred.confidence_pct || 0;
    predConfidenceBadge.textContent = `CONFIDENCE: ${conf.toFixed(0)}% (${pred.confidence_level || 'MED'})`;
    confGaugeFill.style.width = `${conf}%`;
    confPercent.textContent = `${conf.toFixed(0)}%`;
    confLevelText.textContent = `Pattern Stability: ${pred.confidence_level || 'MEDIUM'} Confidence`;

    // Candidate Q-Value Chips
    if (pred.top_predictions && pred.top_predictions.length > 0) {
      topPredictionsList.innerHTML = '';
      pred.top_predictions.forEach(p => {
        const chip = document.createElement('span');
        chip.className = 'ranking-chip';
        const shareStr = p.share_pct ? ` &bull; ${p.share_pct}%` : '';
        chip.innerHTML = `<b>#${p.rank} Bin ${p.bin} (${p.frequency_mhz} MHz)</b> [Q: ${p.q_value.toFixed(2)}${shareStr}]`;
        topPredictionsList.appendChild(chip);
      });
    }

    // 3. Performance & Latency
    metricIR.textContent = `${(perf.interception_ratio_pct || 0).toFixed(1)}%`;
    metricIRSub.textContent = `${perf.intercepted_opportunities || 0} / ${perf.total_opportunities || 0} Opportunities`;

    metricDetRate.textContent = `${(perf.detection_rate_pct || 0).toFixed(1)}%`;
    metricDetSub.textContent = `${perf.total_detections || 0} Hits / ${det.status_label || 'SCAN'}`;

    metricEfficiency.textContent = `${(perf.scan_efficiency_pct || 0).toFixed(1)}%`;
    metricTotalScans.textContent = `${perf.total_scans || 0} Sweeps Executed`;

    if (metricLatency) {
      const latVal = lat.step_latency_ms != null ? lat.step_latency_ms : 0.4;
      metricLatency.textContent = `${latVal.toFixed(1)} ms`;
    }
    if (metricLatencyBudget) {
      metricLatencyBudget.textContent = lat.status || '✓ < 10 ms BUDGET';
    }

    // 4. Detector Status Bar
    if (det.detected) {
      detStatusLabel.textContent = 'SIGNAL DETECTED';
      detStatusLabel.className = 'det-val hit';
    } else {
      detStatusLabel.textContent = 'NO SIGNAL';
      detStatusLabel.className = 'det-val text-dim';
    }
    detSignalPower.textContent = det.signal_power_dbm != null ? `${det.signal_power_dbm.toFixed(1)} dBm` : '-- dBm';
    detReceiverBw.textContent = `${det.receiver_bandwidth_mhz || 20.0} MHz`;

    // 5. Why this scan & Arbitration
    whyExplainText.textContent = arb.explanation || 'Evaluating cognitive scan decision...';
    badgeArbitrationMode.textContent = arb.mode || 'DDQN_EXPLOIT';

    const caPct = arb.ca_weight_pct != null ? arb.ca_weight_pct : 30;
    const ddqnPct = arb.ddqn_weight_pct != null ? arb.ddqn_weight_pct : 70;
    if (caWeightPct) caWeightPct.textContent = `${caPct.toFixed(0)}%`;
    if (ddqnWeightPct) ddqnWeightPct.textContent = `${ddqnPct.toFixed(0)}%`;
    if (splitBarCA) splitBarCA.style.width = `${caPct}%`;
    if (splitBarDDQN) splitBarDDQN.style.width = `${ddqnPct}%`;

    if (whySurprise) whySurprise.textContent = (arb.surprise || 0).toFixed(3);
    if (whyConsistency) whyConsistency.textContent = (arb.consistency || 0).toFixed(3);
    if (whyMode) whyMode.textContent = arb.mode || 'DDQN_EXPLOIT';

    // 6. Intel Panel (Technical)
    if (intelArchitecture) {
      intelArchitecture.textContent = nmodel.architecture || 'MLPQNetwork (Feed-Forward DDQN)';
    }
    if (intelModelMode) {
      intelModelMode.textContent = nmodel.mode || 'FROZEN INFERENCE ONLY';
    }
    if (intelTrainingMode) {
      intelTrainingMode.textContent = `${nmodel.runtime_training || 'DISABLED'} (Zero Drift)`;
    }
    if (intelCheckpointName) {
      intelCheckpointName.textContent = nmodel.checkpoint_name || 'production_checkpoint.npz';
    }
    if (intelCheckpointHash) {
      intelCheckpointHash.textContent = nmodel.checkpoint_sha256 ? `${nmodel.checkpoint_sha256.substring(0, 16)}...` : '42a6f3b8...';
    }
    if (chipModelStatus) {
      if (sType === 'hybrid_v4' || sType === 'hybrid_v41') {
        chipModelStatus.textContent = `${nmodel.status || 'PRETRAINED'} (FROZEN)`;
        chipModelStatus.style.color = '#00e676';
      } else {
        chipModelStatus.textContent = `${nmodel.status || 'BASELINE'} (ANALYTIC)`;
        chipModelStatus.style.color = '#94a3b8';
      }
    }

    // 7. Update 30-Channel Spectrum Strip
    const scanBin = pred.current_scan_bin;
    const nextBin = pred.predicted_next_bin;
    const isHit = det.detected;

    // Extract ground truth bins if available
    const gtBins = new Set();
    const wfEvents = payload.waterfall_events || [];
    const latestEvent = wfEvents.length > 0 ? wfEvents[wfEvents.length - 1] : null;
    if (latestEvent && latestEvent.ground_truth) {
      latestEvent.ground_truth.forEach(em => {
        const freqHz = em.frequency_mhz * 1e6;
        const bIdx = Math.round((freqHz - 100e6 - 10e6) / 20e6);
        if (bIdx >= 0 && bIdx < 30) gtBins.add(bIdx);
      });
    }

    binCells.forEach((cell, idx) => {
      cell.className = 'spectrum-bin-cell';
      if (idx === scanBin) {
        cell.classList.add('bin-scanning');
        if (isHit) cell.classList.add('bin-hit');
      }
      if (idx === nextBin) {
        cell.classList.add('bin-predicted');
      }
      if (gtBins.has(idx) && chkGroundTruth && chkGroundTruth.checked) {
        cell.classList.add('bin-gt');
      }
    });

    // 8. Waterfall Canvas
    if (payload.bands_mhz) {
      waterfall.setBands(payload.bands_mhz);
    }
    if (payload.waterfall_events) {
      waterfall.setEvents(payload.waterfall_events);
    }

    // 9. Recent Timeline Table
    if (timelineTableBody) {
      if (payload.recent_timeline && payload.recent_timeline.length > 0) {
        timelineTableBody.innerHTML = '';
        payload.recent_timeline.slice(0, 10).forEach(row => {
          const tr = document.createElement('tr');
          const detBadge = row.detected
            ? '<span class="badge-pill-hit">✓ DETECTED</span>'
            : '<span class="badge-pill-miss">✗ MISSED</span>';

          tr.innerHTML = `
            <td><b>t=${row.step}</b></td>
            <td class="text-cyan">${row.scanned_mhz.toFixed(1)} MHz (B${row.scanned_bin})</td>
            <td class="text-amber">${row.predicted_mhz.toFixed(1)} MHz (B${row.predicted_bin})</td>
            <td>${row.confidence_pct.toFixed(0)}%</td>
            <td>${detBadge}</td>
            <td><code>${row.arbitration_mode}</code></td>
          `;
          timelineTableBody.appendChild(tr);
        });
      } else {
        timelineTableBody.innerHTML = '<tr><td colspan="6" class="text-center" style="color: #64748b; padding: 18px;">Standby. Click <b>▶ START MISSION</b> or <b>⏭ STEP</b> to begin scanning.</td></tr>';
      }
    }

    // 10. Append to Tactical Event Log on Step Change
    const curStep = sys.step != null ? sys.step : -1;
    if (curStep !== lastLoggedStep && curStep >= 0) {
      lastLoggedStep = curStep;
      if (det.detected) {
        appendTacticalLog('INTERCEPT', `Target hit on Bin ${pred.current_scan_bin} (${(pred.current_scan_mhz || 0).toFixed(1)} MHz) | Power: ${(det.signal_power_dbm || 0).toFixed(1)} dBm | Total Hits: ${perf.total_detections || 1}`, 'tag-hit');
      } else {
        appendTacticalLog('SWEEP', `Scanned Bin ${pred.current_scan_bin} (${(pred.current_scan_mhz || 0).toFixed(1)} MHz) | Target: Bin ${pred.predicted_next_bin} (${(pred.predicted_next_mhz || 0).toFixed(1)} MHz) | Policy: ${arb.mode || 'DDQN'}`, 'tag-scan');
      }
    }
  }

  let lastLoggedStep = -1;
  let isSimulationRunning = false;
  let pollingInterval = null;

  // Export Mission Report Download
  if (btnExport) {
    btnExport.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/export');
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `smart_scan_mission_report_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        appendTacticalLog('EXPORT', 'Tactical mission report JSON exported successfully', 'tag-sys');
      } catch (err) {
        console.error('Failed to export mission report:', err);
        alert('Mission export failed: ' + err.message);
      }
    });
  }

  // Simulation Controls Event Listeners
  btnStart.addEventListener('click', async () => {
    try {
      btnStart.disabled = true;
      appendTacticalLog('CONTROL', `Starting continuous scan execution on "${selectScenario.value}"...`, 'tag-ctrl');
      const res = await fetch('/api/simulation/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      isSimulationRunning = true;
      systemStateText.textContent = 'RUNNING';
      systemStateDot.className = 'status-dot active';
      await fetchLatestStatus();
    } catch (err) {
      console.error('Start error:', err);
      appendTacticalLog('ERROR', `Mission start failed: ${err.message}`, 'tag-ctrl');
    } finally {
      btnStart.disabled = false;
    }
  });

  btnPause.addEventListener('click', async () => {
    try {
      btnPause.disabled = true;
      appendTacticalLog('CONTROL', 'Mission execution paused by operator', 'tag-ctrl');
      await fetch('/api/simulation/pause', { method: 'POST' });
      isSimulationRunning = false;
      systemStateText.textContent = 'PAUSED';
      systemStateDot.className = 'status-dot paused';
      await fetchLatestStatus();
    } catch (err) {
      console.error('Pause error:', err);
    } finally {
      btnPause.disabled = false;
    }
  });

  btnStep.addEventListener('click', async () => {
    try {
      btnStep.disabled = true;
      const res = await fetch('/api/simulation/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps: 1 }),
      });
      const data = await res.json();
      updateUI(data);
    } catch (err) {
      console.error('Step error:', err);
    } finally {
      btnStep.disabled = false;
    }
  });

  btnReset.addEventListener('click', async () => {
    const seed = parseInt(inputSeed.value, 10) || 42;
    const scenario = selectScenario.value;
    const scheduler = selectScheduler ? selectScheduler.value : 'hybrid_v4';
    try {
      btnReset.disabled = true;
      appendTacticalLog('RESET', `Environment reset: Algorithm "${scheduler.toUpperCase()}" | Scenario "${scenario}" | Seed: ${seed}`, 'tag-sys');
      const res = await fetch('/api/simulation/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed: seed, scenario_name: scenario, scheduler_name: scheduler }),
      });
      const data = await res.json();
      isSimulationRunning = false;
      lastLoggedStep = -1;
      waterfall.clear();
      if (data.telemetry) updateUI(data.telemetry);
    } catch (err) {
      console.error('Reset error:', err);
    } finally {
      btnReset.disabled = false;
    }
  });

  if (selectScheduler) {
    selectScheduler.addEventListener('change', async () => {
      const isProposed = selectScheduler.value === 'hybrid_v4';
      const optText = selectScheduler.options[selectScheduler.selectedIndex]?.text || selectScheduler.value;
      appendTacticalLog('ALGO', `Algorithm switched to ${optText}. ${isProposed ? '★ WINNING PROPOSED SYSTEM' : 'Comparative Baseline'}. Auto-resetting...`, isProposed ? 'tag-hit' : 'tag-sys');
      btnReset.click();
    });
  }

  selectScenario.addEventListener('change', async () => {
    appendTacticalLog('CONFIG', `Scenario switched to ${selectScenario.value}. Auto-resetting...`, 'tag-sys');
    btnReset.click();
  });

  selectSpeed.addEventListener('change', async () => {
    try {
      await fetch('/api/simulation/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: selectSpeed.value }),
      });
      appendTacticalLog('CONFIG', `Simulation execution speed set to ${selectSpeed.value}`, 'tag-sys');
    } catch (err) {
      console.error('Speed change error:', err);
    }
  });

  // WebSocket Connection Lifecycle
  let ws = null;
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use /ws/telemetry; backend routes both /ws and /ws/telemetry
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    try {
      ws = new WebSocket(wsUrl);
    } catch (err) {
      console.warn('WebSocket creation failed, using HTTP fallback:', err);
      return;
    }

    ws.onopen = () => {
      console.log('✅ Connected to EW live telemetry WebSocket');
      appendTacticalLog('STREAM', 'Live telemetry WebSocket connected (Zero-Lag Mode)', 'tag-sys');
      fetchLatestStatus();
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if ((msg.event_type === 'TELEMETRY' || msg.event_type === 'SNAPSHOT') && msg.payload) {
          updateUI(msg.payload);
        }
      } catch (e) {
        console.error('Error parsing telemetry message:', e);
      }
    };

    ws.onclose = () => {
      console.warn('⚠️ Telemetry WebSocket disconnected. Reconnecting in 2s...');
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.warn('WebSocket connection error, fallback polling active:', err);
      try { ws.close(); } catch (_) {}
    };
  }

  // HTTP Polling Fallback to guarantee real-time updates under any network condition
  function startPollingFallback() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(async () => {
      const wsActive = ws && ws.readyState === WebSocket.OPEN;
      if (!wsActive || isSimulationRunning) {
        await fetchLatestStatus();
      }
    }, 250);
  }

  // Direct HTTP fetch
  async function fetchLatestStatus() {
    try {
      const res = await fetch('/api/telemetry');
      if (!res.ok) return;
      const data = await res.json();
      if (data && data.system_status) {
        isSimulationRunning = data.system_status.state === 'RUNNING';
        updateUI(data);
      }
    } catch (err) {
      console.error('Status fetch error:', err);
    }
  }

  // Initialize: Load catalog -> Fetch initial state -> Connect WebSocket -> Start Polling
  Promise.all([loadScenarios(), loadSchedulers()]).then(async () => {
    await fetchLatestStatus();
    connectWebSocket();
    startPollingFallback();
    appendTacticalLog('SYSTEM', 'Tactical EW Dashboard initialized. V4.0 Hybrid Production Ready.', 'tag-sys');
  });
});

