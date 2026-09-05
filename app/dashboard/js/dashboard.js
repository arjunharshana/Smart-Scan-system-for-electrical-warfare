/**
 * SIH26055 — Tactical Electronic Warfare Dashboard Controller
 * Connects to live WebSocket stream, handles controls, updates visual telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const systemStateDot = document.getElementById('systemStateDot');
  const systemStateText = document.getElementById('systemStateText');
  const chipTime = document.getElementById('chipTime');
  const chipStep = document.getElementById('chipStep');
  const chipScenario = document.getElementById('chipScenario');

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

  // Performance
  const metricIR = document.getElementById('metricIR');
  const metricIRSub = document.getElementById('metricIRSub');
  const metricDetRate = document.getElementById('metricDetRate');
  const metricDetSub = document.getElementById('metricDetSub');
  const metricEfficiency = document.getElementById('metricEfficiency');
  const metricTotalScans = document.getElementById('metricTotalScans');

  // Detector
  const detStatusLabel = document.getElementById('detStatusLabel');
  const detSignalPower = document.getElementById('detSignalPower');
  const detReceiverBw = document.getElementById('detReceiverBw');

  // Why this scan & Arbitration
  const whyExplainText = document.getElementById('whyExplainText');
  const badgeArbitrationMode = document.getElementById('badgeArbitrationMode');
  const caWeightPct = document.getElementById('caWeightPct');
  const lstmWeightPct = document.getElementById('lstmWeightPct');
  const splitBarCA = document.getElementById('splitBarCA');
  const splitBarLSTM = document.getElementById('splitBarLSTM');

  // Intel panel
  const intelHistoryWindow = document.getElementById('intelHistoryWindow');
  const intelHiddenEnergy = document.getElementById('intelHiddenEnergy');
  const intelCellEnergy = document.getElementById('intelCellEnergy');
  const intelSurprise = document.getElementById('intelSurprise');
  const chipModelStatus = document.getElementById('chipModelStatus');
  const intelModelStatus = document.getElementById('intelModelStatus');
  const intelCheckpointName = document.getElementById('intelCheckpointName');
  const intelCheckpointHash = document.getElementById('intelCheckpointHash');
  const intelTrainingMode = document.getElementById('intelTrainingMode');

  // Timeline
  const timelineTableBody = document.getElementById('timelineTableBody');

  // Controls
  const selectScenario = document.getElementById('selectScenario');
  const inputSeed = document.getElementById('inputSeed');
  const selectSpeed = document.getElementById('selectSpeed');
  const btnStart = document.getElementById('btnStart');
  const btnPause = document.getElementById('btnPause');
  const btnStep = document.getElementById('btnStep');
  const btnReset = document.getElementById('btnReset');

  // View Mode
  const btnModeOverview = document.getElementById('btnModeOverview');
  const btnModeTechnical = document.getElementById('btnModeTechnical');
  const evalBanner = document.getElementById('evalBanner');
  const chkGroundTruth = document.getElementById('chkGroundTruth');
  const evalLegends = document.querySelectorAll('.eval-only');

  // Initialize Spectrum Waterfall
  const waterfall = new SpectrumWaterfallRenderer('waterfallCanvas');

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
      scenarios.forEach(sc => {
        const opt = document.createElement('option');
        opt.value = sc.filename;
        opt.textContent = `${sc.name} (${sc.filename})`;
        selectScenario.appendChild(opt);
      });
    } catch (err) {
      console.error('Failed to load scenarios:', err);
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
    const mem = payload.temporal_memory || {};

    // 1. Header Status
    systemStateText.textContent = sys.state || 'IDLE';
    systemStateDot.className = 'status-dot';
    if (sys.state === 'RUNNING') systemStateDot.classList.add('active');
    else if (sys.state === 'PAUSED') systemStateDot.classList.add('paused');

    chipTime.textContent = `${(sys.simulation_time_s || 0).toFixed(1)} s`;
    chipStep.textContent = sys.step || 0;
    chipScenario.textContent = sys.scenario_name ? sys.scenario_name.replace('.yaml', '') : '--';

    // 2. Prediction Hero Card
    currentScanFreq.textContent = (pred.current_scan_mhz || 0).toFixed(1);
    currentScanBin.textContent = `Channel Bin ${pred.current_scan_bin != null ? pred.current_scan_bin : '--'}`;

    predictedNextFreq.textContent = (pred.predicted_next_mhz || 0).toFixed(1);
    predictedNextBin.textContent = `Channel Bin ${pred.predicted_next_bin != null ? pred.predicted_next_bin : '--'}`;

    const conf = pred.confidence_pct || 0;
    predConfidenceBadge.textContent = `CONFIDENCE: ${conf.toFixed(0)}% (${pred.confidence_level || 'MED'})`;
    confGaugeFill.style.width = `${conf}%`;
    confPercent.textContent = `${conf.toFixed(0)}%`;
    confLevelText.textContent = `Level: ${pred.confidence_level || 'MEDIUM'} Pattern Stability`;

    // Candidate Q-Value Chips
    if (pred.top_predictions && pred.top_predictions.length > 0) {
      topPredictionsList.innerHTML = '';
      pred.top_predictions.forEach(p => {
        const chip = document.createElement('span');
        chip.className = 'ranking-chip';
        chip.innerHTML = `<b>${p.rank}. ${p.frequency_mhz} MHz</b> (Q: ${p.q_value.toFixed(2)})`;
        topPredictionsList.appendChild(chip);
      });
    }

    // 3. Performance Cards
    metricIR.textContent = `${(perf.interception_ratio_pct || 0).toFixed(1)}%`;
    metricIRSub.textContent = `${perf.intercepted_opportunities || 0} / ${perf.total_opportunities || 0} Opportunities`;

    metricDetRate.textContent = `${(perf.detection_rate_pct || 0).toFixed(1)}%`;
    metricDetSub.textContent = `${perf.total_detections || 0} Detections Validated`;

    metricEfficiency.textContent = `${(perf.scan_efficiency_pct || 0).toFixed(1)}%`;
    metricTotalScans.textContent = `${perf.total_scans || 0} Sweeps Executed`;

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
    whyExplainText.textContent = arb.explanation || 'Evaluating neural prediction...';
    badgeArbitrationMode.textContent = arb.mode || 'DDQN_EXPLOIT';

    const caPct = arb.context_aware_weight_pct || 30;
    const lstmPct = arb.lstm_ddqn_weight_pct || 70;
    caWeightPct.textContent = `${caPct.toFixed(0)}%`;
    lstmWeightPct.textContent = `${lstmPct.toFixed(0)}%`;
    splitBarCA.style.width = `${caPct}%`;
    splitBarLSTM.style.width = `${lstmPct}%`;

    // 6. Intel Panel (Technical)
    intelHistoryWindow.textContent = `${mem.history_window || 10} observations`;
    intelHiddenEnergy.textContent = (mem.hidden_state_norm || 0).toFixed(3);
    intelCellEnergy.textContent = (mem.cell_state_norm || 0).toFixed(3);
    intelSurprise.textContent = (arb.surprise || 0).toFixed(3);

    // Neural Model status
    const nmodel = payload.neural_model || {};
    if (chipModelStatus) {
      chipModelStatus.textContent = `${nmodel.status || 'PRETRAINED'} (${nmodel.mode === 'FROZEN_INFERENCE' ? 'FROZEN' : nmodel.mode || 'FROZEN'})`;
    }
    if (intelModelStatus) {
      intelModelStatus.textContent = `${nmodel.status || 'PRETRAINED'} (${nmodel.mode || 'FROZEN'})`;
    }
    if (intelCheckpointName) {
      intelCheckpointName.textContent = nmodel.checkpoint_name || 'production_checkpoint.npz';
    }
    if (intelCheckpointHash) {
      intelCheckpointHash.textContent = nmodel.checkpoint_fingerprint ? `${nmodel.checkpoint_fingerprint}...` : '53abe2fe...';
    }
    if (intelTrainingMode) {
      intelTrainingMode.textContent = `${nmodel.runtime_training || 'DISABLED'} (INFERENCE ONLY)`;
    }

    // 7. Waterfall Canvas
    if (payload.bands_mhz) {
      waterfall.setBands(payload.bands_mhz);
    }
    if (payload.waterfall_events) {
      waterfall.setEvents(payload.waterfall_events);
    }

    // 8. Recent Timeline Table
    if (payload.recent_timeline && payload.recent_timeline.length > 0) {
      timelineTableBody.innerHTML = '';
      payload.recent_timeline.slice(0, 10).forEach(row => {
        const tr = document.createElement('tr');
        const detBadge = row.detected
          ? '<span class="badge-pill-hit">✓ DETECTED</span>'
          : '<span class="badge-pill-miss">✗ MISSED</span>';

        tr.innerHTML = `
          <td><b>t=${row.step}</b></td>
          <td class="text-cyan">${row.scanned_mhz.toFixed(1)} MHz</td>
          <td class="text-amber">${row.predicted_mhz.toFixed(1)} MHz</td>
          <td>${row.confidence_pct.toFixed(0)}%</td>
          <td>${detBadge}</td>
          <td><code>${row.arbitration_mode}</code></td>
        `;
        timelineTableBody.appendChild(tr);
      });
    }
  }

  // Setup WebSocket Stream with Auto-Reconnect
  let ws = null;
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Telemetry WebSocket connected.');
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event_type === 'TELEMETRY' || msg.event_type === 'SNAPSHOT') {
          updateUI(msg.payload);
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket closed. Retrying in 2s...');
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      ws.close();
    };
  }

  // Polling Fallback if WebSocket drops or for immediate initial state
  async function fetchInitialState() {
    try {
      const res = await fetch('/api/telemetry');
      const data = await res.json();
      updateUI(data);

      const statusRes = await fetch('/api/status');
      const statusData = await statusRes.json();
      if (statusData.scenario_name) {
        selectScenario.value = statusData.scenario_name;
      }
      if (statusData.speed) {
        selectSpeed.value = statusData.speed;
      }
    } catch (err) {
      console.error('Failed to fetch initial telemetry:', err);
    }
  }

  // Controls Event Listeners
  btnStart.addEventListener('click', async () => {
    await fetch('/api/simulation/start', { method: 'POST' });
  });

  btnPause.addEventListener('click', async () => {
    await fetch('/api/simulation/pause', { method: 'POST' });
  });

  btnStep.addEventListener('click', async () => {
    const res = await fetch('/api/simulation/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ steps: 1 })
    });
    const data = await res.json();
    updateUI(data);
  });

  btnReset.addEventListener('click', async () => {
    const scenario = selectScenario.value;
    const seed = parseInt(inputSeed.value, 10) || 42;
    const res = await fetch('/api/simulation/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        scenario_name: scenario,
        seed: seed,
        scheduler_name: 'hybrid_v41'
      })
    });
    const data = await res.json();
    waterfall.clear();
    updateUI(data.telemetry);
  });

  selectSpeed.addEventListener('change', async () => {
    await fetch('/api/simulation/speed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed: selectSpeed.value })
    });
  });

  selectScenario.addEventListener('change', async () => {
    btnReset.click();
  });

  // Init
  loadScenarios().then(() => {
    fetchInitialState();
    connectWebSocket();
  });
});
