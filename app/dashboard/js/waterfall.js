/**
 * SIH26055 — High-Performance Frequency-Time Canvas Waterfall Renderer
 * Renders live receiver scan trajectory, predictive targets, and detection events.
 */

class SpectrumWaterfallRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.events = [];
    this.maxEvents = 70;
    this.showGroundTruth = false;

    this.minFreqMHz = 100.0;
    this.maxFreqMHz = 700.0;

    this.initCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  initCanvas() {
    this.resizeCanvas();
  }

  resizeCanvas() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.round(rect.width || this.canvas.clientWidth || this.canvas.offsetWidth || (this.canvas.parentElement ? this.canvas.parentElement.clientWidth : 1200) || 1200);
    const h = Math.round(rect.height || this.canvas.clientHeight || this.canvas.offsetHeight || 300);
    const dpr = window.devicePixelRatio || 1;

    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.scale(dpr, dpr);
    this.width = w;
    this.height = h;
    this.draw();
  }

  setBands(bandsMHz) {
    if (bandsMHz && bandsMHz.length > 0) {
      this.minFreqMHz = Math.min(...bandsMHz) - 10.0;
      this.maxFreqMHz = Math.max(...bandsMHz) + 10.0;
    }
  }

  setEvents(eventsList) {
    this.events = eventsList ? eventsList.slice(-this.maxEvents) : [];
    this.draw();
  }

  pushEvent(event) {
    this.events.push(event);
    if (this.events.length > this.maxEvents) {
      this.events.shift();
    }
    this.draw();
  }

  clear() {
    this.events = [];
    this.draw();
  }

  freqToY(freqMHz) {
    const pad = 24;
    const h = this.height - pad * 2;
    const ratio = (freqMHz - this.minFreqMHz) / Math.max(this.maxFreqMHz - this.minFreqMHz, 1.0);
    // Invert Y so lower frequencies are at bottom, higher at top
    return this.height - pad - ratio * h;
  }

  stepToX(index, total) {
    const padLeft = 60;
    const padRight = 30;
    const w = this.width - padLeft - padRight;
    if (total <= 1) return padLeft + 20;
    const span = Math.max(total - 1, 25);
    return padLeft + (index / span) * w;
  }

  draw() {
    if (!this.width || !this.height || this.width < 50 || this.height < 50) {
      this.resizeCanvas();
    }

    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    if (!w || !h) return;

    // 1. Clear background
    ctx.fillStyle = '#080c14';
    ctx.fillRect(0, 0, w, h);

    // 2. Draw frequency grid lines
    ctx.strokeStyle = '#162032';
    ctx.lineWidth = 1;
    ctx.font = '10px "SF Mono", Menlo, Consolas, monospace';
    ctx.fillStyle = '#475569';
    ctx.textAlign = 'right';

    const freqStep = 100.0;
    const startFreq = Math.ceil(this.minFreqMHz / freqStep) * freqStep;
    for (let f = startFreq; f <= this.maxFreqMHz; f += freqStep) {
      const y = this.freqToY(f);
      ctx.beginPath();
      ctx.moveTo(50, y);
      ctx.lineTo(w - 20, y);
      ctx.stroke();

      ctx.fillText(`${f.toFixed(0)} MHz`, 48, y + 3);
    }

    if (this.events.length === 0) {
      ctx.textAlign = 'center';
      ctx.fillStyle = '#64748b';
      ctx.font = '13px "SF Mono", monospace';
      ctx.fillText('STANDBY — CLICK ▶ START MISSION OR ⏭ STEP TO SCAN', w / 2, h / 2);
      return;
    }

    const n = this.events.length;
    const latestX = this.stepToX(n - 1, n);

    // Subtle sweep head scanline
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(latestX, 10);
    ctx.lineTo(latestX, h - 20);
    ctx.stroke();

    // 3. Draw Scan Trajectory Line (Cyan)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.5)';
    ctx.lineWidth = 2;
    for (let i = 0; i < n; i++) {
      const ev = this.events[i];
      const x = this.stepToX(i, n);
      const y = this.freqToY(ev.scanned_mhz);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 4. Draw Ground Truth Overlay (if enabled)
    if (this.showGroundTruth) {
      ctx.strokeStyle = 'rgba(255, 23, 68, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);

      for (let i = 0; i < n; i++) {
        const ev = this.events[i];
        if (ev.ground_truth && ev.ground_truth.length > 0) {
          const x = this.stepToX(i, n);
          for (const gt of ev.ground_truth) {
            const y = this.freqToY(gt.frequency_mhz);
            ctx.fillStyle = 'rgba(255, 23, 68, 0.7)';
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
      ctx.setLineDash([]);
    }

    // 5. Draw Prediction Markers (Amber Diamond / Reticle)
    for (let i = 0; i < n; i++) {
      const ev = this.events[i];
      if (ev.predicted_mhz != null) {
        const x = this.stepToX(i, n);
        const y = this.freqToY(ev.predicted_mhz);

        ctx.strokeStyle = '#ffab00';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x - 5, y);
        ctx.lineTo(x + 5, y);
        ctx.moveTo(x, y - 5);
        ctx.lineTo(x, y + 5);
        ctx.stroke();
      }
    }

    // 6. Draw Scanned & Detected Points
    for (let i = 0; i < n; i++) {
      const ev = this.events[i];
      const x = this.stepToX(i, n);
      const y = this.freqToY(ev.scanned_mhz);

      if (ev.detected) {
        // Neon Green Glowing Detection Hit
        ctx.fillStyle = '#00e676';
        ctx.shadowColor = '#00e676';
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0; // reset
      } else {
        // Cyan Ring Scanned Point
        ctx.fillStyle = '#080c14';
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    // 7. Time axis labels at bottom
    ctx.textAlign = 'left';
    ctx.fillStyle = '#475569';
    ctx.font = '10px "SF Mono", monospace';
    const firstStep = this.events[0].step;
    const lastStep = this.events[n - 1].step;
    ctx.fillText(`t = ${firstStep}`, 60, h - 6);
    ctx.textAlign = 'right';
    ctx.fillText(`t = ${lastStep} (Latest)`, latestX, h - 6);
  }
}

window.SpectrumWaterfallRenderer = SpectrumWaterfallRenderer;
