/**
 * Analysis Page Simulated Scanner Logic for DeepShield AI
 * Dark Cyber Forensic Reticle Visualizer
 */

document.addEventListener('DOMContentLoaded', () => {
  const progressBar = document.getElementById('analysis-progress');
  const percentEl = document.getElementById('analysis-percent');
  const countdownEl = document.getElementById('analysis-countdown');
  const stepsContainer = document.getElementById('steps-list');

  if (!progressBar || !percentEl) return;

  // Retrieve saved file metadata
  const data = window.getAnalysisData();
  const filenameEl = document.getElementById('analyzing-filename');
  if (filenameEl && data) {
    filenameEl.textContent = data.filename;
  }

  const steps = [
    { id: 'step-1', name: 'Uploading & Parsing Video Stream', duration: 1200 },
    { id: 'step-2', name: 'Extracting Video Frames & Keypoints', duration: 1500 },
    { id: 'step-3', name: 'Facial Landmark & Geometry Detection', duration: 1800 },
    { id: 'step-4', name: 'Audio Spectrum & Lip-Sync Alignment', duration: 1500 },
    { id: 'step-5', name: 'Metadata & Compression Artifact Check', duration: 1200 },
    { id: 'step-6', name: 'Compiling Final Forensic Report', duration: 800 }
  ];

  // Render initial steps list
  if (stepsContainer) {
    stepsContainer.innerHTML = steps.map((step, idx) => `
      <div class="analysis-step-item ${idx === 0 ? 'active' : 'pending'}" id="${step.id}" style="display: flex; align-items: center; justify-content: space-between; padding: 1rem 1.25rem; background: var(--bg-card-solid); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); margin-bottom: 0.75rem; transition: all 200ms ease;">
        <div style="display: flex; align-items: center; gap: 1rem;">
          <div class="step-status-icon" style="width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; background: rgba(255,255,255,0.05); color: var(--text-muted);">
            ${idx + 1}
          </div>
          <span class="step-label" style="font-weight: 500; font-size: 0.95rem; color: ${idx === 0 ? 'var(--text-primary)' : 'var(--text-secondary)'};">${step.name}</span>
        </div>
        <div class="step-badge" style="font-size: 0.8rem; font-weight: 600;">
          ${idx === 0 ? '<span style="color: var(--cyan);">Processing...</span>' : '<span style="color: var(--text-muted);">Queued</span>'}
        </div>
      </div>
    `).join('');
  }

  // Animate Canvas Reticle Grid
  initAnalysisCanvas();

  // Progress Interval
  const totalDuration = steps.reduce((sum, s) => sum + s.duration, 0);
  const startTime = Date.now();

  const progressInterval = setInterval(() => {
    const elapsed = Date.now() - startTime;
    const progress = Math.min(Math.floor((elapsed / totalDuration) * 100), 100);

    progressBar.style.width = progress + '%';
    percentEl.textContent = progress + '%';

    // Countdown calculation
    const remainingSecs = Math.max(Math.ceil((totalDuration - elapsed) / 1000), 0);
    if (countdownEl) {
      countdownEl.textContent = `Estimated inspection time: ${remainingSecs}s`;
    }

    // Step updates
    let stepCumulativeTime = 0;
    for (let i = 0; i < steps.length; i++) {
      stepCumulativeTime += steps[i].duration;
      if (elapsed >= stepCumulativeTime) {
        markStepComplete(i);
        if (i + 1 < steps.length) {
          markStepActive(i + 1);
        }
      }
    }

    if (progress >= 100) {
      clearInterval(progressInterval);
      markStepComplete(steps.length - 1);
      
      if (countdownEl) {
        countdownEl.textContent = 'Inspection Complete. Compiling Report...';
      }

      setTimeout(() => {
        window.location.href = '/pages/results.html';
      }, 900);
    }
  }, 100);

  function markStepActive(idx) {
    const item = document.getElementById(steps[idx].id);
    if (!item || item.classList.contains('active') || item.classList.contains('completed')) return;

    item.className = 'analysis-step-item active';
    item.style.borderColor = 'var(--cyan)';
    
    const icon = item.querySelector('.step-status-icon');
    if (icon) {
      icon.style.background = 'var(--cyan)';
      icon.style.color = '#FFFFFF';
      icon.textContent = '•';
    }

    const label = item.querySelector('.step-label');
    if (label) label.style.color = 'var(--text-primary)';

    const badge = item.querySelector('.step-badge');
    if (badge) badge.innerHTML = '<span style="color: var(--cyan); font-weight: 600;">Analyzing...</span>';
  }

  function markStepComplete(idx) {
    const item = document.getElementById(steps[idx].id);
    if (!item) return;

    item.className = 'analysis-step-item completed';
    item.style.borderColor = 'var(--border-subtle)';

    const icon = item.querySelector('.step-status-icon');
    if (icon) {
      icon.style.background = 'var(--success)';
      icon.style.color = '#ffffff';
      icon.textContent = '✓';
    }

    const badge = item.querySelector('.step-badge');
    if (badge) badge.innerHTML = '<span style="color: var(--success); font-weight: 600;">Complete</span>';
  }

  function initAnalysisCanvas() {
    const canvas = document.getElementById('scanner-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width = canvas.width = canvas.parentElement.clientWidth || 280;
    let height = canvas.height = canvas.parentElement.clientHeight || 280;

    let angle = 0;

    function render() {
      ctx.clearRect(0, 0, width, height);

      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) / 2 - 20;

      // Outer reticle circle
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.stroke();

      // Inner concentric ring
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.2)';
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * 0.6, 0, Math.PI * 2);
      ctx.stroke();

      // Crosshairs
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.beginPath();
      ctx.moveTo(centerX - radius - 10, centerY);
      ctx.lineTo(centerX + radius + 10, centerY);
      ctx.moveTo(centerX, centerY - radius - 10);
      ctx.lineTo(centerX, centerY + radius + 10);
      ctx.stroke();

      // Sweeping reticle arc line
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(angle);

      ctx.strokeStyle = '#06B6D4';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#06B6D4';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 0.35);
      ctx.stroke();

      ctx.restore();

      angle += 0.025;
      requestAnimationFrame(render);
    }

    render();
  }
});
