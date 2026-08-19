/**
 * Results Dashboard Logic for DeepShield AI
 */

document.addEventListener('DOMContentLoaded', () => {
  const data = window.getAnalysisData();
  if (!data) return;

  renderTopHeroCard(data);
  renderSummaryStats(data);
  renderExplanationBox(data);
  renderRiskMeter(data);
  renderAuthenticityTimeline(data);
  renderFrameGrid(data);
  renderForensicCards(data);
  renderMetadataTable(data);
  renderIndicatorsChecklist(data);

  // Initialize Heatmap Toggle Listener
  const heatmapBtn = document.getElementById('toggle-heatmap-btn');
  if (heatmapBtn) {
    let heatmapActive = false;
    heatmapBtn.addEventListener('click', () => {
      heatmapActive = !heatmapActive;
      document.querySelectorAll('.frame-heatmap-overlay').forEach(overlay => {
        overlay.style.opacity = heatmapActive ? '0.75' : '0';
      });
      heatmapBtn.classList.toggle('btn-cyan', heatmapActive);
      heatmapBtn.classList.toggle('btn-outline', !heatmapActive);
      heatmapBtn.innerHTML = heatmapActive 
        ? '⚡ Heatmap Active (Click to Hide)' 
        : '🔍 Toggle Facial Heatmap Overlay';
      
      if (window.showToast) {
        window.showToast(heatmapActive ? 'Facial anomaly heatmap overlay enabled' : 'Heatmap hidden', 'info');
      }
    });
  }

  // Share Modal Listener
  const shareBtn = document.getElementById('share-results-btn');
  if (shareBtn) {
    shareBtn.addEventListener('click', () => {
      if (window.openModal) {
        window.openModal('share-modal');
      }
    });
  }

  // Copy share link button
  const copyLinkBtn = document.getElementById('copy-share-link-btn');
  if (copyLinkBtn) {
    copyLinkBtn.addEventListener('click', () => {
      const urlInput = document.getElementById('share-url-input');
      if (urlInput) {
        urlInput.select();
        navigator.clipboard.writeText(urlInput.value).then(() => {
          if (window.showToast) {
            window.showToast('Results link copied to clipboard!', 'success');
          }
        });
      }
    });
  }
});

function renderTopHeroCard(data) {
  const isFake = data.result === 'FAKE';
  const badgeEl = document.getElementById('hero-result-badge');
  const confidenceEl = document.getElementById('hero-confidence-score');
  const circleProgress = document.getElementById('hero-circle-progress');
  const filenameEl = document.getElementById('results-filename');

  if (filenameEl) filenameEl.textContent = data.filename;

  if (badgeEl) {
    badgeEl.className = `badge ${isFake ? 'badge-fake animate-pulse-red' : 'badge-real animate-pulse-green'}`;
    badgeEl.innerHTML = `<span class="badge-dot"></span> DETECTED ${data.result}`;
  }

  // Animate confidence number from 0 to target
  if (confidenceEl) {
    let currentScore = 0;
    const targetScore = data.confidence;
    const duration = 1200;
    const stepTime = 20;
    const increment = targetScore / (duration / stepTime);

    const timer = setInterval(() => {
      currentScore += increment;
      if (currentScore >= targetScore) {
        currentScore = targetScore;
        clearInterval(timer);
      }
      confidenceEl.textContent = currentScore.toFixed(1) + '%';
    }, stepTime);
  }

  // Animate circular SVG stroke-dashoffset
  if (circleProgress) {
    const circumference = 2 * Math.PI * 54; // r=54 -> ~339.29
    circleProgress.style.strokeDasharray = circumference;
    circleProgress.style.strokeDashoffset = circumference;

    const offset = circumference - (data.confidence / 100) * circumference;
    setTimeout(() => {
      circleProgress.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1)';
      circleProgress.style.strokeDashoffset = offset;
      circleProgress.style.stroke = isFake ? 'var(--danger)' : 'var(--success)';
    }, 100);
  }
}

function renderSummaryStats(data) {
  const statsContainer = document.getElementById('summary-stats-grid');
  if (!statsContainer) return;

  const isFake = data.result === 'FAKE';

  const stats = [
    { label: 'Detection Result', val: data.result, color: isFake ? 'var(--danger)' : 'var(--success)' },
    { label: 'Confidence Score', val: data.confidence + '%', color: 'var(--cyan)' },
    { label: 'Analysis Speed', val: data.analysisTime, color: 'var(--text-primary)' },
    { label: 'Video Duration', val: data.videoLength, color: 'var(--text-primary)' },
    { label: 'File Codec', val: data.fileFormat, color: 'var(--text-primary)' },
    { label: 'Frames Processed', val: data.framesProcessed + ' frames', color: 'var(--text-primary)' },
    { label: 'Model Version', val: 'v4.2 Neural-Scan', color: 'var(--primary)' }
  ];

  statsContainer.innerHTML = stats.map(s => `
    <div class="card card-solid" style="padding: 1rem 1.25rem; border-radius: var(--radius-md);">
      <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.3rem;">${s.label}</div>
      <div style="font-size: 1.25rem; font-weight: 700; font-family: var(--font-heading); color: ${s.color};">${s.val}</div>
    </div>
  `).join('');
}

function renderExplanationBox(data) {
  const el = document.getElementById('ai-explanation-text');
  if (el) {
    el.textContent = `"${data.explanation}"`;
  }
}

function renderRiskMeter(data) {
  const marker = document.getElementById('risk-meter-marker');
  const levelText = document.getElementById('risk-level-text');

  if (marker) {
    setTimeout(() => {
      marker.style.transition = 'left 1s cubic-bezier(0.16, 1, 0.3, 1)';
      marker.style.left = `${data.riskScore}%`;
    }, 200);
  }

  if (levelText) {
    levelText.textContent = `${data.riskLevel} Risk (${data.riskScore}/100)`;
    levelText.style.color = data.riskScore > 70 ? 'var(--danger)' : data.riskScore > 40 ? 'var(--warning)' : 'var(--success)';
  }
}

function renderAuthenticityTimeline(data) {
  const container = document.getElementById('timeline-bars');
  if (!container) return;

  const totalFrames = 45; // 45 seconds timeline
  const isFake = data.result === 'FAKE';

  let barsHtml = '';
  for (let i = 0; i < totalFrames; i++) {
    // Generate realistic confidence dips if fake
    let frameConf = 98 - Math.random() * 5;
    if (isFake && (i >= 12 && i <= 32)) {
      frameConf = 15 + Math.random() * 25; // low confidence (fake)
    }

    let color = 'var(--success)';
    if (frameConf < 50) color = 'var(--danger)';
    else if (frameConf < 80) color = 'var(--warning)';

    const timeSec = i < 10 ? `00:0${i}` : `00:${i}`;

    barsHtml += `
      <div class="timeline-bar-item" style="flex: 1; height: 100%; display: flex; align-items: flex-end; position: relative; group" title="Sec ${timeSec} - ${frameConf.toFixed(1)}% Authenticity">
        <div style="width: 100%; height: ${frameConf}%; background-color: ${color}; border-radius: 2px; transition: height 600ms ease;"></div>
      </div>
    `;
  }

  container.innerHTML = barsHtml;
}

function renderFrameGrid(data) {
  const grid = document.getElementById('frame-analysis-grid');
  if (!grid) return;

  const isFake = data.result === 'FAKE';
  let framesHtml = '';

  for (let i = 1; i <= 8; i++) {
    const timestamp = `00:${(i * 5).toString().padStart(2, '0')}`;
    const frameNumber = i * 150;
    const isManipulated = isFake && (i >= 3 && i <= 6);
    const score = isManipulated ? (88 + Math.random() * 10).toFixed(1) : (3 + Math.random() * 8).toFixed(1);

    framesHtml += `
      <div class="card card-solid" style="padding: 0.75rem; border-radius: var(--radius-md); position: relative; overflow: hidden;">
        <div style="width: 100%; height: 110px; background: #0B1120; border-radius: var(--radius-sm); position: relative; display: flex; align-items: center; justify-content: center;">
          <!-- SVG Wireframe Face Silhouette -->
          <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="${isManipulated ? 'var(--danger)' : 'var(--text-muted)'}" stroke-width="1.5">
            <path d="M12 2a8 8 0 0 0-8 8c0 5.25 7 12 8 12s8-6.75 8-12a8 8 0 0 0-8-8z"/>
            <circle cx="9" cy="9" r="1"/>
            <circle cx="15" cy="9" r="1"/>
            <path d="M9 15s1.5 1.5 3 1.5 3-1.5 3-1.5"/>
          </svg>
          
          <!-- Heatmap Overlay Canvas -->
          <div class="frame-heatmap-overlay" style="position: absolute; top:0; left:0; width:100%; height:100%; background: radial-gradient(circle at 50% 45%, ${isManipulated ? 'rgba(239, 68, 68, 0.65)' : 'rgba(34, 197, 94, 0.2)'} 0%, transparent 60%); opacity: 0; pointer-events: none; transition: opacity 300ms ease;"></div>

          <div style="position: absolute; bottom: 6px; left: 8px; font-size: 0.75rem; font-family: var(--font-mono); color: var(--text-secondary); background: rgba(0,0,0,0.6); padding: 2px 6px; border-radius: 4px;">
            ${timestamp}
          </div>
        </div>
        <div style="margin-top: 0.6rem; display: flex; align-items: center; justify-content: space-between;">
          <span style="font-size: 0.8rem; font-weight: 500; color: var(--text-secondary);">Frame #${frameNumber}</span>
          <span class="badge ${isManipulated ? 'badge-fake' : 'badge-real'}" style="font-size: 0.7rem; padding: 0.15rem 0.5rem;">
            ${isManipulated ? score + '% FAKE' : 'REAL'}
          </span>
        </div>
      </div>
    `;
  }

  grid.innerHTML = framesHtml;
}

function renderForensicCards(data) {
  // Face Analysis
  const faceContainer = document.getElementById('face-cards-grid');
  if (faceContainer && data.faceAnalysis) {
    const items = [
      { label: 'Faces Detected', val: data.faceAnalysis.facesDetected, icon: '👤' },
      { label: 'Landmark Stability', val: data.faceAnalysis.landmarkStability, icon: '📐' },
      { label: 'Eye Blink Consistency', val: data.faceAnalysis.eyeBlinkRate, icon: '👁️' },
      { label: 'Facial Expression', val: data.faceAnalysis.expressionSymmetry, icon: '😊' },
      { label: 'Reflection Analysis', val: data.faceAnalysis.cornealReflection, icon: '💡' },
      { label: 'Skin Texture', val: data.faceAnalysis.skinTexture, icon: '🔍' }
    ];

    faceContainer.innerHTML = items.map(item => `
      <div class="card card-solid" style="padding: 1rem; border-radius: var(--radius-md);">
        <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--cyan); margin-bottom: 0.4rem;">
          <span>${item.icon}</span>
          <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary);">${item.label}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary);">${item.val}</div>
      </div>
    `).join('');
  }

  // Audio Analysis
  const audioContainer = document.getElementById('audio-cards-grid');
  if (audioContainer && data.audioAnalysis) {
    const items = [
      { label: 'Lip Sync Score', val: data.audioAnalysis.lipSyncScore, icon: '👄' },
      { label: 'Voice Consistency', val: data.audioAnalysis.voiceConsistency, icon: '🎙️' },
      { label: 'Speech Timing', val: data.audioAnalysis.speechTiming, icon: '⏱️' },
      { label: 'Background Noise', val: data.audioAnalysis.backgroundNoise, icon: '🔊' },
      { label: 'Compression Artifacts', val: data.audioAnalysis.compressionArtifacts, icon: '📊' }
    ];

    audioContainer.innerHTML = items.map(item => `
      <div class="card card-solid" style="padding: 1rem; border-radius: var(--radius-md);">
        <div style="display: flex; align-items: center; gap: 0.5rem; color: var(--primary); margin-bottom: 0.4rem;">
          <span>${item.icon}</span>
          <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary);">${item.label}</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-primary);">${item.val}</div>
      </div>
    `).join('');
  }
}

function renderMetadataTable(data) {
  const tableBody = document.getElementById('metadata-table-body');
  if (!tableBody) return;

  const rows = [
    { key: 'File Name', val: data.filename },
    { key: 'File Size', val: data.fileSize },
    { key: 'Resolution', val: data.resolution },
    { key: 'Container Format / Codec', val: data.fileFormat },
    { key: 'Frame Rate', val: data.fps + ' FPS' },
    { key: 'Total Frames Analyzed', val: data.framesProcessed },
    { key: 'Scan Completed Date', val: new Date(data.timestamp).toLocaleString() },
    { key: 'Report Reference Hash', val: data.reportId }
  ];

  tableBody.innerHTML = rows.map(r => `
    <tr style="border-bottom: 1px solid var(--border-subtle);">
      <td style="padding: 0.8rem 1rem; font-weight: 500; color: var(--text-secondary); width: 35%;">${r.key}</td>
      <td style="padding: 0.8rem 1rem; font-family: var(--font-mono); color: var(--text-primary);">${r.val}</td>
    </tr>
  `).join('');
}

function renderIndicatorsChecklist(data) {
  const container = document.getElementById('suspicious-indicators-list');
  if (!container || !data.indicators) return;

  container.innerHTML = data.indicators.map(ind => `
    <div style="display: flex; align-items: flex-start; gap: 0.8rem; padding: 0.85rem 1rem; background: rgba(15, 23, 42, 0.4); border: 1px solid var(--border-subtle); border-radius: var(--radius-md);">
      <div style="font-size: 1.1rem; color: ${ind.detected ? 'var(--danger)' : 'var(--success)'}; margin-top: 2px;">
        ${ind.detected ? '✕' : '✓'}
      </div>
      <div>
        <div style="font-size: 0.95rem; font-weight: 600; color: ${ind.detected ? 'var(--text-primary)' : 'var(--text-secondary)'};">
          ${ind.label} ${ind.detected ? '<span style="font-size: 0.75rem; color: var(--danger); font-weight: 700; margin-left: 6px;">[ANOMALY]</span>' : ''}
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">
          ${ind.desc}
        </div>
      </div>
    </div>
  `).join('');
}
