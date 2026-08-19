/**
 * Shared Helper Utilities for DeepShield AI
 */

// Toast Notifications System
window.showToast = function(message, type = 'info', title = '') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const iconMap = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ'
  };

  const defaultTitles = {
    success: 'Success',
    error: 'Error',
    warning: 'Warning',
    info: 'Information'
  };

  toast.innerHTML = `
    <div style="font-weight: bold; font-size: 1.1rem; color: currentColor;">${iconMap[type] || 'ℹ'}</div>
    <div style="flex: 1;">
      <div style="font-weight: 600; font-size: 0.9rem;">${title || defaultTitles[type]}</div>
      <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 2px;">${message}</div>
    </div>
    <button class="toast-close" aria-label="Close">&times;</button>
  `;

  toast.querySelector('.toast-close').addEventListener('click', () => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(-10px)';
    setTimeout(() => toast.remove(), 200);
  });

  container.appendChild(toast);

  setTimeout(() => {
    if (document.body.contains(toast)) {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(-10px)';
      toast.style.transition = 'all 200ms ease';
      setTimeout(() => toast.remove(), 200);
    }
  }, 4200);
};

// Modal Handling System
window.openModal = function(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add('open');
  }
};

window.closeModal = function(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.remove('open');
  }
};

// Button Ripple Effect
window.initRippleEffect = function() {
  document.addEventListener('click', function(e) {
    const btn = e.target.closest('.btn');
    if (!btn) return;

    const rect = btn.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const ripple = document.createElement('span');
    ripple.className = 'ripple';
    ripple.style.left = `${x}px`;
    ripple.style.top = `${y}px`;

    btn.appendChild(ripple);

    setTimeout(() => {
      ripple.remove();
    }, 600);
  });
};

// Scroll Reveal Observer
window.initScrollObserver = function() {
  const elements = document.querySelectorAll('.fade-on-scroll');
  if (!elements.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-fade-in');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  elements.forEach(el => observer.observe(el));
};

// Mock Analysis Data Storage Helpers
const STORAGE_KEY = 'deepshield_analysis_result';

window.getDefaultAnalysisData = function(filename = 'speech_manipulation_sample.mp4', isFake = true) {
  return {
    filename: filename,
    fileSize: '42.8 MB',
    videoLength: '00:45',
    resolution: '1920x1080 (1080p)',
    fileFormat: 'MP4 / H.264',
    fps: 30,
    framesProcessed: 1350,
    analysisTime: '4.2s',
    timestamp: new Date().toISOString(),
    reportId: 'DS-' + Math.floor(100000 + Math.random() * 900000),
    result: isFake ? 'FAKE' : 'REAL',
    confidence: isFake ? 94.7 : 98.2,
    riskLevel: isFake ? 'Critical' : 'Very Low',
    riskScore: isFake ? 92 : 4,
    explanation: isFake 
      ? "Forensic neural scan detected critical facial spatial-temporal inconsistencies between frames 140 and 310. Speech audio exhibits a 180ms desynchronization relative to primary lip boundary keypoints. Generative diffusion artifacts were confirmed along the jawline and eye orbits."
      : "No neural generative artifacts detected. Spatial continuity across 1,350 frames exhibits authentic facial geometry and natural corneal reflection stability. Audio spectrogram matches natural human vocal harmonics.",
    faceAnalysis: {
      facesDetected: '1 Primary Face',
      landmarkStability: isFake ? '32% Instability (High)' : '98.5% Stable',
      eyeBlinkRate: isFake ? '0.2 blinks/min (Irregular)' : '14 blinks/min (Normal)',
      expressionSymmetry: isFake ? 'Asymmetrical Motion' : 'Natural Symmetrical',
      cornealReflection: isFake ? 'Light Angle Mismatch' : 'Consistent Light Vector',
      skinTexture: isFake ? 'Generative Diffusion Smoothing' : 'Authentic Subsurface Scatter'
    },
    audioAnalysis: {
      lipSyncScore: isFake ? '180ms Delay' : '< 12ms (Synchronized)',
      voiceConsistency: isFake ? 'Pitch Shift Artifacts' : 'Natural Vocal Spectrum',
      speechTiming: isFake ? 'Unnatural Pause Gaps' : 'Natural Cadence',
      backgroundNoise: isFake ? 'Discontinuous Noise Floor' : 'Consistent Acoustic Room',
      compressionArtifacts: isFake ? 'Phased Frequency Cutoff' : 'Standard AAC Codec'
    },
    indicators: [
      { label: 'Face Boundary Warping', detected: isFake, desc: 'Generative blending artifacts around jawline.' },
      { label: 'Lighting Vector Mismatch', detected: isFake, desc: 'Illumination on face does not match background.' },
      { label: 'Boundary Blend Artifacts', detected: isFake, desc: 'Unnatural edge blurring along hairline.' },
      { label: 'Head Movement Instability', detected: isFake, desc: 'Jittery translation between keyframe poses.' },
      { label: 'Texture Continuity Inconsistency', detected: isFake, desc: 'Temporal smoothing on cheeks and forehead.' },
      { label: 'Compression Codec Mismatch', detected: isFake, desc: 'Different quantization matrices in facial region.' },
      { label: 'Eye Blink Frequency Anomaly', detected: isFake, desc: 'Abnormally long intervals without eyelid movement.' },
      { label: 'Temporal Spectral Desync', detected: isFake, desc: 'Audio spectrogram lags facial gesture.' }
    ]
  };
};

window.saveAnalysisData = function(data) {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
};

window.getAnalysisData = function() {
  const saved = sessionStorage.getItem(STORAGE_KEY);
  if (saved) {
    try {
      return JSON.parse(saved);
    } catch(e) {
      console.error('Error parsing analysis data:', e);
    }
  }
  // Return default fake result if nothing saved
  return window.getDefaultAnalysisData();
};
