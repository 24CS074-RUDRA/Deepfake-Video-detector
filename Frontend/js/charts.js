/**
 * Chart.js Visualizations for DeepShield AI Results Dashboard
 * Dark Cyber Forensic Theme
 */

document.addEventListener('DOMContentLoaded', () => {
  // Check if Chart.js is available
  if (typeof Chart === 'undefined') {
    console.warn('Chart.js not loaded.');
    return;
  }

  const data = window.getAnalysisData();
  const isFake = data ? data.result === 'FAKE' : true;

  initConfidenceLineChart(isFake);
  initRiskDoughnutChart(isFake);
  initDetectionBarChart(isFake);
});

function initConfidenceLineChart(isFake) {
  const ctx = document.getElementById('confidenceLineChart');
  if (!ctx) return;

  const labels = Array.from({ length: 15 }, (_, i) => `${i * 3}s`);
  
  // Data generator for line chart
  const confScores = labels.map((_, i) => {
    if (isFake && (i >= 3 && i <= 10)) {
      return Math.floor(15 + Math.random() * 25);
    }
    return Math.floor(92 + Math.random() * 7);
  });

  const primaryColor = isFake ? '#EF4444' : '#10B981';
  const fillColor = isFake ? 'rgba(239, 68, 68, 0.15)' : 'rgba(16, 185, 129, 0.15)';

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Frame Authenticity (%)',
        data: confScores,
        borderColor: primaryColor,
        backgroundColor: fillColor,
        fill: true,
        tension: 0.3,
        pointRadius: 4,
        pointBackgroundColor: primaryColor,
        pointBorderColor: '#0B0F19',
        pointBorderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#111827',
          titleColor: '#F9FAFB',
          bodyColor: '#9CA3AF',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          padding: 10
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#9CA3AF', font: { family: 'Inter', size: 11 } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#9CA3AF', font: { family: 'Inter', size: 11 } }
        }
      }
    }
  });
}

function initRiskDoughnutChart(isFake) {
  const ctx = document.getElementById('riskDoughnutChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Facial Geometry', 'Audio Lip-Sync', 'Metadata Codec', 'Temporal Continuity'],
      datasets: [{
        data: isFake ? [42, 28, 18, 12] : [8, 5, 4, 3],
        backgroundColor: ['#EF4444', '#06B6D4', '#3B82F6', '#8B5CF6'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#9CA3AF', padding: 14, font: { family: 'Inter', size: 11 } }
        }
      },
      cutout: '72%'
    }
  });
}

function initDetectionBarChart(isFake) {
  const ctx = document.getElementById('detectionBarChart');
  if (!ctx) return;

  const barColor = isFake ? '#EF4444' : '#10B981';

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Mesh Stability', 'Blink Rate', 'Lip Alignment', 'Frequency Noise', 'Compression'],
      datasets: [{
        label: 'Anomaly Score',
        data: isFake ? [88, 92, 79, 65, 84] : [12, 8, 15, 9, 11],
        backgroundColor: barColor,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#9CA3AF', font: { family: 'Inter', size: 11 } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#9CA3AF', font: { family: 'Inter', size: 11 } }
        }
      }
    }
  });
}
