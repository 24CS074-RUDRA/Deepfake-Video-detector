/**
 * Detailed Forensic Report Logic for DeepShield AI
 */

document.addEventListener('DOMContentLoaded', () => {
  const data = window.getAnalysisData();
  if (!data) return;

  renderReportDetails(data);

  // Download PDF / Print Handler
  const printBtn = document.getElementById('print-report-btn');
  if (printBtn) {
    printBtn.addEventListener('click', () => {
      window.print();
    });
  }

  const pdfBtn = document.getElementById('download-pdf-btn');
  if (pdfBtn) {
    pdfBtn.addEventListener('click', () => {
      if (window.showToast) {
        window.showToast('Preparing PDF document... Select "Save as PDF" in print dialog.', 'info');
      }
      setTimeout(() => {
        window.print();
      }, 500);
    });
  }

  // Download JSON Handler
  const jsonBtn = document.getElementById('download-json-btn');
  if (jsonBtn) {
    jsonBtn.addEventListener('click', () => {
      const jsonString = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `DeepShield_Forensic_Report_${data.reportId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      if (window.showToast) {
        window.showToast('JSON Forensic Log downloaded successfully', 'success');
      }
    });
  }
});

function renderReportDetails(data) {
  const isFake = data.result === 'FAKE';

  // Report Header Metadata
  const reportIdEl = document.getElementById('report-id-text');
  const reportDateEl = document.getElementById('report-date-text');
  const reportFilenameEl = document.getElementById('report-filename-text');

  if (reportIdEl) reportIdEl.textContent = data.reportId;
  if (reportDateEl) reportDateEl.textContent = new Date(data.timestamp).toUTCString();
  if (reportFilenameEl) reportFilenameEl.textContent = data.filename;

  // Verdict Banner
  const verdictBadge = document.getElementById('report-verdict-badge');
  const confidenceScore = document.getElementById('report-confidence-score');
  const riskLevel = document.getElementById('report-risk-level');

  if (verdictBadge) {
    verdictBadge.className = `badge ${isFake ? 'badge-fake' : 'badge-real'}`;
    verdictBadge.textContent = `VERDICT: ${data.result}`;
  }

  if (confidenceScore) confidenceScore.textContent = `${data.confidence}%`;
  if (riskLevel) {
    riskLevel.textContent = `${data.riskLevel} (${data.riskScore}/100)`;
    riskLevel.style.color = isFake ? 'var(--danger)' : 'var(--success)';
  }

  // Video Metadata Table
  const metaTable = document.getElementById('report-meta-table');
  if (metaTable) {
    metaTable.innerHTML = `
      <tr><td style="padding:0.5rem; font-weight:600; color:var(--text-secondary);">File Name:</td><td style="padding:0.5rem; font-family:var(--font-mono);">${data.filename}</td></tr>
      <tr><td style="padding:0.5rem; font-weight:600; color:var(--text-secondary);">File Size:</td><td style="padding:0.5rem; font-family:var(--font-mono);">${data.fileSize}</td></tr>
      <tr><td style="padding:0.5rem; font-weight:600; color:var(--text-secondary);">Resolution:</td><td style="padding:0.5rem; font-family:var(--font-mono);">${data.resolution}</td></tr>
      <tr><td style="padding:0.5rem; font-weight:600; color:var(--text-secondary);">Codec:</td><td style="padding:0.5rem; font-family:var(--font-mono);">${data.fileFormat}</td></tr>
      <tr><td style="padding:0.5rem; font-weight:600; color:var(--text-secondary);">Duration / Frames:</td><td style="padding:0.5rem; font-family:var(--font-mono);">${data.videoLength} (${data.framesProcessed} frames)</td></tr>
    `;
  }

  // Explanation
  const expEl = document.getElementById('report-explanation-text');
  if (expEl) {
    expEl.textContent = data.explanation;
  }

  // Suspicious Indicators List
  const indContainer = document.getElementById('report-indicators-grid');
  if (indContainer && data.indicators) {
    indContainer.innerHTML = data.indicators.map(ind => `
      <div style="padding: 0.65rem 0.85rem; background: rgba(15,23,42,0.5); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); font-size: 0.85rem; display: flex; justify-content: space-between; align-items: center;">
        <span>${ind.label}</span>
        <span style="font-weight:700; color:${ind.detected ? 'var(--danger)' : 'var(--success)'};">
          ${ind.detected ? 'DETECTED' : 'CLEAR'}
        </span>
      </div>
    `).join('');
  }
}
