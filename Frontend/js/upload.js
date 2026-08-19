/**
 * Upload Page Logic for DeepShield AI
 */

document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');
  const uploadProgress = document.getElementById('upload-progress');
  const progressBar = document.getElementById('progress-bar');
  const progressPercent = document.getElementById('progress-percent');
  const uploadSpeed = document.getElementById('upload-speed');
  const videoPreviewContainer = document.getElementById('video-preview-container');
  const videoElement = document.getElementById('preview-video');
  const videoNameEl = document.getElementById('video-name');
  const videoSizeEl = document.getElementById('video-size');
  const videoDurationEl = document.getElementById('video-duration');
  const removeVideoBtn = document.getElementById('remove-video-btn');
  const analyzeBtn = document.getElementById('analyze-btn');

  let currentFile = null;
  let isSampleFile = false;
  let sampleType = 'fake';

  if (!dropZone || !fileInput) return;

  // Click to browse
  if (browseBtn) {
    browseBtn.addEventListener('click', (e) => {
      e.preventDefault();
      fileInput.click();
    });
  }

  // File selected via input
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag & Drop events
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.add('dragging');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.remove('dragging');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  // Sample video buttons
  document.querySelectorAll('[data-sample]').forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-sample'); // 'fake' or 'real'
      loadSampleVideo(type);
    });
  });

  function validateFile(file) {
    const validTypes = ['video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    const maxSize = 500 * 1024 * 1024; // 500MB

    // Check extension if type is empty
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ['mp4', 'avi', 'mov', 'mkv', 'webm'];

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
      if (window.showToast) {
        window.showToast('Invalid file format. Please upload MP4, AVI, MOV, MKV, or WEBM.', 'error', 'Upload Error');
      }
      return false;
    }

    if (file.size > maxSize) {
      if (window.showToast) {
        window.showToast('File exceeds 500MB size limit. Please choose a smaller file.', 'warning', 'File Too Large');
      }
      return false;
    }

    return true;
  }

  function handleFileSelected(file) {
    if (!validateFile(file)) return;

    currentFile = file;
    isSampleFile = false;

    simulateUpload(file.name, (file.size / (1024 * 1024)).toFixed(1) + ' MB', () => {
      // Show preview
      const objectUrl = URL.createObjectURL(file);
      showVideoPreview(file.name, (file.size / (1024 * 1024)).toFixed(1) + ' MB', objectUrl);
    });
  }

  function loadSampleVideo(type) {
    isSampleFile = true;
    sampleType = type;
    const filename = type === 'fake' ? 'deepfake_speech_manipulation.mp4' : 'authentic_presidential_address.mp4';
    const sizeStr = type === 'fake' ? '42.8 MB' : '38.4 MB';

    simulateUpload(filename, sizeStr, () => {
      // Using sample videos or placeholder canvas
      showVideoPreview(filename, sizeStr, 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4');
    });
  }

  function simulateUpload(filename, sizeStr, onComplete) {
    dropZone.style.display = 'none';
    uploadProgress.style.display = 'block';
    videoPreviewContainer.style.display = 'none';

    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.floor(Math.random() * 15) + 10;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        setTimeout(() => {
          uploadProgress.style.display = 'none';
          onComplete();
          if (window.showToast) {
            window.showToast(`"${filename}" uploaded successfully!`, 'success', 'Upload Complete');
          }
        }, 300);
      }

      progressBar.style.width = progress + '%';
      progressPercent.textContent = progress + '%';
      uploadSpeed.textContent = `${(12 + Math.random() * 5).toFixed(1)} MB/s • ${Math.ceil((100 - progress) / 20)}s remaining`;
    }, 150);
  }

  function showVideoPreview(name, size, videoSrc) {
    videoNameEl.textContent = name;
    videoSizeEl.textContent = size;
    videoElement.src = videoSrc;
    
    videoElement.onloadedmetadata = () => {
      const duration = videoElement.duration;
      const mins = Math.floor(duration / 60);
      const secs = Math.floor(duration % 60);
      videoDurationEl.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    videoPreviewContainer.style.display = 'block';
  }

  // Remove video button
  if (removeVideoBtn) {
    removeVideoBtn.addEventListener('click', () => {
      currentFile = null;
      isSampleFile = false;
      videoElement.src = '';
      videoPreviewContainer.style.display = 'none';
      dropZone.style.display = 'block';
      fileInput.value = '';
    });
  }

  // Analyze button click -> save data and redirect to analysis.html
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', () => {
      let analysisData;
      if (isSampleFile) {
        analysisData = window.getDefaultAnalysisData(
          sampleType === 'fake' ? 'deepfake_speech_manipulation.mp4' : 'authentic_presidential_address.mp4',
          sampleType === 'fake'
        );
      } else if (currentFile) {
        const isFakeMock = Math.random() > 0.3; // 70% chance of fake for demo
        analysisData = window.getDefaultAnalysisData(currentFile.name, isFakeMock);
        analysisData.fileSize = (currentFile.size / (1024 * 1024)).toFixed(1) + ' MB';
      } else {
        analysisData = window.getDefaultAnalysisData();
      }

      window.saveAnalysisData(analysisData);
      window.location.href = '/pages/analysis.html';
    });
  }
});
