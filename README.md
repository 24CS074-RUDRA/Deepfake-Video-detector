# Explainable Face-Swap Deepfake Detection

Smart India Hackathon project for detecting face-swap deepfake videos with spatial, frequency, and temporal analysis.

## Implemented

- OpenCV video frame extraction.
- RetinaFace face detection and largest-face cropping.
- FFT magnitude images for cropped faces.
- Video-level train/validation/test metadata splits.
- RGB + FFT sequence dataset: `(rgb_seq, fft_seq, label)`.
- Three-branch PyTorch model:
  - EfficientNet-B0 frame branch: 1280 features.
  - CNN frequency branch: 128 features.
  - BiLSTM temporal branch: 256 features.
  - Fusion classifier with auxiliary branch heads.
- Training, ablations, checkpoints, metrics, ROC, confusion matrix, and failure reports.
- Gradient heatmaps, FFT heatmaps, and optional SHAP explanations.

## Structure

```text
ai/models/          Model and dataset
ai/preprocessing/   Frames, faces, FFT, metadata
ai/inference/       Video score aggregation
ai/explain/         Heatmaps and branch explanations
scripts/            Preprocess, train, evaluate, explain
 tests/             Pytest tests
dataset/raw/        Input videos, not committed
dataset/processed/ Generated data, not committed
trained_models/     Checkpoints, not committed
reports/            Metrics and explanation outputs
```

## Setup

Windows PowerShell, Python 3.13:

```powershell
git clone <repository-url>
cd DeepFakeDetector
py -3.13 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m scripts.setup
```

After future updates:

```powershell
git pull
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Place videos here:

```text
dataset/raw/ffpp_real/*.mp4
dataset/raw/ffpp_fake/*.mp4
```

Real files use names such as `000.mp4`. Fake FaceForensics++ files use names such as `000_003.mp4`.

## Run Pipeline

1. Preprocess a small sample first:

```powershell
python -m scripts.preprocess --limit 5
```

This creates frames, cropped faces, FFT images, and `dataset/processed/metadata.csv`. Face detection can be slow on CPU.

2. Inspect data and run tests:

```powershell
python -m scripts.inspect_dataset
pytest -q tests
```

3. Train the full model:

```powershell
python -m scripts.train --epochs 10 --batch-size 4 --lr 1e-4 --branches full --seq-len 10 --run-name full_model
```

CPU smoke training:

```powershell
python -m scripts.train --epochs 2 --batch-size 4 --branches full --seq-len 10 --limit 5 --run-name cpu_smoke
```

4. Run ablations:

```powershell
python -m scripts.ablation --epochs 5 --batch-size 4 --seq-len 10 --limit 5
```

Configurations: `frame`, `frequency`, `frame+temporal`, and `full`.

5. Evaluate the held-out test split:

```powershell
python -m scripts.evaluate --run-name full_model
```

6. Explain one video:

```powershell
python -m scripts.explain --video dataset/raw/ffpp_real/000.mp4 --run-name full_model
```

## Outputs

```text
dataset/processed/metadata.csv
trained_models/<run>/best.pt
trained_models/<run>/last.pt
trained_models/<run>/config.json
reports/<run>/history.csv
reports/<run>/test_metrics.json
reports/<run>/roc_curve.png
reports/<run>/confusion_matrix.png
reports/<run>/failure_cases.json
reports/ablation_val.csv
reports/explanations/<video>/
```

Training uses only train/validation data. The test split is reserved for final evaluation. Runtime settings such as `FRAME_INTERVAL`, `IMG_SIZE`, and performance thresholds are in `ai/utils/config.py`.

## Current Limitations

- FastAPI backend and React frontend are not implemented.
- CPU RetinaFace processing is slow.
- SHAP uses a fallback attribution method when the optional package is unavailable.
- Evaluation metrics are unreliable when only a few test videos are available.
