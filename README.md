# Explainable Face-Swap Deepfake Detection

AI/ML-based face-swap deepfake detection project for the Smart India Hackathon (SIH) problem statement SIH 1683.

## Current Status

The repository currently contains a working, resumable data-preparation pipeline, a split-aware multimodal sequence dataset, and a configurable three-branch PyTorch detector.

Implemented:

- Video frame extraction with a shared frame interval.
- RetinaFace face detection and cropping.
- Largest-face selection for frames containing multiple faces.
- Missing-face and detection-failure logging.
- Per-face FFT magnitude image generation.
- Video-level train/validation/test metadata splits.
- Source-identity grouping for FaceForensics++ fake filenames.
- RGB and FFT sequence loading with frame-order preservation.
- RGB-only training augmentation.
- Dataset inspection reports.
- Focused pytest coverage for splits, ordering, and FFT dimensions.
- Explicit frame, frequency, temporal, and fusion branches with auxiliary heads.
- Configurable branch ablations, weighted loss, AdamW, AMP, clipping, checkpoints, and early stopping.
- Per-run training history and metric curves.

Still to be implemented:

- Audio extraction and lip-sync analysis.
- FastAPI upload and prediction endpoints.
- React frontend.

Implemented for evaluation and explanation:

- Validation-selected threshold evaluation on the held-out test split.
- Video score aggregation using mean, top-k mean, median, and majority vote.
- ROC, confusion matrix, accuracy, precision, recall, F1, AUC, and EER reports.
- Branch-level test ablation reports and failure-case JSON.
- Grad-CAM-compatible frame overlays with FFT heatmaps.
- SHAP fusion-head attribution when SHAP is installed, with an embedding fallback otherwise.

## Technology Stack

- Python 3.13
- PyTorch and TorchVision
- OpenCV
- RetinaFace
- EfficientNet-B0 and BiLSTM baseline
- NumPy FFT
- Pandas and scikit-learn
- FastAPI and React planned for Part B

## Project Structure

```text
ai/
  models/
    dataset.py                 Split-aware RGB/FFT sequence dataset
    deepfake_detector.py       Three-branch EfficientNet/CNN/BiLSTM model
  preprocessing/
    frame_extractor.py         Video-to-frame extraction
    face_detector.py           RetinaFace cropping and logging
    fft_transform.py           FFT image generation
    metadata_generator.py      Paired metadata and dataset splits
    preprocess_pipeline.py     Preprocessing orchestration
  utils/
    config.py                  Shared paths and preprocessing settings
scripts/
  preprocess.py                Run the preprocessing pipeline
  inspect_dataset.py           Generate dataset reports
  train.py                     Train the baseline model
  ablation.py                  Run frame/frequency/temporal/fusion ablations
  evaluate.py                  Evaluate the held-out test split
  predict.py                   Predict a single video
  download_dataset.py          Download and extract the dataset
tests/
  test_data_pipeline.py        Data-pipeline tests
dataset/
  raw/                         Input videos; do not commit
  processed/                   Frames, faces, FFT images, metadata
trained_models/                Model checkpoints; do not commit
reports/                       Generated CSV files and figures
logs/                          Runtime logs
```

## Installation

Create and activate a Python 3.13 virtual environment on Windows:

```powershell
py -3.13 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The repository also contains a setup check:

```powershell
python -m scripts.setup
```

## Run After Pulling From GitHub

From a fresh clone or after pulling new changes, run these commands in PowerShell from the repository root:

1. Update the repository and enter the project directory:

```powershell
git pull
cd C:\DeepFakeDetector
```

2. Create and activate the virtual environment. Skip the first command if `venv` already exists:

```powershell
py -3.13 -m venv venv
venv\Scripts\activate
```

3. Install or update dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the project setup check:

```powershell
python -m scripts.setup
```

5. Place the dataset in the expected folders:

```text
dataset/raw/ffpp_real/*.mp4
dataset/raw/ffpp_fake/*.mp4
```

Do not commit datasets or model checkpoints. If dataset download settings are configured in `.env`, the downloader can be used with:

```powershell
python -m scripts.download_dataset
```

6. Run a small preprocessing check first:

```powershell
python -m scripts.preprocess --limit 5
```

This creates frames, cropped faces, FFT images, and `dataset/processed/metadata.csv`. Face detection can be slow on CPU, especially during the first RetinaFace model initialization.

7. Generate inspection reports and run tests:

```powershell
python -m scripts.inspect_dataset
pytest -q tests
```

8. Train the full model or run the CPU smoke configuration:

```powershell
python -m scripts.train --epochs 10 --batch-size 4 --lr 1e-4 --branches full --seq-len 10 --run-name full_model
python -m scripts.train --epochs 2 --batch-size 4 --branches full --seq-len 10 --limit 5 --run-name cpu_smoke
```

9. Run branch ablations and evaluate a saved run:

```powershell
python -m scripts.ablation --epochs 5 --batch-size 4 --seq-len 10 --limit 5
python -m scripts.evaluate --run-name full_model --limit 5
```

Expected outputs include:

- `dataset/processed/metadata.csv`
- `trained_models/<run-name>/best.pt`
- `trained_models/<run-name>/last.pt`
- `trained_models/<run-name>/config.json`
- `reports/<run-name>/history.csv`
- `reports/ablation_val.csv`

## Dataset Layout

Place videos in:

```text
dataset/raw/ffpp_real/*.mp4
dataset/raw/ffpp_fake/*.mp4
```

The available real videos use numeric names such as `000.mp4`. The fake videos use the FaceForensics++-style form `<target>_<source>.mp4`, such as `000_003.mp4`. Metadata groups fake videos by the source token and real videos by their numeric identity so related videos remain in one split.

Datasets and trained weights are intentionally excluded from version control.

## Preprocessing

Run the complete pipeline:

```powershell
python -m scripts.preprocess
```

Run a quick smoke test on up to five videos per class:

```powershell
python -m scripts.preprocess --limit 5
```

The pipeline performs:

1. Frame extraction using `FRAME_INTERVAL = 30`.
2. Face detection and resizing to `IMG_SIZE = (224, 224)`.
3. FFT magnitude generation at full `224x224` resolution.
4. Metadata generation with `image_path`, `fft_path`, `label`, `video_id`, `frame_idx`, and `split`.

Processing is resumable. Existing frame, face, and FFT files are skipped.

Face detection can also be run directly:

```powershell
python -m ai.preprocessing.face_detector --limit 5
```

Face-detection warnings and failures are written to `logs/face_detector.log`.

## Dataset Inspection

After preprocessing, generate reports with:

```powershell
python -m scripts.inspect_dataset
```

Generated files in `reports/`:

- `class_balance_by_split.csv`
- `faces_per_video_histogram.png`
- `face_fft_samples.png`

## Training and Evaluation

Run preprocessing before training or ablations. The training commands require `dataset/processed/metadata.csv`, paired FFT images, and train/val split assignments:

```powershell
python -m scripts.preprocess --limit 5
```

```powershell
python -m scripts.train --epochs 10 --batch-size 4 --lr 1e-4 --branches full --seq-len 10 --run-name full_model
python -m scripts.train --epochs 2 --batch-size 4 --branches full --seq-len 10 --limit 5 --run-name cpu_smoke
python -m scripts.ablation --epochs 5 --batch-size 4 --seq-len 10 --limit 5
python -m scripts.evaluate --run-name full_model --limit 5
```

Training uses only the `train` and `val` splits. The test split is reserved for final evaluation. DataLoaders use `num_workers=0` for Windows compatibility.

Supported `--branches` values are `frame`, `frequency`, `frame+temporal`, and `full`. Training freezes the EfficientNet backbone for the first two epochs, then unfreezes it with a lower learning rate. It uses class-weighted BCE, AdamW, a plateau scheduler, CUDA mixed precision when available, gradient clipping, validation-AUC early stopping, and saves `best.pt`, `last.pt`, and `config.json` under `trained_models/<run-name>/`.

Training history and curves are saved under `reports/<run-name>/`. The ablation command writes `reports/ablation_val.csv` and prints a recommendation based on validation AUC.

Evaluation writes test artifacts to `reports/<run-name>/`, including `test_metrics.json`, `test_branch_ablation.csv`, `aggregation_val.csv`, `roc_curve.png`, `confusion_matrix.png`, and `failure_cases.json`. It also copies the evaluated checkpoint to `trained_models/final/best.pth` with its configuration.

Run explanations for a video with:

```powershell
python -m scripts.explain --video dataset/raw/ffpp_real/000.mp4 --run-name ablation_full
```

Explanation outputs are saved under `reports/explanations/<video>/`.

The sequence dataset returns:

```text
(rgb_seq, fft_seq, label)
```

where each sequence has `sequence_length` ordered frames. Augmentation is applied only to RGB images in the training split; FFT tensors are not spatially augmented.

The detector returns the fused logit, per-frame logits, branch logits, and frame/frequency/temporal/fusion embeddings for later Grad-CAM and SHAP work.

## Tests

```powershell
pytest -q tests
```

Current tests verify:

- A video ID does not occur in multiple splits.
- Metadata rows are loaded in ascending frame order.
- FFT output is `224x224` and `uint8`.
- All detector branch configurations return the expected tensor shapes.
- The frequency branch can overfit a one-batch CPU smoke test.

## Configuration

Shared paths and settings are defined in [ai/utils/config.py](ai/utils/config.py):

- `FRAME_INTERVAL`
- `IMG_SIZE`
- `RANDOM_SEED`
- raw, processed, model, report, and log directories


