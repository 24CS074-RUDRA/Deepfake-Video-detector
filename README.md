# Explainable Face-Swap Deepfake Detection

AI/ML-based face-swap deepfake detection project for the Smart India Hackathon (SIH) problem statement SIH 1683.

## Current Status

The repository currently contains a working, resumable data-preparation pipeline and a split-aware multimodal sequence dataset. The existing PyTorch detector remains the baseline RGB-plus-internal-FFT EfficientNet-B0/BiLSTM model.

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

Still to be implemented:

- Explicit RGB/FFT feature fusion in the detector forward pass.
- Audio extraction and lip-sync analysis.
- Grad-CAM and SHAP forensic explanations.
- FastAPI upload and prediction endpoints.
- React frontend.

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
    deepfake_detector.py       EfficientNet-B0/BiLSTM baseline model
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

```powershell
python -m scripts.train --limit 5
python -m scripts.evaluate --limit 5
```

Training uses the `train` split and validation uses the `val` split. Evaluation uses the held-out `test` split. DataLoaders use `num_workers=0` for Windows compatibility.

The sequence dataset returns:

```text
(rgb_seq, fft_seq, label)
```

where each sequence has `sequence_length` ordered frames. Augmentation is applied only to RGB images in the training split; FFT tensors are not spatially augmented.

The baseline training script currently passes the RGB sequence to the existing detector. Explicit consumption of the precomputed FFT sequence will be part of the next model-fusion step.

## Tests

```powershell
pytest -q tests
```

Current tests verify:

- A video ID does not occur in multiple splits.
- Metadata rows are loaded in ascending frame order.
- FFT output is `224x224` and `uint8`.

## Configuration

Shared paths and settings are defined in [ai/utils/config.py](ai/utils/config.py):

- `FRAME_INTERVAL`
- `IMG_SIZE`
- `RANDOM_SEED`
- raw, processed, model, report, and log directories


