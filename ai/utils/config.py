from pathlib import Path

# Root Project
ROOT_DIR = Path(__file__).resolve().parents[2]

# Dataset Paths
RAW_DATASET = ROOT_DIR / "dataset" / "raw"
PROCESSED_DATASET = ROOT_DIR / "dataset" / "processed"

# Raw Videos
REAL_VIDEOS = RAW_DATASET / "ffpp_real"
FAKE_VIDEOS = RAW_DATASET / "ffpp_fake"

# Extracted Frames
REAL_FRAMES = PROCESSED_DATASET / "frames_real"
FAKE_FRAMES = PROCESSED_DATASET / "frames_fake"

# Cropped Faces
REAL_FACES = PROCESSED_DATASET / "faces_real"
FAKE_FACES = PROCESSED_DATASET / "faces_fake"

# Models
MODEL_DIR = ROOT_DIR / "trained_models"

# Reports
REPORT_DIR = ROOT_DIR / "reports"

from dotenv import load_dotenv
import os

load_dotenv()

# --------------------------------------------------
# Dataset Download Configuration
# --------------------------------------------------

DATASET_NAME = os.getenv("DATASET_NAME")

DATASET_FILE_ID = os.getenv("DATASET_FILE_ID")