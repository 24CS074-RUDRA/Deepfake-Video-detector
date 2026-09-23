"""
Dataset Downloader

Downloads the FaceForensics dataset from Google Drive,
extracts it automatically,
verifies the dataset,
and removes temporary files.

Run:
    python -m scripts.download_dataset
"""

from pathlib import Path
import zipfile
import shutil
import gdown

from ai.utils.config import (
    ROOT_DIR,
    RAW_DATASET,
    DATASET_NAME,
    DATASET_FILE_ID
)

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

TEMP_DIR = ROOT_DIR / "temp"
ZIP_FILE = TEMP_DIR / (DATASET_NAME or "dataset.zip")

REAL_FOLDER = RAW_DATASET / "ffpp_real"
FAKE_FOLDER = RAW_DATASET / "ffpp_fake"


# ----------------------------------------------------------
# Check Dataset
# ----------------------------------------------------------

def dataset_exists():

    return REAL_FOLDER.exists() and FAKE_FOLDER.exists()


# ----------------------------------------------------------
# Download Dataset
# ----------------------------------------------------------

def download_dataset():

    print("\nDownloading dataset from Google Drive...\n")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    gdown.download(
        id=DATASET_FILE_ID,
        output=str(ZIP_FILE),
        quiet=False
    )


# ----------------------------------------------------------
# Extract Dataset
# ----------------------------------------------------------

def extract_dataset():

    print("\nExtracting dataset...\n")

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(RAW_DATASET)

    nested_dir = RAW_DATASET / "face++dataset"
    if nested_dir.exists():
        for item in nested_dir.iterdir():
            dest = RAW_DATASET / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        shutil.rmtree(nested_dir)


def verify_dataset():

    print("\nVerifying dataset...\n")

    if dataset_exists():

        print("✅ Dataset verified successfully.")

        print(f"\nReal Videos : {REAL_FOLDER}")

        print(f"Fake Videos : {FAKE_FOLDER}")

    else:

        raise Exception(
            "Dataset verification failed.\n"
            "Required folders 'ffpp_real' and 'ffpp_fake' were not found."
        )


# ----------------------------------------------------------
# Cleanup
# ----------------------------------------------------------

def cleanup():

    if ZIP_FILE.exists():

        ZIP_FILE.unlink()

    if TEMP_DIR.exists():

        shutil.rmtree(TEMP_DIR)

    print("\n🗑 Temporary files removed.")


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------

def main():

    print("=" * 60)
    print(" DeepFake Detector - Dataset Downloader ")
    print("=" * 60)

    if dataset_exists():

        print("\n✅ Dataset already exists.")
        return

    download_dataset()

    extract_dataset()

    verify_dataset()

    cleanup()

    print("\n🎉 Dataset is ready for preprocessing!")


if __name__ == "__main__":

    main()