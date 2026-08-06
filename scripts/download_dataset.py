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
ZIP_FILE = TEMP_DIR / DATASET_NAME

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

    # Check if an extra folder exists
    extracted_folders = [
        folder for folder in RAW_DATASET.iterdir()
        if folder.is_dir()
    ]

    # If there is only one folder (e.g. faceforensics++)
    # move its contents to dataset/raw
    if len(extracted_folders) == 1:

        parent = extracted_folders[0]

        if parent.name not in ["ffpp_real", "ffpp_fake"]:

            print(f"Found parent folder: {parent.name}")

            for item in parent.iterdir():

                destination = RAW_DATASET / item.name

                if destination.exists():
                    shutil.rmtree(destination)

                shutil.move(str(item), str(destination))

            parent.rmdir()

            print("Dataset organized successfully.")


# ----------------------------------------------------------
# Verify Dataset
# ----------------------------------------------------------

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