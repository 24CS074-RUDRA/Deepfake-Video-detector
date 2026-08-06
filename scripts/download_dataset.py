"""
Dataset Downloader

Downloads the FaceForensics dataset from Google Drive,
extracts it automatically,
verifies the dataset,
and removes temporary files.

Run:
    python -m scripts.download_dataset
"""

from pathlib import Path, PurePosixPath
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

    RAW_DATASET.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:

        for member in zip_ref.infolist():
            member_path = PurePosixPath(member.filename)
            parts = member_path.parts

            # The archive stores the videos below a wrapper directory (for
            # example, ``face++dataset/ffpp_fake/...``).  Strip that wrapper
            # so the two class folders live directly in ``dataset/raw``.
            try:
                dataset_index = next(
                    index
                    for index, part in enumerate(parts)
                    if part in {"ffpp_real", "ffpp_fake"}
                )
            except StopIteration:
                continue

            relative_path = parts[dataset_index:]

            # Only extract normal, relative paths from the expected folders.
            if any(part in {"", ".", ".."} for part in relative_path):
                raise ValueError(f"Unsafe archive path: {member.filename}")

            destination = RAW_DATASET.joinpath(*relative_path)

            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)


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
