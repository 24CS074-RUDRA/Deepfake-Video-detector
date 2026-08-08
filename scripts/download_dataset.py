from pathlib import Path
import zipfile
import shutil
import gdown

from ai.utils.config import (
    ROOT_DIR,
    RAW_DATASET,
    DATASET_NAME,
    DATASET_URL
)

TEMP_DIR = ROOT_DIR / "temp"
ZIP_FILE = TEMP_DIR / DATASET_NAME


def dataset_exists():
    return (
        (RAW_DATASET / "ffpp_real").exists()
        and
        (RAW_DATASET / "ffpp_fake").exists()
    )


def download_dataset():

    print("\nDownloading dataset...")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    gdown.download(
        DATASET_URL,
        str(ZIP_FILE),
        quiet=False
    )


def extract_dataset():

    print("\nExtracting dataset...")

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

    print("\nVerifying dataset...")

    if dataset_exists():

        print("✅ Dataset verified successfully.")

    else:

        raise Exception(
            "Dataset verification failed."
        )


def cleanup():

    if ZIP_FILE.exists():

        ZIP_FILE.unlink()

    if TEMP_DIR.exists():

        shutil.rmtree(TEMP_DIR)

    print("Temporary files removed.")


def main():

    print("=" * 60)
    print("Dataset Downloader")
    print("=" * 60)

    if dataset_exists():

        print("✅ Dataset already exists.")

        return

    download_dataset()

    extract_dataset()

    verify_dataset()

    cleanup()

    print("\n🎉 Dataset is ready!")


if __name__ == "__main__":

    main()