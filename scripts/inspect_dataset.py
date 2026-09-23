import argparse

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from ai.utils.config import PROCESSED_DATASET, REPORT_DIR


def inspect_dataset(metadata_path=PROCESSED_DATASET / "metadata.csv"):
    metadata = pd.read_csv(metadata_path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    balance = (
        metadata.groupby(["split", "label"])
        .size()
        .rename("frames")
        .reset_index()
    )
    balance.to_csv(REPORT_DIR / "class_balance_by_split.csv", index=False)

    faces_per_video = metadata.groupby(["split", "video_id"]).size()
    figure, axis = plt.subplots(figsize=(8, 5))
    for split, values in faces_per_video.groupby(level=0):
        axis.hist(values.to_numpy(), bins=20, alpha=0.6, label=split)
    axis.set_xlabel("Faces per video")
    axis.set_ylabel("Videos")
    axis.set_title("Faces per video by split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "faces_per_video_histogram.png", dpi=150)
    plt.close(figure)

    samples = []
    for label in (0, 1):
        subset = metadata[metadata["label"] == label].drop_duplicates("video_id").head(4)
        samples.extend(subset.to_dict("records"))

    figure, axes = plt.subplots(2, 8, figsize=(16, 6))
    for index, row in enumerate(samples):
        row_index = 0 if row["label"] == 0 else 1
        column = (index % 4) * 2
        image = Image.open(PROCESSED_DATASET / row["image_path"]).convert("RGB")
        fft_image = Image.open(PROCESSED_DATASET / row["fft_path"]).convert("L")
        axes[row_index, column].imshow(image)
        axes[row_index, column].set_title(
            f"{'Real' if row['label'] == 0 else 'Fake'} {row['video_id']}"
        )
        axes[row_index, column].axis("off")
        axes[row_index, column].set_xlabel("RGB")
        axes[row_index, column + 1].imshow(fft_image, cmap="gray")
        axes[row_index, column + 1].axis("off")
        axes[row_index, column + 1].set_xlabel("FFT")

    figure.suptitle("Face samples and FFT magnitude images")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "face_fft_samples.png", dpi=150)
    plt.close(figure)
    print(f"Reports saved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect processed deepfake data.")
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(PROCESSED_DATASET / "metadata.csv"),
    )
    inspect_dataset(parser.parse_args().metadata)
