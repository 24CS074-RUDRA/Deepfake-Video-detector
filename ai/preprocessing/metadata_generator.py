from pathlib import Path
import pandas as pd

from ai.utils.config import PROCESSED_DATASET


def generate_metadata():

    rows = []

    datasets = [
        ("faces_real", 0),
        ("faces_fake", 1)
    ]

    for folder_name, label in datasets:

        folder = PROCESSED_DATASET / folder_name

        if not folder.exists():
            continue

        for image in folder.rglob("*.jpg"):

            rows.append({
                "image_path": str(
                    image.relative_to(PROCESSED_DATASET)
                ),
                "label": label
            })

    df = pd.DataFrame(rows)

    save_path = PROCESSED_DATASET / "metadata.csv"

    df.to_csv(save_path, index=False)

    print(f"\nMetadata saved to:\n{save_path}")

    print(f"Total Images : {len(df)}")