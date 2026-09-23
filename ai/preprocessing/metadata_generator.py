import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ai.utils.config import (
    FAKE_FACES,
    FAKE_FACES_FFT,
    RANDOM_SEED,
    REAL_FACES,
    REAL_FACES_FFT,
    PROCESSED_DATASET,
)


FRAME_INDEX_PATTERN = re.compile(r"frame_(\d+)")


def source_identity(video_id: str) -> str:
    """Return the source identity encoded by FF++ fake names when available."""
    return video_id.rsplit("_", 1)[-1]


def _split_groups(video_records: list[dict]) -> dict[str, str]:
    groups = pd.DataFrame(
        {
            "group_id": [record["group_id"] for record in video_records],
            "label": [record["label"] for record in video_records],
        }
    ).drop_duplicates("group_id")
    if len(groups) < 3 or groups["label"].nunique() < 2:
        return {group_id: "train" for group_id in groups["group_id"]}

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.15, random_state=RANDOM_SEED
    )
    train_indices, test_indices = next(
        splitter.split(groups["group_id"], groups["label"])
    )
    train_groups = groups.iloc[train_indices].reset_index(drop=True)
    test_groups = groups.iloc[test_indices]

    val_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=15 / 85, random_state=RANDOM_SEED
    )
    train_indices, val_indices = next(
        val_splitter.split(train_groups["group_id"], train_groups["label"])
    )
    split_map = {group_id: "test" for group_id in test_groups["group_id"]}
    split_map.update(
        {group_id: "val" for group_id in train_groups.iloc[val_indices]["group_id"]}
    )
    split_map.update(
        {group_id: "train" for group_id in train_groups.iloc[train_indices]["group_id"]}
    )
    return split_map


def _collect_videos(folder: Path, fft_folder: Path, label: int):
    records = []
    if not folder.exists():
        return records
    for video_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
        images = sorted(video_dir.glob("*.jpg"))
        valid_images = [
            image
            for image in images
            if (fft_folder / video_dir.name / image.name).exists()
        ]
        if not valid_images:
            continue
        video_id = video_dir.name
        records.append(
            {
                "video_id": video_id,
                "label": label,
                "group_id": source_identity(video_id),
                "images": valid_images,
                "fft_folder": fft_folder,
            }
        )
    return records


def generate_metadata(limit: int | None = None):
    real_videos = _collect_videos(REAL_FACES, REAL_FACES_FFT, 0)
    fake_videos = _collect_videos(FAKE_FACES, FAKE_FACES_FFT, 1)
    if limit is not None:
        real_videos = real_videos[:limit]
        fake_videos = fake_videos[:limit]
    videos = [*real_videos, *fake_videos]
    split_map = _split_groups(videos)
    rows = []
    for video in videos:
        for image in video["images"]:
            match = FRAME_INDEX_PATTERN.search(image.stem)
            frame_idx = int(match.group(1)) if match else len(rows)
            fft_path = video["fft_folder"] / video["video_id"] / image.name
            rows.append(
                {
                    "image_path": str(image.relative_to(PROCESSED_DATASET)),
                    "fft_path": str(fft_path.relative_to(PROCESSED_DATASET)),
                    "label": video["label"],
                    "video_id": video["video_id"],
                    "frame_idx": frame_idx,
                    "split": split_map.get(video["group_id"], "train"),
                }
            )

    columns = ["image_path", "fft_path", "label", "video_id", "frame_idx", "split"]
    metadata = pd.DataFrame(rows, columns=columns)
    if not metadata.empty:
        metadata = metadata.sort_values(["video_id", "frame_idx"])
    save_path = PROCESSED_DATASET / "metadata.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(save_path, index=False)
    print(f"Metadata saved to: {save_path}")
    print(f"Total frames: {len(metadata)}")
    if not metadata.empty:
        print(
            "Videos by split: "
            f"{metadata.drop_duplicates('video_id')['split'].value_counts().to_dict()}"
        )
    return metadata
