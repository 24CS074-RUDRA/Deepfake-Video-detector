import numpy as np
import pandas as pd
from PIL import Image

from ai.models.dataset import DeepfakeSequenceDataset
from ai.preprocessing.fft_transform import face_to_fft


def test_no_video_id_appears_in_two_splits(tmp_path):
    metadata = pd.DataFrame(
        {
            "video_id": ["000", "000", "001", "001"],
            "split": ["train", "train", "test", "test"],
        }
    )
    split_counts = metadata.groupby("video_id")["split"].nunique()
    assert split_counts.max() == 1


def test_sequence_order_is_preserved(tmp_path):
    rows = []
    for frame_idx in (2, 0, 1):
        rgb_path = tmp_path / f"rgb_{frame_idx}.jpg"
        fft_path = tmp_path / f"fft_{frame_idx}.jpg"
        Image.new("RGB", (224, 224), (frame_idx, 0, 0)).save(rgb_path)
        Image.new("L", (224, 224), frame_idx).save(fft_path)
        rows.append(
            {
                "image_path": rgb_path.name,
                "fft_path": fft_path.name,
                "label": 0,
                "video_id": "video-a",
                "frame_idx": frame_idx,
                "split": "val",
            }
        )
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    dataset = DeepfakeSequenceDataset(
        metadata_path, tmp_path, sequence_length=3, split="val"
    )
    assert [row["frame_idx"] for row in dataset.videos[0]["rows"]] == [0, 1, 2]


def test_fft_output_shape_and_dtype():
    face = np.zeros((224, 224, 3), dtype=np.uint8)
    output = face_to_fft(face)
    assert output.shape == (224, 224)
    assert output.dtype == np.uint8
    assert output.min() >= 0
    assert output.max() <= 255
