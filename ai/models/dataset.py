from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ai.utils.config import IMG_SIZE


class JPEGCompression:
    def __init__(self, quality_range=(45, 95)):
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(
            ".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


class DeepfakeSequenceDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        dataset_dir,
        sequence_length=10,
        split="train",
        transform=None,
        fft_transform=None,
        limit=None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        self.split = split
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

        metadata = pd.read_csv(metadata_path)
        required_columns = {
            "image_path", "fft_path", "label", "video_id", "frame_idx", "split"
        }
        missing = required_columns.difference(metadata.columns)
        if missing:
            raise ValueError(f"Metadata missing columns: {sorted(missing)}")
        metadata = metadata[metadata["split"] == split]

        self.transform = transform or transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.fft_transform = fft_transform or transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.RandomApply([JPEGCompression()], p=0.3),
        ]) if split == "train" else None

        self.videos = []
        for (label, video_id), group in metadata.groupby(
            ["label", "video_id"], sort=True
        ):
            ordered = group.sort_values("frame_idx")
            self.videos.append({
                "label": int(label),
                "video_id": video_id,
                "rows": ordered[["image_path", "fft_path", "frame_idx"]].to_dict("records"),
            })
        if limit is not None:
            self.videos = self.videos[:limit]

    def __len__(self):
        return len(self.videos)

    def _sample_rows(self, rows):
        indices = np.linspace(0, len(rows) - 1, self.sequence_length, dtype=int)
        return [rows[index] for index in indices]

    def __getitem__(self, idx):
        video = self.videos[idx]
        rgb_frames = []
        fft_frames = []
        for row in self._sample_rows(video["rows"]):
            rgb_path = self.dataset_dir / row["image_path"]
            fft_path = self.dataset_dir / row["fft_path"]
            try:
                rgb_image = Image.open(rgb_path).convert("RGB")
                fft_image = Image.open(fft_path).convert("L")
                if self.augmentation is not None:
                    rgb_image = self.augmentation(rgb_image)
                rgb_frames.append(self.transform(rgb_image))
                fft_frames.append(self.fft_transform(fft_image))
            except (OSError, ValueError):
                blank_rgb = Image.new("RGB", IMG_SIZE, (0, 0, 0))
                blank_fft = Image.new("L", IMG_SIZE, 0)
                rgb_frames.append(self.transform(blank_rgb))
                fft_frames.append(self.fft_transform(blank_fft))

        return (
            torch.stack(rgb_frames),
            torch.stack(fft_frames),
            torch.tensor(video["label"], dtype=torch.float32),
        )
