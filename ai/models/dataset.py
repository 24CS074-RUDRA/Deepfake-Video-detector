import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class DeepfakeSequenceDataset(Dataset):
    """
    Dataset to load sequences of cropped faces from each video directory.
    Groups frames from the same video folder and samples a fixed sequence of frames.
    """
    def __init__(self, metadata_path, dataset_dir, sequence_length=10, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load metadata
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
            
        df = pd.read_csv(metadata_path)
        
        # Group by video_id (which is the parent folder of the frame image)
        df['video_id'] = df['image_path'].apply(lambda x: Path(x).parent.name)
        
        self.videos = []
        grouped = df.groupby(['label', 'video_id'])
        
        for (label, video_id), group in grouped:
            sorted_images = sorted(group['image_path'].tolist())
            if len(sorted_images) > 0:
                self.videos.append({
                    'label': label,
                    'video_id': video_id,
                    'image_paths': sorted_images
                })
                
        print(f"Loaded {len(self.videos)} videos from metadata.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        image_paths = video['image_paths']
        label = video['label']
        
        n_frames = len(image_paths)
        
        # Sample sequence_length frames uniformly
        indices = np.linspace(0, n_frames - 1, self.sequence_length, dtype=int)
        
        frames = []
        for i in indices:
            img_path = self.dataset_dir / image_paths[i]
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception as e:
                # Fallback: create a blank image if loading fails
                blank = Image.new('RGB', (224, 224), (0, 0, 0))
                if self.transform:
                    blank = self.transform(blank)
                frames.append(blank)
                
        # Stack frames: Shape (sequence_length, channels, height, width)
        frames_tensor = torch.stack(frames)
        
        # Label: Float for BCEWithLogitsLoss
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return frames_tensor, label_tensor
