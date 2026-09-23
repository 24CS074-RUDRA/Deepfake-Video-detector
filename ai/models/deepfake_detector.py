from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class FrameBranch(nn.Module):
    output_dim = 1280

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            except AttributeError:
                weights = None
        try:
            backbone = models.efficientnet_b0(weights=weights)
        except Exception:
            backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.head = nn.Linear(self.output_dim, 1)

    def forward(self, frames: torch.Tensor):
        features = self.pool(self.features(frames)).flatten(1)
        return features, self.head(features).squeeze(-1)

    def freeze_backbone(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = True


class FrequencyBranch(nn.Module):
    output_dim = 128

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, self.output_dim)
        self.head = nn.Linear(self.output_dim, 1)

    def forward(self, fft_images: torch.Tensor):
        features = self.projection(self.encoder(fft_images).flatten(1))
        return features, self.head(features).squeeze(-1)


class TemporalBranch(nn.Module):
    output_dim = 256

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(self.output_dim, 1)

    def forward(self, sequence: torch.Tensor):
        outputs, _ = self.lstm(sequence)
        embedding = outputs.mean(dim=1)
        return embedding, self.head(embedding).squeeze(-1)


class DeepfakeDetector(nn.Module):
    """Three-branch face-swap detector with branch ablation flags."""

    def __init__(
        self,
        sequence_length: int = 10,
        use_frame: bool = True,
        use_freq: bool = True,
        use_temporal: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        if not any((use_frame, use_freq, use_temporal)):
            raise ValueError("At least one detector branch must be enabled")
        if use_temporal and not (use_frame or use_freq):
            raise ValueError("Temporal branch requires frame or frequency features")

        self.sequence_length = sequence_length
        self.use_frame = use_frame
        self.use_freq = use_freq
        self.use_temporal = use_temporal
        self.frame_branch = FrameBranch(pretrained=pretrained) if use_frame else None
        self.freq_branch = FrequencyBranch() if use_freq else None

        frame_dim = FrameBranch.output_dim if use_frame else 0
        freq_dim = FrequencyBranch.output_dim if use_freq else 0
        temporal_input_dim = frame_dim + freq_dim
        self.temporal_branch = (
            TemporalBranch(temporal_input_dim) if use_temporal else None
        )

        fusion_dim = 0
        if use_frame:
            fusion_dim += frame_dim
        if use_freq:
            fusion_dim += freq_dim
        if use_temporal:
            fusion_dim += TemporalBranch.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, rgb_seq: torch.Tensor, fft_seq: torch.Tensor):
        batch_size, seq_len = rgb_seq.shape[:2]
        rgb_flat = rgb_seq.reshape(batch_size * seq_len, *rgb_seq.shape[2:])
        fft_flat = fft_seq.reshape(batch_size * seq_len, *fft_seq.shape[2:])

        frame_features = None
        freq_features = None
        branch_logits = {}
        per_frame_logits = {}

        if self.frame_branch is not None:
            frame_features, frame_logits = self.frame_branch(rgb_flat)
            frame_features = frame_features.view(batch_size, seq_len, -1)
            per_frame_logits["frame"] = frame_logits.view(batch_size, seq_len)
            branch_logits["frame"] = frame_logits.view(batch_size, seq_len).mean(dim=1)

        if self.freq_branch is not None:
            freq_features, freq_logits = self.freq_branch(fft_flat)
            freq_features = freq_features.view(batch_size, seq_len, -1)
            per_frame_logits["freq"] = freq_logits.view(batch_size, seq_len)
            branch_logits["freq"] = freq_logits.view(batch_size, seq_len).mean(dim=1)

        temporal_features = None
        if self.temporal_branch is not None:
            sequence_features = torch.cat(
                [feature for feature in (frame_features, freq_features) if feature is not None],
                dim=-1,
            )
            temporal_features, temporal_logits = self.temporal_branch(sequence_features)
            branch_logits["temporal"] = temporal_logits

        fusion_features = [
            feature.mean(dim=1)
            for feature in (frame_features, freq_features)
            if feature is not None
        ]
        if temporal_features is not None:
            fusion_features.append(temporal_features)
        fused = torch.cat(fusion_features, dim=-1)
        logit = self.fusion(fused).squeeze(-1)

        return {
            "logit": logit,
            "per_frame_logits": per_frame_logits,
            "branch_logits": branch_logits,
            "branch_embeddings": {
                "frame": None if frame_features is None else frame_features,
                "frequency": None if freq_features is None else freq_features,
                "temporal": temporal_features,
                "fusion": fused,
            },
        }

    def freeze_frame_backbone(self):
        if self.frame_branch is not None:
            self.frame_branch.freeze_backbone()

    def unfreeze_frame_backbone(self):
        if self.frame_branch is not None:
            self.frame_branch.unfreeze_backbone()
