import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FFTFeatureExtractor(nn.Module):
    """
    Computes the 2D FFT magnitude spectrum of grayscale face frames,
    resizes it to a fixed size, flattens, and projects to a lower-dimensional feature vector.
    """
    def __init__(self, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        
        # Projection MLP
        self.fc = nn.Sequential(
            nn.Linear(64 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        # Input shape: (N, 3, H, W)
        # Convert RGB to Grayscale
        gray = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]
        
        # Compute 2D Fast Fourier Transform
        fft_res = torch.fft.fft2(gray)
        
        # Shift the zero-frequency component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_res, dim=(-2, -1))
        
        # Compute the magnitude spectrum (log scale for numerical stability)
        magnitude = torch.log(1.0 + torch.abs(fft_shifted) + 1e-8)
        
        # Resize to fixed shape (64, 64) for classification
        magnitude_unsqueezed = magnitude.unsqueeze(1) # Shape: (N, 1, H, W)
        magnitude_resized = F.interpolate(
            magnitude_unsqueezed, 
            size=(64, 64), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Flatten and project
        flat_magnitude = magnitude_resized.view(magnitude_resized.size(0), -1)
        features = self.fc(flat_magnitude)
        
        return features


class DeepfakeDetector(nn.Module):
    """
    Spatiotemporal Deepfake Detector combining:
      - EfficientNet-B0 (Spatial Features)
      - FFT Magnitude Spectrum (Frequency Features)
      - BiLSTM (Temporal Sequence Classification)
    """
    def __init__(self, sequence_length=10, lstm_hidden_dim=128, fft_feature_dim=128):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # 1. Spatial Branch: EfficientNet-B0
        try:
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT
        except ImportError:
            weights = None
            
        if weights is not None:
            self.efficientnet = models.efficientnet_b0(weights=weights)
        else:
            self.efficientnet = models.efficientnet_b0(pretrained=True)
            
        # Remove original classifier layer to only extract feature map
        # EfficientNet-B0 features has 1280 channels at the output of average pool
        self.spatial_dim = 1280
        
        # 2. Frequency Branch: FFT Feature Extractor
        self.fft_extractor = FFTFeatureExtractor(output_dim=fft_feature_dim)
        
        # 3. Fusion and Temporal Modeling: BiLSTM
        # Input features dim: spatial_dim + fft_feature_dim
        input_dim = self.spatial_dim + fft_feature_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if lstm_hidden_dim > 1 else 0.0
        )
        
        # 4. Classification Head
        # Output features dim from BiLSTM is lstm_hidden_dim * 2 (since bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Input shape x: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        
        # Flatten batch and sequence dimensions to extract frame-level features
        x_flat = x.view(batch_size * seq_len, channels, height, width)
        
        # Spatial branch: shape (batch_size * seq_len, 1280)
        spatial_features = self.efficientnet.features(x_flat)
        spatial_features = self.efficientnet.avgpool(spatial_features)
        spatial_features = torch.flatten(spatial_features, 1)
        
        # Frequency branch: shape (batch_size * seq_len, fft_feature_dim)
        freq_features = self.fft_extractor(x_flat)
        
        # Concatenate spatial and frequency representations: shape (batch_size * seq_len, 1408)
        combined_features = torch.cat([spatial_features, freq_features], dim=1)
        
        # Reshape back to sequence form: shape (batch_size, sequence_length, 1408)
        sequence_features = combined_features.view(batch_size, seq_len, -1)
        
        # Temporal sequence modeling using BiLSTM
        lstm_out, _ = self.lstm(sequence_features) # Shape: (batch_size, sequence_length, lstm_hidden_dim * 2)
        
        # Average pooling over the sequence time dimension
        pooled_out = torch.mean(lstm_out, dim=1) # Shape: (batch_size, lstm_hidden_dim * 2)
        
        # Binary Classification Logits
        logits = self.classifier(pooled_out).squeeze(-1) # Shape: (batch_size,)
        
        return logits
