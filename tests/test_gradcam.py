import numpy as np

from ai.explain.gradcam import fft_heatmap


def test_fft_heatmap_accepts_single_channel_image():
    heatmap = fft_heatmap(np.zeros((1, 224, 224), dtype=np.uint8))
    assert heatmap.shape == (224, 224, 3)