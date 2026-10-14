from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _normalize(values):
    values = values.astype(np.float32)
    values -= values.min()
    maximum = values.max()
    return values / maximum if maximum > 0 else values


def fft_heatmap(fft_image):
    array = np.asarray(fft_image.convert("L") if hasattr(fft_image, "convert") else fft_image)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"FFT heatmap expects a 2D image, got shape {array.shape}")
    heatmap = cv2.applyColorMap((_normalize(array) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap


def overlay_heatmap(face, heatmap, alpha=0.45):
    image = np.asarray(face.convert("RGB") if hasattr(face, "convert") else face)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def _gradient_cam(model, rgb_frame, fft_frame, branch="frame"):
    model.zero_grad(set_to_none=True)
    rgb = rgb_frame.unsqueeze(0).unsqueeze(0)
    fft = fft_frame.unsqueeze(0).unsqueeze(0)
    rgb.requires_grad_(branch == "frame")
    fft.requires_grad_(branch in {"frequency", "freq"})
    output = model(rgb, fft)
    score = output["branch_logits"].get(branch, output["logit"]).sum()
    score.backward()
    source = rgb.grad if branch == "frame" else fft.grad
    if source is None:
        source = rgb if branch == "frame" else fft
    heatmap = source.detach().abs().mean(dim=tuple(range(source.ndim - 2))).cpu().numpy()
    return cv2.applyColorMap((_normalize(heatmap) * 255).astype(np.uint8), cv2.COLORMAP_JET)


def generate_gradcam_overlays(
    model,
    faces,
    fft_images,
    scores,
    output_dir,
    top_n=4,
    device="cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = np.argsort(np.asarray(scores))[-top_n:][::-1]
    model.eval()
    overlays = []
    for index in indices:
        face = faces[index]
        fft_image = fft_images[index]
        try:
            heatmap = _gradient_cam(
                model,
                face.to(device),
                fft_image.to(device),
                branch="frame" if model.use_frame else "freq",
            )
        except RuntimeError:
            heatmap = fft_heatmap(fft_image.cpu().numpy()[0])
        face_array = face.cpu().permute(1, 2, 0).numpy()
        face_array = np.clip((face_array * 255), 0, 255).astype(np.uint8)
        overlay = overlay_heatmap(face_array, heatmap)
        path = output_dir / f"frame_{int(index):04d}_gradcam.jpg"
        cv2.imwrite(str(path), overlay)
        overlays.append({"frame_index": int(index), "score": float(scores[index]), "path": str(path)})
    return overlays
