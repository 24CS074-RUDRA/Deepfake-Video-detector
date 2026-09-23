from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from retinaface import RetinaFace
from torchvision import transforms

from ai.explain.gradcam import generate_gradcam_overlays
from ai.explain.shap_explainer import explain_branch_embeddings, suspicion_timeline
from ai.models.deepfake_detector import DeepfakeDetector
from ai.utils.config import IMG_SIZE, MODEL_DIR, REPORT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(run_name):
    checkpoint = torch.load(MODEL_DIR / run_name / "best.pt", map_location=DEVICE)
    config = checkpoint.get("config", {})
    branches = config.get("branches", "full")
    flags = {
        "frame": (True, False, False),
        "frequency": (False, True, False),
        "frame+temporal": (True, False, True),
        "full": (True, True, True),
    }.get(branches, (True, True, True))
    model = DeepfakeDetector(
        sequence_length=int(config.get("seq_len", 10)),
        use_frame=flags[0],
        use_freq=flags[1],
        use_temporal=flags[2],
        pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(config.get("seq_len", 10))


def extract_faces(video_path, limit=30):
    capture = cv2.VideoCapture(str(video_path))
    faces = []
    frame_index = 0
    while len(faces) < limit:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % 30:
            frame_index += 1
            continue
        temporary = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = RetinaFace.extract_faces(
            img_path=temporary, target_size=IMG_SIZE, min_max_norm=False
        )
        if detections:
            faces.append(Image.fromarray(detections[0].astype(np.uint8)))
        frame_index += 1
    capture.release()
    return faces


def run(video, run_name):
    model, sequence_length = load_model(run_name)
    faces = extract_faces(video, limit=max(sequence_length, 20))
    if not faces:
        raise SystemExit("No faces detected in the supplied video")
    indices = np.linspace(0, len(faces) - 1, sequence_length, dtype=int)
    faces = [faces[index] for index in indices]
    rgb_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    fft_transform = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
    rgb = torch.stack([rgb_transform(face) for face in faces])
    fft = []
    for face in faces:
        array = cv2.cvtColor(np.asarray(face), cv2.COLOR_RGB2GRAY)
        spectrum = np.fft.fftshift(np.fft.fft2(cv2.resize(array, IMG_SIZE)))
        magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        fft.append(fft_transform(Image.fromarray(magnitude)))
    fft = torch.stack(fft)

    with torch.no_grad():
        outputs = model(rgb.unsqueeze(0).to(DEVICE), fft.unsqueeze(0).to(DEVICE))
    output_dir = REPORT_DIR / "explanations" / Path(video).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_scores = next(iter(outputs["per_frame_logits"].values()), outputs["logit"].repeat(sequence_length))
    frame_scores = torch.sigmoid(frame_scores[0] if frame_scores.ndim > 1 else frame_scores)
    overlays = generate_gradcam_overlays(
        model, rgb, fft, frame_scores.detach().cpu().numpy(), output_dir, device=DEVICE
    )
    shap_result = explain_branch_embeddings(model, outputs["branch_embeddings"], output_dir)
    result = {
        "video": str(video),
        "run_name": run_name,
        "fake_score": float(torch.sigmoid(outputs["logit"])[0].cpu()),
        "overlays": overlays,
        "branch_contributions": shap_result,
        "suspicion_timeline": suspicion_timeline(outputs["per_frame_logits"]),
    }
    (output_dir / "explanation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video explanations.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--run-name", default="ablation_full")
    arguments = parser.parse_args()
    run(arguments.video, arguments.run_name)
