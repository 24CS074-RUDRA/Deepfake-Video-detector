import argparse
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2
from retinaface import RetinaFace

from ai.utils.config import ROOT_DIR, MODEL_DIR
from ai.models.deepfake_detector import DeepfakeDetector

# Settings
SEQUENCE_LENGTH = 10
TEMP_DIR = ROOT_DIR / "temp" / "predict"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_faces_from_video(video_path: Path, temp_frames_dir: Path, temp_faces_dir: Path):
    """
    Extract frames from video and crop faces from them.
    """
    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    temp_faces_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Extracting frames from video...")
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    saved_frames = []
    
    # Read frames with step of 10
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % 10 == 0:
            frame_path = temp_frames_dir / f"frame_{len(saved_frames):04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(frame_path)
        frame_count += 1
    cap.release()
    print(f"Extracted {len(saved_frames)} frames.")

    print("Step 2: Detecting and cropping faces...")
    saved_faces = []
    for idx, img_path in enumerate(saved_frames):
        try:
            # Extract face and resize
            faces = RetinaFace.extract_faces(
                img_path=str(img_path),
                target_size=(224, 224),
                min_max_norm=False
            )
            if faces:
                face_rgb = faces[0]
                face_bgr = face_rgb[:, :, ::-1] # RGB to BGR
                face_path = temp_faces_dir / f"face_{idx:04d}.jpg"
                cv2.imwrite(str(face_path), face_bgr)
                saved_faces.append(face_path)
        except Exception as e:
            pass

    print(f"Detected {len(saved_faces)} faces in total.")
    return saved_faces

def predict_video(video_path_str: str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"✖ Error: Video file not found at: {video_path}")
        return

    model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        print(f"✖ Error: Model checkpoint not found at {model_path}.")
        print("Please train the model first: python -m scripts.train")
        return

    print("=" * 60)
    print(f"Inference for: {video_path.name}")
    print("=" * 60)

    # 1. Create temporary folders
    temp_frames = TEMP_DIR / "frames"
    temp_faces = TEMP_DIR / "faces"
    
    try:
        # 2. Extract faces
        face_paths = extract_faces_from_video(video_path, temp_frames, temp_faces)
        
        if len(face_paths) == 0:
            print("✖ Error: No faces detected in the video. Prediction aborted.")
            return

        # 3. Load faces and sample sequence
        n_faces = len(face_paths)
        indices = np.linspace(0, n_faces - 1, SEQUENCE_LENGTH, dtype=int)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        frames = []
        for i in indices:
            img = Image.open(face_paths[i]).convert('RGB')
            frames.append(transform(img))
            
        # Shape: (1, sequence_length, channels, height, width)
        input_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)

        # 4. Load trained model and predict
        device_name = torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU"
        print(f"\nLoading model and predicting on GPU device: {device_name}...")
        print("[GPU Processing] Moving model and inputs to GPU memory...")
        model = DeepfakeDetector(sequence_length=SEQUENCE_LENGTH).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()

        # 5. Output Result
        print("\n" + "=" * 40)
        print("PREDICTION RESULT")
        print("=" * 40)
        if prob >= 0.5:
            print(f"Result    : 🚨 DEEPFAKE (FAKE)")
            print(f"Confidence: {prob * 100:.2f}%")
        else:
            print(f"Result    : ✅ REAL VIDEO")
            print(f"Confidence: {(1.0 - prob) * 100:.2f}%")
        print("=" * 40)

    finally:
        # 6. Cleanup temporary files
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print("\nTemporary files cleaned up.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a video is Real or Fake.")
    parser.add_argument(
        "--video", 
        type=str, 
        default="dataset/raw/ffpp_real/001.mp4", 
        help="Path to the video file to predict."
    )
    args = parser.parse_args()
    
    predict_video(args.video)
