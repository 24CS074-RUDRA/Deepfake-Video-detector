from pathlib import Path
import cv2
from tqdm import tqdm
from retinaface import RetinaFace

from ai.utils.config import (
    REAL_FRAMES,
    FAKE_FRAMES,
    REAL_FACES,
    FAKE_FACES
)

def process_frames_to_faces(frames_dir: Path, faces_dir: Path):
    if not frames_dir.exists():
        print(f"Directory not found: {frames_dir}")
        return

    # Find all video directories (each contains frame images)
    video_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    # TIP: For a quick verification run, you can slice the list (e.g., video_dirs = video_dirs[:5])
    print(f"\nProcessing face detection on frames from {frames_dir.name} ({len(video_dirs)} videos)")

    for video_dir in tqdm(video_dirs):
        output_video_dir = faces_dir / video_dir.name
        output_video_dir.mkdir(parents=True, exist_ok=True)

        # Find all frame images
        frame_images = sorted(video_dir.glob("*.jpg"))
        for img_path in frame_images:
            output_face_path = output_video_dir / f"{img_path.stem}_face.jpg"

            # Skip if already processed
            if output_face_path.exists():
                continue

            try:
                # Extract face and resize it to 224x224
                # Set min_max_norm=False to get uint8 values for cv2.imwrite
                faces = RetinaFace.extract_faces(
                    img_path=str(img_path),
                    target_size=(224, 224),
                    min_max_norm=False
                )

                if faces:
                    # Save the first detected face (usually there is 1 face per frame in deepfakes)
                    face_rgb = faces[0]
                    # Convert RGB to BGR for cv2.imwrite
                    face_bgr = face_rgb[:, :, ::-1]
                    cv2.imwrite(str(output_face_path), face_bgr)
            except Exception as e:
                # Keep processing other frames
                pass

def detect_faces():
    print("\n" + "=" * 60)
    print("FACE DETECTION")
    print("=" * 60)

    process_frames_to_faces(REAL_FRAMES, REAL_FACES)
    process_frames_to_faces(FAKE_FRAMES, FAKE_FACES)

    print("\n✅ Face detection completed.")
