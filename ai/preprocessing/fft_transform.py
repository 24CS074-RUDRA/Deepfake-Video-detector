from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ai.utils.config import (
    FAKE_FACES,
    FAKE_FACES_FFT,
    IMG_SIZE,
    REAL_FACES,
    REAL_FACES_FFT,
)


def face_to_fft(face: np.ndarray) -> np.ndarray:
    """Convert a BGR face to a normalized uint8 FFT magnitude image."""
    if face is None:
        raise ValueError("face image is empty")
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def process_faces_to_fft(
    faces_dir: Path, fft_dir: Path, limit: int | None = None
):
    video_dirs = (
        sorted(path for path in faces_dir.iterdir() if path.is_dir())
        if faces_dir.exists()
        else []
    )
    if limit is not None:
        video_dirs = video_dirs[:limit]

    processed = 0
    failures = 0
    for video_dir in tqdm(video_dirs, desc=f"FFT {faces_dir.name}"):
        output_dir = fft_dir / video_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        for face_path in sorted(video_dir.glob("*.jpg")):
            output_path = output_dir / face_path.name
            if output_path.exists():
                processed += 1
                continue
            try:
                fft_image = face_to_fft(cv2.imread(str(face_path)))
                if not cv2.imwrite(str(output_path), fft_image):
                    raise OSError(f"Could not write {output_path}")
                processed += 1
            except Exception:
                failures += 1
    return processed, failures


def generate_fft_images(limit: int | None = None):
    summaries = [
        process_faces_to_fft(faces_dir, fft_dir, limit=limit)
        for faces_dir, fft_dir in (
            (REAL_FACES, REAL_FACES_FFT),
            (FAKE_FACES, FAKE_FACES_FFT),
        )
    ]
    print(
        "FFT summary: "
        f"processed={sum(item[0] for item in summaries)}, "
        f"failures={sum(item[1] for item in summaries)}"
    )
    return summaries
