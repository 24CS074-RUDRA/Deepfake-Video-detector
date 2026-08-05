"""
Frame Extraction Module

Reads videos from dataset/raw
and extracts frames into
dataset/processed.
"""

from pathlib import Path
import cv2
from tqdm import tqdm

from ai.utils.config import (
    REAL_VIDEOS,
    FAKE_VIDEOS,
    REAL_FRAMES,
    FAKE_FRAMES
)


FRAME_INTERVAL = 10


def extract_video_frames(video_path: Path, output_folder: Path):

    output_folder.mkdir(parents=True, exist_ok=True)

    # Skip if already processed
    if any(output_folder.glob("*.jpg")):
        print(f"⏭ Skipping {video_path.name} (already processed)")
        return

    cap = cv2.VideoCapture(str(video_path))

    frame_count = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_count % FRAME_INTERVAL == 0:

            filename = output_folder / f"frame_{saved_count:04d}.jpg"

            cv2.imwrite(str(filename), frame)

            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"✔ {video_path.name} → {saved_count} frames")


def process_folder(input_folder: Path, output_folder: Path):

    if not input_folder.exists():
        print(f"Folder not found: {input_folder}")
        return

    videos = sorted(input_folder.glob("*.mp4"))

    print(f"\nProcessing {len(videos)} videos from {input_folder.name}")

    for video in tqdm(videos):

        save_folder = output_folder / video.stem

        extract_video_frames(video, save_folder)


def extract_frames():

    print("\n" + "=" * 60)
    print("FRAME EXTRACTION")
    print("=" * 60)

    process_folder(REAL_VIDEOS, REAL_FRAMES)

    process_folder(FAKE_VIDEOS, FAKE_FRAMES)

    print("\n✅ Frame extraction completed.")