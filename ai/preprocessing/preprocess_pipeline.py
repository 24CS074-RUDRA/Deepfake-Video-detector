from ai.preprocessing.frame_extractor import extract_frames
from ai.preprocessing.face_detector import detect_faces
from ai.preprocessing.fft_transform import generate_fft_images
from ai.preprocessing.metadata_generator import generate_metadata


def run_pipeline(limit: int | None = None):

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    extract_frames(limit=limit)

    detect_faces(limit=limit)

    generate_fft_images(limit=limit)

    generate_metadata(limit=limit)

    print("\nPreprocessing completed successfully.")