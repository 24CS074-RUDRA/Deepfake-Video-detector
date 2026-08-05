from ai.preprocessing.frame_extractor import extract_frames
from ai.preprocessing.face_detector import detect_faces
from ai.preprocessing.metadata_generator import generate_metadata


def run_pipeline():

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    extract_frames()

    detect_faces()

    generate_metadata()

    print("\nPreprocessing completed successfully.")