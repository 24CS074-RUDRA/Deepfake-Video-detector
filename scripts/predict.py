import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.utils.config import MODEL_DIR

try:
    from scripts.explain import run
except ModuleNotFoundError:
    from explain import run

def predict_video(video_path_str: str, run_name: str = "full_v3"):
    video_path = Path(video_path_str)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at {video_path}")

    model_path = MODEL_DIR / run_name / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print("=" * 60)
    print(f"Inference for: {video_path.name}")
    print("=" * 60)
    result = run(video_path, run_name)
    threshold = float(result.get("decision_threshold", 0.5))
    score = float(result["fake_score"])
    label = "FAKE" if score >= threshold else "REAL"
    confidence = score if label == "FAKE" else 1 - score
    print("\n" + "=" * 40)
    print("PREDICTION RESULT")
    print("=" * 40)
    print(f"Result    : {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 40)
    return {**result, "label": label, "confidence": confidence, "decision_threshold": threshold}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a video is Real or Fake.")
    parser.add_argument(
        "--video", 
        type=str, 
        required=True,
        help="Path to the video file to predict."
    )
    parser.add_argument("--run-name", default="full_v3")
    args = parser.parse_args()
    
    predict_video(args.video, args.run_name)
