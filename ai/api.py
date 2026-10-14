"""Local web API for deepfake-video analysis."""

from __future__ import annotations

import shutil
import uuid
import json
import math
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai.utils.config import MODEL_DIR, REPORT_DIR, ROOT_DIR

FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOAD_DIR = ROOT_DIR / "temp" / "uploads"
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm"}
MODEL_RUN = "full_v3"


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value

app = FastAPI(title="Deepfake Video Detector")
app.mount("/assets", StaticFiles(directory=REPORT_DIR), name="assets")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def decision_threshold() -> float:
    """Use the operating threshold selected during the completed evaluation."""
    metrics_path = REPORT_DIR / MODEL_RUN / "test_metrics.json"
    if not metrics_path.exists():
        return 0.5
    return float(json.loads(metrics_path.read_text()).get("threshold", 0.5))


def evaluation_metrics() -> dict:
    metrics_path = REPORT_DIR / MODEL_RUN / "test_metrics.json"
    if not metrics_path.exists():
        return {}
    metrics = json.loads(metrics_path.read_text())
    return {
        key: metrics[key]
        for key in ("accuracy", "precision", "recall", "f1", "auc", "eer")
        if key in metrics
    }


def build_interpretation(label, fake_score, threshold, branch_scores, overlays):
    names = {
        "frame": "spatial face appearance",
        "freq": "frequency-domain texture patterns",
        "temporal": "temporal consistency across frames",
    }
    ranked = sorted(
        ((name, float(score)) for name, score in branch_scores.items() if score is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    strongest = names.get(ranked[0][0], ranked[0][0]) if ranked else "the available model branches"
    strongest_score = ranked[0][1] if ranked else fake_score
    branch_text = "; ".join(
        f"{names.get(name, name)} {float(score):.1%}" for name, score in ranked
    ) or "no branch scores available"
    verdict_text = "supports a manipulated-video finding" if label == "FAKE" else "supports a real-video finding"
    frame_text = "No Grad-CAM frames were returned."
    if overlays:
        frame_text = "Highest-suspicion frames: " + ", ".join(
            f"{item['frame_index']} ({float(item['score']):.1%})" for item in overlays
        ) + "."
    return (
        f"The model's {label} result {verdict_text}. The strongest branch signal was "
        f"{strongest} at {strongest_score:.1%}; the fused fake score was {fake_score:.1%} "
        f"against the operating threshold of {threshold:.2%}. "
        f"Detected feature signals were: {branch_text}. Spatial features refer to face appearance, "
        "frequency features refer to FFT texture artifacts, and temporal features refer to consistency "
        "across the sampled sequence. "
        f"Grad-CAM evidence was generated for {len(overlays)} high-suspicion sampled frame(s). "
        f"{frame_text} "
        "These signals are supporting evidence for review, not standalone proof of authenticity."
    )


def build_key_indicators(branch_scores, overlays):
    labels = {
        "frame": ("Spatial appearance", "Face shape, texture, and visual detail patterns."),
        "freq": ("Frequency artifacts", "FFT-domain texture and compression inconsistency patterns."),
        "temporal": ("Temporal consistency", "Feature continuity across the sampled frame sequence."),
    }
    ranked = sorted(
        ((name, float(score)) for name, score in branch_scores.items() if score is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    leading = ranked[0][0] if ranked else None
    indicators = []
    for name in ("frame", "freq", "temporal"):
        title, description = labels[name]
        score = branch_scores.get(name)
        indicators.append({
            "label": title,
            "score": score,
            "status": "leading signal" if name == leading else "supporting signal",
            "description": description,
        })
    indicators.append({
        "label": "Frame evidence",
        "score": None,
        "status": f"{len(overlays)} overlays",
        "description": "Grad-CAM overlays identify the sampled frames that most influenced the frame-level suspicion ranking.",
    })
    return indicators


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    checkpoint = MODEL_DIR / MODEL_RUN / "best.pt"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "ready": checkpoint.exists(),
        "model_run": MODEL_RUN,
        "torch_device": device,
        "torch_device_name": device_name,
        "face_detector_device": "cpu",
    }


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if file.content_type and file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, AVI, or WebM video.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video file is required.")
    if Path(file.filename).suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, AVI, or WebM video.")

    checkpoint = MODEL_DIR / MODEL_RUN / "best.pt"
    if not checkpoint.exists():
        raise HTTPException(status_code=503, detail=f"Model checkpoint is missing: {checkpoint}")

    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
    output_dir = REPORT_DIR / "web_analyses" / job_id
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with upload_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
        from scripts.explain import run

        result = run(upload_path, MODEL_RUN, output_dir=output_dir)
        fake_score = result["fake_score"]
        threshold = decision_threshold()
        is_fake = fake_score >= threshold
        confidence = fake_score if is_fake else 1 - fake_score
        overlays = [
            {
                "frame_index": overlay["frame_index"],
                "score": overlay["score"],
                "url": f"/assets/web_analyses/{job_id}/{Path(overlay['path']).name}",
            }
            for overlay in result["overlays"]
        ]
        branch_scores = result.get("branch_scores", {})
        interpretation = build_interpretation(
            "FAKE" if is_fake else "REAL",
            fake_score,
            threshold,
            branch_scores,
            overlays,
        )
        key_indicators = build_key_indicators(branch_scores, overlays)
        findings = [
            {
                "modality": "Spatial / frame",
                "score": branch_scores.get("frame"),
                "finding": "Frame-level appearance patterns from detected faces.",
            },
            {
                "modality": "Frequency",
                "score": branch_scores.get("freq"),
                "finding": "FFT-domain artifacts and texture inconsistencies.",
            },
            {
                "modality": "Temporal",
                "score": branch_scores.get("temporal"),
                "finding": "Consistency of visual features across the sampled sequence.",
            },
        ]
        report = {
            "report_type": "deepfake_forensic_analysis",
            "assessment": "FAKE" if is_fake else "REAL",
            "interpretation": interpretation,
            "key_indicators": key_indicators,
            "confidence": confidence,
            "fake_score": fake_score,
            "decision_threshold": threshold,
            "model_run": MODEL_RUN,
            "device": result.get("device", "cpu"),
            "device_name": result.get("device_name", "CPU"),
            "sequence_length": result.get("sequence_length"),
            "sampled_frame_count": result.get("sampled_frame_count"),
            "findings": findings,
            "frame_evidence": overlays,
            "branch_contributions": result["branch_contributions"],
            "attribution_method": result["branch_contributions"].get("method"),
            "evaluation_metrics": evaluation_metrics(),
            "limitations": [
                "This is an assistive screening result, not proof of authenticity.",
                "Face detection is CPU-bound on native Windows TensorFlow installations.",
                "Confidence reflects model score relative to the operating threshold and is not a guarantee of correctness.",
            ],
        }
        (output_dir / "forensic_report.json").write_text(
            json.dumps(json_safe(report), indent=2)
        )
        return json_safe({
            "job_id": job_id,
            "filename": file.filename,
            "device": result.get("device", "cpu"),
            "device_name": result.get("device_name", "CPU"),
            "label": "FAKE" if is_fake else "REAL",
            "fake_score": fake_score,
            "confidence": confidence,
            "decision_threshold": threshold,
            "branch_scores": branch_scores,
            "detected_frames": overlays,
            "branch_contributions": result["branch_contributions"],
            "suspicion_timeline": result["suspicion_timeline"],
            "branch_chart_url": f"/assets/web_analyses/{job_id}/shap_branch_contributions.png",
            "forensic_report": report,
            "interpretation": interpretation,
            "ai_analysis": interpretation,
            "key_indicators": key_indicators,
            "report_url": f"/assets/web_analyses/{job_id}/forensic_report.json",
        })
    except SystemExit as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}") from error
    finally:
        await file.close()
        upload_path.unlink(missing_ok=True)
