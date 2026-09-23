from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from ai.models.dataset import DeepfakeSequenceDataset
from ai.models.deepfake_detector import DeepfakeDetector
from ai.inference.aggregation import aggregate_video, compare_aggregations
from ai.utils.config import (
    IMG_SIZE,
    MODEL_DIR,
    PERFORMANCE_MIN_AUC,
    PERFORMANCE_MIN_F1,
    PROCESSED_DATASET,
    REPORT_DIR,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def branch_flags(name):
    return {
        "frame": (True, False, False),
        "frequency": (False, True, False),
        "freq": (False, True, False),
        "frame+temporal": (True, False, True),
        "full": (True, True, True),
    }.get(name, (True, True, True))


def load_run(run_name):
    run_dir = MODEL_DIR / run_name
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint.get("config", {})
    branches = config.get("branches", "full")
    use_frame, use_freq, use_temporal = branch_flags(branches)
    model = DeepfakeDetector(
        sequence_length=int(config.get("seq_len", 10)),
        use_frame=use_frame,
        use_freq=use_freq,
        use_temporal=use_temporal,
        pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, config


def make_dataset(split, sequence_length, limit=None):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return DeepfakeSequenceDataset(
        PROCESSED_DATASET / "metadata.csv",
        PROCESSED_DATASET,
        sequence_length=sequence_length,
        split=split,
        transform=transform,
        limit=limit,
    )


def collect_predictions(model, dataset):
    records = []
    with torch.no_grad():
        for index in range(len(dataset)):
            rgb, fft_images, label = dataset[index]
            outputs = model(
                rgb.unsqueeze(0).to(DEVICE),
                fft_images.unsqueeze(0).to(DEVICE),
            )
            frame_scores = {
                key: torch.sigmoid(value[0]).cpu().numpy()
                for key, value in outputs["per_frame_logits"].items()
            }
            branch_scores = {
                key: float(torch.sigmoid(value[0]).cpu())
                for key, value in outputs["branch_logits"].items()
            }
            records.append({
                "video_id": dataset.videos[index]["video_id"],
                "label": int(label.item()),
                "score": float(torch.sigmoid(outputs["logit"])[0].cpu()),
                "frame_scores": frame_scores,
                "branch_scores": branch_scores,
                "dataset_index": index,
            })
    return records


def choose_threshold(records):
    labels = np.asarray([record["label"] for record in records])
    scores = np.asarray([record["score"] for record in records])
    thresholds = np.unique(np.r_[0.0, scores, 0.5, 1.0])
    best = (0.5, -1.0)
    for threshold in thresholds:
        value = f1_score(labels, scores >= threshold, zero_division=0)
        if value > best[1]:
            best = (float(threshold), float(value))
    return best[0]


def equal_error_rate(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def metrics(records, threshold):
    labels = np.asarray([record["label"] for record in records])
    scores = np.asarray([record["score"] for record in records])
    predictions = scores >= threshold
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auc": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan"),
        "eer": equal_error_rate(labels, scores) if len(np.unique(labels)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def save_roc(report_dir, result):
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(result["fpr"], result["tpr"], label=f"AUC={result['auc']:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "roc_curve.png", dpi=150)
    plt.close(figure)


def save_confusion_matrix(report_dir, matrix):
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Real", "Fake"])
    axis.set_yticks([0, 1], ["Real", "Fake"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row][column], ha="center", va="center")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def evaluate_model(run_name="ablation_full", limit=None, robustness=False):
    metadata_path = PROCESSED_DATASET / "metadata.csv"
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    model, checkpoint, config = load_run(run_name)
    sequence_length = int(config.get("seq_len", 10))
    val_records = collect_predictions(model, make_dataset("val", sequence_length, limit))
    test_records = collect_predictions(model, make_dataset("test", sequence_length, limit))
    if not val_records or not test_records:
        raise SystemExit("Both val and test splits must contain videos")

    threshold = choose_threshold(val_records)
    result = metrics(test_records, threshold)
    report_dir = REPORT_DIR / run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame([result]).drop(columns=["confusion_matrix", "fpr", "tpr"]).to_csv(
        report_dir / "test_metrics.csv", index=False
    )
    save_roc(report_dir, result)
    save_confusion_matrix(report_dir, result["confusion_matrix"])

    branch_rows = []
    for branch in sorted({key for record in test_records for key in record["branch_scores"]}):
        scores = [record["branch_scores"].get(branch, 0.5) for record in test_records]
        labels = [record["label"] for record in test_records]
        branch_rows.append({
            "branch": branch,
            "accuracy": accuracy_score(labels, np.asarray(scores) >= threshold),
            "f1": f1_score(labels, np.asarray(scores) >= threshold, zero_division=0),
            "auc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan"),
        })
    branch_rows.append({"branch": "fusion", "accuracy": result["accuracy"], "f1": result["f1"], "auc": result["auc"]})
    pd.DataFrame(branch_rows).to_csv(report_dir / "test_branch_ablation.csv", index=False)

    aggregation_rows = compare_aggregations(
        [
            next(iter(record["frame_scores"].values()), np.asarray([record["score"]]))
            for record in val_records
        ],
        [record["label"] for record in val_records],
        threshold=threshold,
    )
    pd.DataFrame(aggregation_rows).to_csv(report_dir / "aggregation_val.csv", index=False)

    failures = []
    for record in test_records:
        prediction = int(record["score"] >= threshold)
        if prediction != record["label"]:
            rows = make_dataset("test", sequence_length, limit).videos[record["dataset_index"]]["rows"]
            failures.append({
                "video_id": record["video_id"],
                "label": record["label"],
                "prediction": prediction,
                "score": record["score"],
                "face_path": rows[0]["image_path"],
            })
    (report_dir / "failure_cases.json").write_text(json.dumps(failures[:4], indent=2))

    passed = result["auc"] >= PERFORMANCE_MIN_AUC and result["f1"] >= PERFORMANCE_MIN_F1
    print(json.dumps({"run": run_name, "test": result, "performance": "PASS" if passed else "FAIL"}, indent=2))
    if not passed:
        print("Tune: learning rate, EfficientNet unfreezing, number of frames, and augmentation.")

    final_dir = MODEL_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_DIR / run_name / "best.pt", final_dir / "best.pth")
    (final_dir / "config.json").write_text(json.dumps(config, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained run on the held-out test split.")
    parser.add_argument("--run-name", default="ablation_full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--robustness", action="store_true")
    arguments = parser.parse_args()
    evaluate_model(arguments.run_name, arguments.limit, arguments.robustness)
