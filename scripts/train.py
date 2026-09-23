from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from ai.models.dataset import DeepfakeSequenceDataset
from ai.models.deepfake_detector import DeepfakeDetector
from ai.utils.config import IMG_SIZE, MODEL_DIR, PROCESSED_DATASET, REPORT_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRANCH_CONFIGS = {
    "frame": (True, False, False),
    "frequency": (False, True, False),
    "freq": (False, True, False),
    "frame+temporal": (True, False, True),
    "full": (True, True, True),
}


def parse_branches(value: str):
    key = value.lower().strip()
    if key not in BRANCH_CONFIGS:
        raise argparse.ArgumentTypeError(
            f"branches must be one of: {', '.join(BRANCH_CONFIGS)}"
        )
    return key


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(metadata_path, split, sequence_length, batch_size, limit):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = DeepfakeSequenceDataset(
        metadata_path=metadata_path,
        dataset_dir=PROCESSED_DATASET,
        sequence_length=sequence_length,
        split=split,
        limit=limit,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


def branch_flags(branches):
    return BRANCH_CONFIGS[branches]


def compute_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = float("nan")
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "auc": float(auc),
    }


def model_loss(outputs, labels, criterion):
    loss = criterion(outputs["logit"], labels)
    auxiliary = list(outputs["branch_logits"].values())
    if auxiliary:
        loss = loss + 0.25 * sum(criterion(logits, labels) for logits in auxiliary) / len(auxiliary)
    return loss


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    losses = []
    labels_all = []
    logits_all = []
    for rgb_seq, fft_seq, labels in loader:
        rgb_seq = rgb_seq.to(DEVICE, non_blocking=True)
        fft_seq = fft_seq.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = DEVICE.type == "cuda"
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(rgb_seq, fft_seq)
            loss = model_loss(outputs, labels, criterion)

        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
        losses.append(loss.item() * labels.size(0))
        labels_all.extend(labels.detach().cpu().numpy())
        logits_all.extend(outputs["logit"].detach().float().cpu().numpy())

    count = max(len(labels_all), 1)
    metrics = compute_metrics(labels_all, logits_all) if labels_all else {"acc": 0.0, "auc": float("nan")}
    metrics["loss"] = float(sum(losses) / count)
    return metrics


def save_curves(history, report_dir):
    frame = pd.DataFrame(history)
    for metric in ("loss", "acc", "auc"):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(frame["epoch"], frame[f"train_{metric}"], label="train")
        axis.plot(frame["epoch"], frame[f"val_{metric}"], label="val")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.upper())
        axis.legend()
        figure.tight_layout()
        figure.savefig(report_dir / f"{metric}.png", dpi=150)
        plt.close(figure)


def train_model(args):
    seed_everything()
    metadata_path = PROCESSED_DATASET / "metadata.csv"
    if not metadata_path.exists():
        raise SystemExit(
            f"Metadata file not found: {metadata_path}\n"
            "Run preprocessing first: "
            "python -m scripts.preprocess --limit 5"
        )

    run_dir = MODEL_DIR / args.run_name
    report_dir = REPORT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device": str(DEVICE), "branches": args.branches})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_loader = make_loader(metadata_path, "train", args.seq_len, args.batch_size, args.limit)
    val_loader = make_loader(metadata_path, "val", args.seq_len, args.batch_size, args.limit)
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise RuntimeError("Train and validation splits must both contain videos")

    use_frame, use_freq, use_temporal = branch_flags(args.branches)
    model = DeepfakeDetector(
        sequence_length=args.seq_len,
        use_frame=use_frame,
        use_freq=use_freq,
        use_temporal=use_temporal,
        pretrained=True,
    ).to(DEVICE)
    model.freeze_frame_backbone()

    labels = [video["label"] for video in train_loader.dataset.videos]
    positives = max(sum(labels), 1)
    negatives = max(len(labels) - sum(labels), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], device=DEVICE).squeeze()
    )
    backbone_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        (backbone_parameters if name.startswith("frame_branch.features") else other_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": other_parameters, "lr": args.lr},
        {"params": backbone_parameters, "lr": args.lr * 0.1},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    start_epoch = 0
    best_auc = -float("inf")
    stale_epochs = 0
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_auc = checkpoint.get("best_auc", best_auc)
        history = checkpoint.get("history", [])

    for epoch in range(start_epoch, args.epochs):
        if epoch == 2:
            model.unfreeze_frame_backbone()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion)
        val_auc = val_metrics["auc"]
        scheduler.step(-1.0 if np.isnan(val_auc) else val_auc)
        row = {"epoch": epoch + 1}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        print(row)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc": best_auc,
            "history": history,
            "config": config,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            stale_epochs = 0
            checkpoint["best_auc"] = best_auc
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                print("Early stopping: validation AUC did not improve.")
                break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(report_dir / "history.csv", index=False)
    save_curves(history, report_dir)
    return history_frame


def build_parser():
    parser = argparse.ArgumentParser(description="Train the three-branch deepfake detector.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--branches", type=parse_branches, default="full")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="three_branch")
    return parser


if __name__ == "__main__":
    train_model(build_parser().parse_args())
