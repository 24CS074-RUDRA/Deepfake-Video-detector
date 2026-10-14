from __future__ import annotations

import argparse
from types import SimpleNamespace

import pandas as pd

from ai.utils.config import PROCESSED_DATASET, REPORT_DIR
from scripts.train import BRANCH_CONFIGS, parse_branches, train_model


ABLATIONS = ["frame", "frequency", "frame+temporal", "full"]


def run_ablations(args):
    metadata_path = PROCESSED_DATASET / "metadata.csv"
    if not metadata_path.exists():
        raise SystemExit(
            f"Metadata file not found: {metadata_path}\n"
            "Run preprocessing first: "
            "python -m scripts.preprocess --limit 5"
        )
    results = []
    for branches in ABLATIONS:
        run_name = f"{args.run_name}_{branches.replace('+', '_')}"
        train_args = SimpleNamespace(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            branches=branches,
            seq_len=args.seq_len,
            limit=args.limit,
            resume=None,
            run_name=run_name,
        )
        history = train_model(train_args)
        if history.empty:
            continue
        best = history.loc[history["val_auc"].fillna(-1).idxmax()]
        results.append({
            "configuration": branches,
            "run_name": run_name,
            "val_loss": best["val_loss"],
            "val_acc": best["val_acc"],
            "val_auc": best["val_auc"],
            "best_epoch": int(best["epoch"]),
        })

    report_dir = REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(report_dir / "ablation_val.csv", index=False)
    print(result_frame.to_string(index=False))
    if not result_frame.empty:
        recommendation = result_frame.sort_values(
            ["val_auc", "val_acc", "val_loss"],
            ascending=[False, False, True],
        ).iloc[0]
        print(
            f"Recommended final configuration: {recommendation['configuration']} "
            f"(validation AUC={recommendation['val_auc']:.4f})"
        )
    return result_frame


def build_parser():
    parser = argparse.ArgumentParser(description="Run train/val branch ablations.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-name", default="ablation")
    return parser


if __name__ == "__main__":
    run_ablations(build_parser().parse_args())
