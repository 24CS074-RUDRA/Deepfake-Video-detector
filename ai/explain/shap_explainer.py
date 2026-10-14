from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

# Explanations save report images and do not require a desktop display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _finite_number(value):
    number = float(value)
    return number if math.isfinite(number) else 0.0


def explain_branch_embeddings(model, embeddings, output_dir, background=None):
    names = ["Frame", "Frequency", "Temporal"]
    available = [
        embeddings.get("frame"),
        embeddings.get("frequency"),
        embeddings.get("temporal"),
    ]
    values = []
    shap_values = None
    attribution_method = "fusion_ablation"
    feature_blocks = []
    for name, value in zip(names, available):
        if value is None:
            continue
        tensor = value.mean(dim=1) if value.ndim == 3 else value
        feature_blocks.append((name, tensor.detach().cpu()))
    if feature_blocks:
        fused = torch.cat([tensor for _, tensor in feature_blocks], dim=-1)
        if fused.ndim != 2 or fused.shape[0] == 0:
            raise ValueError("branch embeddings must contain at least one batch item")
        try:
            import shap

            background_tensor = torch.as_tensor(
                background, dtype=fused.dtype
            ) if background is not None else fused[: min(20, len(fused))]
            if background_tensor.ndim != 2 or background_tensor.shape[1] != fused.shape[1]:
                raise ValueError("background must match the fused embedding width")

            def fusion_predict(array):
                tensor = torch.as_tensor(array, dtype=fused.dtype, device=next(model.parameters()).device)
                with torch.no_grad():
                    return torch.sigmoid(model.fusion(tensor)).cpu().numpy()

            explainer = shap.KernelExplainer(
                fusion_predict,
                background_tensor.numpy(),
            )
            shap_values = np.asarray(explainer.shap_values(fused.numpy(), silent=True))
            if shap_values.ndim == 3:
                shap_values = shap_values[0]
            valid_shap = (
                shap_values.ndim == 2
                and shap_values.shape[1] == fused.shape[1]
                and np.isfinite(shap_values).all()
                and np.any(np.abs(shap_values) > 0)
            )
            if not valid_shap:
                shap_values = None
            else:
                attribution_method = "shap_kernel"
        except (ImportError, ModuleNotFoundError, ValueError, RuntimeError):
            shap_values = None

    fused_device = fused.to(next(model.parameters()).device) if feature_blocks else None
    if shap_values is None and fused_device is not None:
        with torch.no_grad():
            baseline_score = torch.sigmoid(model.fusion(fused_device)).flatten()

    offset = 0
    for name, tensor in feature_blocks:
        width = tensor.shape[-1]
        if shap_values is None and fused_device is not None:
            ablated = fused_device.clone()
            ablated[:, offset : offset + width] = 0
            with torch.no_grad():
                ablated_score = torch.sigmoid(model.fusion(ablated)).flatten()
            contribution = float((baseline_score - ablated_score).abs().mean())
        elif shap_values is not None:
            contribution = float(np.nan_to_num(np.abs(shap_values[:, offset : offset + width])).mean())
        else:
            contribution = 0.0
        offset += width
        values.append({"branch": name, "contribution": _finite_number(contribution)})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([item["branch"] for item in values], [item["contribution"] for item in values])
    axis.set_ylabel("Mean absolute embedding contribution")
    axis.set_title("Branch contribution to fake score")
    figure.tight_layout()
    figure.savefig(output_dir / "shap_branch_contributions.png", dpi=150)
    plt.close(figure)
    result = {item["branch"].lower(): item["contribution"] for item in values}
    result["method"] = attribution_method
    (output_dir / "shap_branch_contributions.json").write_text(json.dumps(result, indent=2))
    return result


def suspicion_timeline(per_frame_logits):
    timeline = {}
    for branch, logits in per_frame_logits.items():
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        scores = np.squeeze(scores)
        timeline[branch] = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).tolist()
    return timeline
