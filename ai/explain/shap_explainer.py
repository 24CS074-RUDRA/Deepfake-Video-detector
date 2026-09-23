from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def explain_branch_embeddings(model, embeddings, output_dir, background=None):
    names = ["Frame", "Frequency", "Temporal"]
    available = [
        embeddings.get("frame"),
        embeddings.get("frequency"),
        embeddings.get("temporal"),
    ]
    values = []
    shap_values = None
    feature_blocks = []
    for name, value in zip(names, available):
        if value is None:
            continue
        tensor = value.mean(dim=1) if value.ndim == 3 else value
        feature_blocks.append((name, tensor.detach().cpu()))
    if feature_blocks:
        fused = torch.cat([tensor for _, tensor in feature_blocks], dim=-1)
        try:
            import shap

            background_tensor = fused[: min(20, len(fused))]

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
        except (ImportError, ModuleNotFoundError):
            shap_values = None

    offset = 0
    for name, tensor in feature_blocks:
        width = tensor.shape[-1]
        if shap_values is None:
            contribution = float(tensor.abs().mean())
        else:
            contribution = float(np.abs(shap_values[:, offset : offset + width]).mean())
        offset += width
        values.append({"branch": name, "contribution": contribution})

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
    result["method"] = "shap_kernel" if shap_values is not None else "embedding attribution fallback"
    (output_dir / "shap_branch_contributions.json").write_text(json.dumps(result, indent=2))
    return result


def suspicion_timeline(per_frame_logits):
    timeline = {}
    for branch, logits in per_frame_logits.items():
        timeline[branch] = torch.sigmoid(logits).detach().cpu().numpy().tolist()
    return timeline
