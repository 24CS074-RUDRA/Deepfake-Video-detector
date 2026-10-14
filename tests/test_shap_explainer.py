import math

import torch

from ai.explain.shap_explainer import explain_branch_embeddings


class FusionProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = torch.nn.Linear(6, 1, bias=False)
        with torch.no_grad():
            self.fusion.weight.fill_(1.0)


def test_branch_attribution_is_finite_and_informative(tmp_path):
    model = FusionProbe()
    embeddings = {
        "frame": torch.tensor([[[1.0, 2.0], [2.0, 3.0]]]),
        "frequency": torch.tensor([[[1.0], [2.0]]]),
        "temporal": torch.tensor([[1.0, 2.0, 3.0]]),
    }

    result = explain_branch_embeddings(model, embeddings, tmp_path)

    contributions = [result[name] for name in ("frame", "frequency", "temporal")]
    assert all(math.isfinite(value) for value in contributions)
    assert any(value > 0 for value in contributions)
    assert result["method"] in {"shap_kernel", "fusion_ablation"}
