import torch

from ai.models.deepfake_detector import DeepfakeDetector


CONFIGS = {
    "frame": (True, False, False),
    "frequency": (False, True, False),
    "frame_temporal": (True, False, True),
    "full": (True, True, True),
}


def make_inputs():
    return torch.randn(2, 4, 3, 32, 32), torch.randn(2, 4, 1, 32, 32)


def test_all_branch_configurations_have_expected_shapes():
    rgb, fft = make_inputs()
    for use_frame, use_freq, use_temporal in CONFIGS.values():
        model = DeepfakeDetector(
            use_frame=use_frame,
            use_freq=use_freq,
            use_temporal=use_temporal,
            pretrained=False,
        )
        outputs = model(rgb, fft)
        assert outputs["logit"].shape == (2,)
        assert outputs["branch_embeddings"]["fusion"].shape[0] == 2
        if use_frame:
            assert outputs["branch_embeddings"]["frame"].shape == (2, 4, 1280)
            assert outputs["per_frame_logits"]["frame"].shape == (2, 4)
        if use_freq:
            assert outputs["branch_embeddings"]["frequency"].shape == (2, 4, 128)
            assert outputs["per_frame_logits"]["freq"].shape == (2, 4)
        if use_temporal:
            assert outputs["branch_embeddings"]["temporal"].shape == (2, 256)
            assert outputs["branch_logits"]["temporal"].shape == (2,)


def test_frequency_branch_can_overfit_one_batch():
    torch.manual_seed(7)
    rgb, fft = make_inputs()
    labels = torch.tensor([0.0, 1.0])
    model = DeepfakeDetector(
        use_frame=False,
        use_freq=True,
        use_temporal=False,
        pretrained=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    initial_loss = None
    final_loss = None
    for step in range(25):
        optimizer.zero_grad()
        loss = criterion(model(rgb, fft)["logit"], labels)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    assert final_loss < initial_loss
