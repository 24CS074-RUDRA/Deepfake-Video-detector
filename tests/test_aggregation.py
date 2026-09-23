import pytest

from ai.inference.aggregation import aggregate_video


def test_video_score_aggregation_methods():
    scores = [0.1, 0.4, 0.9, 0.8]
    assert aggregate_video(scores, "mean") == pytest.approx(0.55)
    assert aggregate_video(scores, "top-k", top_k=2) == pytest.approx(0.85)
    assert aggregate_video(scores, "median") == pytest.approx(0.6)
    assert aggregate_video(scores, "majority", threshold=0.5) == 1.0