import numpy as np

from cdm_challenger_v1 import probability_model, select_boxes, top_numbers


def sample_history():
    rng = np.random.default_rng(42)
    return rng.integers(0, 10, size=(1000, 3))


def test_probability_model_is_valid_and_deterministic():
    history = sample_history()
    first, components, diagnostics = probability_model(history)
    second, _, _ = probability_model(history)
    assert first.shape == (1000,)
    assert np.isclose(first.sum(), 1.0)
    assert np.all(first > 0)
    assert np.array_equal(first, second)
    assert set(components) == {"cdm", "position"}
    assert 1.0 <= diagnostics["concentration"] <= 10000.0


def test_selections_are_distinct_and_boxes_have_matching_coverage():
    probabilities, _, _ = probability_model(sample_history())
    straights = top_numbers(probabilities)
    boxes = select_boxes(probabilities)
    assert len(straights) == len(set(straights)) == 3
    assert len(boxes) == 3
    assert all(box["ways"] in (3, 6) for box in boxes)
    assert all(len(set(box["digits"])) in (2, 3) for box in boxes)
