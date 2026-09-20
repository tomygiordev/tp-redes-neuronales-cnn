"""Fast checks for split hygiene, reproducibility and portable model export."""
import os
from pathlib import Path
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from cnn import build_model, deduplicate, group_splits, select_development


def test_exact_duplicates_removed_without_using_labels():
    x = np.array([[[[1]]], [[[2]]], [[[1]]], [[[3]]]], dtype=np.uint8)
    y = np.array([0, 1, 9, 2])  # Conflicting duplicate label does not change priority.
    test = np.array([[[[2]]], [[[4]]], [[[4]]]], dtype=np.uint8)
    a, labels, b, _, groups, stats = deduplicate(x, y, test, np.array([1, 3, 3]), np.array([1, 2, 3, 4]))
    assert a.ravel().tolist() == [1, 2, 3]
    assert labels.tolist() == [0, 1, 2]
    assert b.ravel().tolist() == [4]
    assert groups.tolist() == [1, 2, 4]
    assert stats["train_removed"] == 1 and stats["test_removed"] == 2


def test_selection_reproducible_and_group_cv_disjoint_complete():
    x = np.arange(500)[:, None]
    y = np.tile(np.arange(10), 50)
    groups = np.repeat(np.arange(5), 100)
    a, labels, selected_groups, ids = select_development(x, y, groups, 40, 42)
    assert np.array_equal(ids, select_development(x, y, groups, 40, 42)[3])
    assert len(a) == 200
    validation_ids = []
    for train, val in group_splits(a, labels, selected_groups):
        assert not set(selected_groups[train]) & set(selected_groups[val])
        assert not set(train) & set(val)
        assert len(train) + len(val) == 200
        validation_ids.extend(val.tolist())
    assert sorted(validation_ids) == list(range(200))


def test_model_reload_preserves_probabilities(tmp_path):
    import tensorflow as tf
    model = build_model(42)
    x = np.random.default_rng(42).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    model.train_on_batch(x, np.array([0, 1, 2]))
    before = model(x, training=False).numpy()
    path = tmp_path / "model.keras"
    model.save(path)
    after = tf.keras.models.load_model(path)(x, training=False).numpy()
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(after.sum(axis=1), 1, atol=1e-6)
    assert after.shape == (3, 10)
