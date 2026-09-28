"""Checks for track integrity, shared preprocessing and portable model export."""
import os
from pathlib import Path
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
import pytest

from cnn import CLASS_NAMES, build_model
from data import audit_partitions, exclude_test_overlap_tracks, preprocess_image, split_tracks, track_id, write_manifest


def test_track_split_is_reproducible_disjoint_and_keeps_every_class():
    labels = np.repeat(np.arange(43), 15)
    groups = np.asarray([track_id(label, f"{track:05d}_{frame:05d}.ppm")
                         for label in range(43) for track in range(5) for frame in range(3)])
    train, val = split_tracks(labels, groups)
    train_again, val_again = split_tracks(labels, groups)
    np.testing.assert_array_equal(train, train_again)
    np.testing.assert_array_equal(val, val_again)
    assert not set(groups[train]) & set(groups[val])
    assert set(labels[train]) == set(labels[val]) == set(range(43))
    assert sorted(np.concatenate((train, val)).tolist()) == list(range(len(labels)))
    assert len(val) == 129
    assert track_id(0, "00001_00000.ppm") != track_id(1, "00001_00000.ppm")
    assert track_id(0, "00001_00000.ppm") == track_id(0, "00001_00002.ppm")


def test_track_split_rejects_mixed_classes_and_insufficient_tracks():
    with pytest.raises(ValueError, match="multiple classes"):
        split_tracks(np.array([0, 1]), np.array(["same", "same"]))
    with pytest.raises(ValueError, match="at least two"):
        split_tracks(np.array([0, 0]), np.array(["same", "same"]))


def test_duplicate_audit_rejects_leakage():
    data = {"development_groups": np.array(["a", "b"]),
            "development_fingerprints": np.array(["x", "y"]),
            "development_resized_fingerprints": np.array(["rx", "ry"]),
            "test_fingerprints": np.array(["z"]), "test_resized_fingerprints": np.array(["rz"])}
    assert not any(audit_partitions(data, np.array([0]), np.array([1])).values())
    data["development_fingerprints"] = np.array(["x", "x"])
    with pytest.raises(ValueError, match="duplicates"):
        audit_partitions(data, np.array([0]), np.array([1]))


def test_image_preprocessing_is_shared_and_handles_grayscale(tmp_path):
    import infer
    assert infer.preprocess_image is preprocess_image
    original = Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10))
    path = tmp_path / "gray.png"
    original.save(path)
    with Image.open(path) as reloaded:
        pixels = preprocess_image(reloaded)
    assert pixels.shape == (32, 32, 3)
    assert pixels.dtype == np.uint8
    np.testing.assert_array_equal(pixels[:, :, 0], pixels[:, :, 1])
    np.testing.assert_array_equal(pixels, np.asarray(original.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)))


def test_model_reload_preserves_probabilities(tmp_path):
    import tensorflow as tf
    model = build_model(42)
    x = np.random.default_rng(42).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    model.train_on_batch(x, np.array([0, 1, 42]))
    before = model(x, training=False).numpy()
    path = tmp_path / "model.keras"
    model.save(path)
    after = tf.keras.models.load_model(path)(x, training=False).numpy()
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(after.sum(axis=1), 1, atol=1e-6)
    assert after.shape == (3, len(CLASS_NAMES)) == (3, 43)
    reference = build_model(42, baseline=True)
    assert reference.get_layer(index=2).output.shape[-1] == 32 * 32 * 3
    assert reference.count_params() == (32 * 32 * 3 + 1) * 43


@pytest.mark.parametrize("key", ["fingerprints", "resized_fingerprints"])
@pytest.mark.parametrize("development_index", [0, 1])
def test_duplicate_audit_rejects_test_overlap(key, development_index):
    data = {"development_groups": np.array(["a", "b"]),
            "development_fingerprints": np.array(["x", "y"]),
            "development_resized_fingerprints": np.array(["rx", "ry"]),
            "test_fingerprints": np.array(["z"]), "test_resized_fingerprints": np.array(["rz"])}
    data[f"test_{key}"][0] = data[f"development_{key}"][development_index]
    with pytest.raises(ValueError, match="duplicates"):
        audit_partitions(data, np.array([0]), np.array([1]))


@pytest.mark.parametrize("key", ["fingerprints", "resized_fingerprints"])
def test_exclusion_removes_whole_track_and_preserves_split_and_test(tmp_path, key):
    data = {"development_groups": np.array(["a", "a", "b", "c", "c", "d"]),
            "development_y": np.array([0, 0, 0, 1, 1, 1]),
            "development_names": np.array([f"train/{i}.ppm" for i in range(6)]),
            "development_fingerprints": np.array([f"raw{i}" for i in range(6)]),
            "development_resized_fingerprints": np.array([f"resize{i}" for i in range(6)]),
            "test_fingerprints": np.array(["otherraw"]),
            "test_resized_fingerprints": np.array(["otherresize"]),
            "test_y": np.array([1]), "test_names": np.array(["test/0.ppm"])}
    data[f"test_{key}"][0] = data[f"development_{key}"][0]
    original = {name: values.copy() for name, values in data.items()}
    train, val = np.array([0, 1, 3, 4]), np.array([2, 5])
    retained_train, retained_val, excluded, info = exclude_test_overlap_tracks(data, train, val)
    np.testing.assert_array_equal(retained_train, [3, 4])
    np.testing.assert_array_equal(retained_val, val)
    np.testing.assert_array_equal(excluded, [0, 1])
    assert info["exact_overlap_images"] == 1
    assert info["excluded_images"] == 2
    assert info["excluded_track_ids"] == ["a"]
    assert sorted(np.concatenate((retained_train, retained_val, excluded))) == list(range(6))
    for name, values in original.items():
        np.testing.assert_array_equal(data[name], values)
    assert not any(audit_partitions(data, retained_train, retained_val).values())
    path = tmp_path / "manifest.csv"
    write_manifest(path, data, retained_train, retained_val, excluded)
    import csv
    with path.open() as source:
        rows = list(csv.DictReader(source))
    assert [row["split"] for row in rows] == ["excluded_test_overlap", "excluded_test_overlap", "validation", "train", "train", "validation", "test"]
    assert rows[0]["rationale"] and rows[-1]["filename"] == "test/0.ppm"
    with pytest.raises(ValueError, match="partition"):
        write_manifest(path, data, train, val, excluded)
