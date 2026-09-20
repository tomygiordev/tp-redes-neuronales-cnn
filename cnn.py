"""Reusable CIFAR-10 data hygiene, group partitioning and Keras architecture."""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def deduplicate(train_x, train_y, test_x, test_y, groups):
    """Keep first pixel-identical image; development has priority over test.

    Labels never influence deletion. SHA256 keys are checked on original uint8
    pixels, before sampling or transformations. Near duplicates are not detected.
    """
    seen = set()
    kept = []
    for images in (train_x, test_x):
        indices = []
        for i, image in enumerate(images):
            key = hashlib.sha256(image.tobytes()).digest()
            if key not in seen:
                seen.add(key)
                indices.append(i)
        kept.append(np.asarray(indices, dtype=np.int64))
    a, b = kept
    stats = {"train_removed": len(train_x) - len(a), "test_removed": len(test_x) - len(b),
             "method": "SHA256 exact original pixels; first occurrence; train priority"}
    return train_x[a], train_y[a], test_x[b], test_y[b], groups[a], stats


def select_development(x, y, groups, samples_per_batch, seed):
    selected = []
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        if samples_per_batch and samples_per_batch < len(ids):
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=samples_per_batch, random_state=seed)
            local, _ = next(splitter.split(ids, y[ids]))
            ids = ids[local]
        selected.extend(ids.tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return x[selected], y[selected], groups[selected], selected


def group_splits(x, y, groups):
    return list(GroupKFold(n_splits=5).split(x, y, groups))


def build_model(seed=42, baseline=False):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    inputs = layers.Input((32, 32, 3), dtype="float32", name="rgb_0_255")
    x = layers.Rescaling(1.0 / 255, name="pixel_scaling")(inputs)
    if baseline:
        x = layers.AveragePooling2D(4)(x)
        x = layers.Flatten()(x)
    else:
        for index, filters in enumerate((16, 32, 64), 1):
            x = layers.Conv2D(filters, 3, strides=1, padding="same", activation="relu",
                              kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                              name=f"conv_{index}")(x)
            x = layers.MaxPooling2D(pool_size=2, strides=2, name=f"pool_{index}")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu", kernel_initializer="he_normal",
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="linear_baseline" if baseline else "cifar10_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
