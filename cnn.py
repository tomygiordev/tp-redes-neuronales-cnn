"""Compact GTSRB CNN and a linear softmax reference."""
from __future__ import annotations

CLASS_NAMES = [
    "Speed limit (20 km/h)", "Speed limit (30 km/h)", "Speed limit (50 km/h)",
    "Speed limit (60 km/h)", "Speed limit (70 km/h)", "Speed limit (80 km/h)",
    "End of speed limit (80 km/h)", "Speed limit (100 km/h)", "Speed limit (120 km/h)",
    "No passing", "No passing for vehicles over 3.5 tonnes", "Right-of-way at next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 tonnes prohibited",
    "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing for vehicles over 3.5 tonnes",
]


def build_model(seed=42, baseline=False):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    inputs = layers.Input((32, 32, 3), dtype="float32", name="rgb_0_255")
    x = layers.Rescaling(1.0 / 255, name="pixel_scaling")(inputs)
    if baseline:
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
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="linear_baseline" if baseline else "gtsrb_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
