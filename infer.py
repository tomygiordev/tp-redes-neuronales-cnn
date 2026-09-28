"""Predict one user-supplied RGB image using the exported Keras model."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from cnn import CLASS_NAMES
from data import preprocess_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to image; resized to 32 x 32 RGB")
    parser.add_argument("--model", default="results/model.keras")
    args = parser.parse_args()
    os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    with Image.open(args.image) as image:
        pixels = preprocess_image(image)
    if model.output_shape[-1] != len(CLASS_NAMES):
        parser.error("The model is not a 43-class GTSRB model")
    probabilities = model(pixels[None], training=False).numpy()[0]
    order = np.argsort(probabilities)[::-1]
    print(json.dumps({CLASS_NAMES[i]: float(probabilities[i]) for i in order}, indent=2))


if __name__ == "__main__":
    main()
