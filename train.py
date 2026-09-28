"""Train on GTSRB with track-disjoint validation and one reserved official test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

from cnn import CLASS_NAMES, build_model
from data import ARCHIVES, BASE_URL, audit_partitions, exclude_test_overlap_tracks, load_gtsrb, split_tracks, write_manifest

MAIN_SEED = 42
RUN_SEEDS = (42, 43, 44)
SPLIT_SEED = 42


def metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    class_ids = np.arange(len(CLASS_NAMES))
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
            "cross_entropy": float(log_loss(labels, probabilities, labels=class_ids)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
            "classification_report": classification_report(labels, predictions, labels=class_ids,
                                                            target_names=CLASS_NAMES, output_dict=True, zero_division=0)}


def predict(model, x, batch_size=128):
    return np.concatenate([model(x[i:i + batch_size], training=False).numpy()
                           for i in range(0, len(x), batch_size)])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(output, runs, test_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run, color in zip(runs, ("#2563eb", "#d97706", "#16a34a")):
        history = run["history"]
        epochs = np.arange(1, len(history["loss"]) + 1)
        for axis, metric in zip(axes, ("accuracy", "loss")):
            axis.plot(epochs, history[metric], color=color, alpha=.65, label=f"Seed {run['seed']} train")
            axis.plot(epochs, history[f"val_{metric}"], "--", color=color, label=f"Seed {run['seed']} val")
            axis.axvline(run["best_epoch"], color=color, alpha=.2, linewidth=.8)
    axes[0].set(title="Track-disjoint validation: accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].set(title="CNN regularized objective (includes L2)", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(17, 15))
    cm = np.array(test_metrics["confusion_matrix"])
    normalized = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
           xticklabels=range(len(CLASS_NAMES)),
           yticklabels=[f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)],
           xlabel="Predicted class ID", ylabel="True class", title="Official test: row-normalized confusion matrix")
    ax.tick_params(axis="both", labelsize=8)
    for row, col in zip(*np.nonzero(cm)):
        ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=5,
                color="white" if normalized[row, col] > .5 else "black")
    fig.colorbar(im, ax=ax, label="Fraction within true class", shrink=.7)
    fig.tight_layout()
    fig.savefig(output / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/gtsrb"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs, batch-size and threads must be positive")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print("Loading official GTSRB archives...", flush=True)
    data, archive_hashes = load_gtsrb(args.cache)
    x, y, groups = data["development_x"], data["development_y"], data["development_groups"]
    train_ids, val_ids = split_tracks(y, groups, SPLIT_SEED)
    train_ids, val_ids, excluded_ids, exclusions = exclude_test_overlap_tracks(data, train_ids, val_ids)
    integrity = audit_partitions(data, train_ids, val_ids)
    manifest_hash = write_manifest(args.output / "split_manifest.csv", data, train_ids, val_ids, excluded_ids)
    print(f"Train={len(train_ids)}, validation={len(val_ids)}, test={len(data['test_y'])}; integrity={integrity}", flush=True)
    dataset_info = {
        "name": "GTSRB (official final training and test sets)", "source_urls": [BASE_URL + name for name in ARCHIVES],
        "archive_hashes": archive_hashes, "training_sha256_provenance": "Locally computed, not an independently published reference",
        "original_development_count": len(y), "development_count": len(train_ids) + len(val_ids),
        **exclusions, "train_count": len(train_ids), "validation_count": len(val_ids),
        "test_count": len(data["test_y"]), "train_groups": len(np.unique(groups[train_ids])),
        "validation_groups": len(np.unique(groups[val_ids])), "test_groups": None,
        "test_group_note": "Test filenames omit track IDs; official test partition is documented as track-disjoint",
        "class_count": len(CLASS_NAMES), "class_names": CLASS_NAMES, "split_seed": SPLIT_SEED,
        "validation_track_fraction": .2, "split_hash": manifest_hash, "integrity": integrity,
        "preprocessing": "RGB, full provided image, bilinear resize to 32x32; model rescales by 1/255",
        "class_counts": {name: np.bincount(labels, minlength=len(CLASS_NAMES)).tolist() for name, labels in
                         (("train", y[train_ids]), ("validation", y[val_ids]), ("test", data["test_y"]))},
    }

    def dataset(images, labels, training, seed):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if training:
            ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(args.batch_size)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = args.threads
        return ds.with_options(options).prefetch(1)

    def fit(seed, baseline=False):
        tf.keras.backend.clear_session()
        model = build_model(seed, baseline=baseline)
        print(f"{'Linear reference' if baseline else 'CNN'} seed={seed}", flush=True)
        history = model.fit(dataset(x[train_ids], y[train_ids], True, seed),
                            validation_data=dataset(x[val_ids], y[val_ids], False, seed),
                            epochs=args.epochs, verbose=2,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                                        restore_best_weights=True)])
        history = {key: [float(v) for v in values] for key, values in history.history.items()}
        entry = {"seed": seed, "parameter_count": model.count_params(),
                 "best_epoch": int(np.argmin(history["val_loss"]) + 1), "history": history,
                 "train": metrics(y[train_ids], predict(model, x[train_ids], args.batch_size)),
                 "validation": metrics(y[val_ids], predict(model, x[val_ids], args.batch_size))}
        return model, entry

    results = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "versions": {"python": platform.python_version(), **{package: importlib.metadata.version(package)
                            for package in ("tensorflow", "keras", "numpy", "scikit-learn", "Pillow", "matplotlib")}},
               "dataset": dataset_info,
               "config": {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                          "early_stopping_patience": 5, "selection_metric": "val_loss", "cnn_seeds": list(RUN_SEEDS),
                          "learning_rate": .001, "l2": .0001, "dropout": .3, "clipnorm": 1.0},
               "main_seed": MAIN_SEED, "main_seed_policy": "Fixed before training; no selection among seeds",
               "runs": [], "exported_model": "model.keras"}
    for seed in RUN_SEEDS:
        model, entry = fit(seed)
        results["runs"].append(entry)
        if seed == MAIN_SEED:
            model.save(args.output / "model.keras")
            lines = []
            model.summary(print_fn=lambda line, **kwargs: lines.append(line))
            (args.output / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        write_json(args.output / "development_results.json", results)
        del model
    model, results["baseline"] = fit(MAIN_SEED, baseline=True)
    del model
    accuracies = [run["validation"]["accuracy"] for run in results["runs"]]
    results["validation_accuracy_mean"] = float(np.mean(accuracies))
    results["validation_accuracy_std"] = float(np.std(accuracies, ddof=1))
    write_json(args.output / "development_results.json", results)
    # All development runs are complete before this sole official-test model evaluation.
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(args.output / "model.keras")
    results["test"] = metrics(data["test_y"], predict(model, data["test_x"], args.batch_size))
    results["elapsed_seconds"] = time.monotonic() - start
    write_json(args.output / "results.json", results)
    plot_results(args.output, results["runs"], results["test"])
    print(f"DONE: validation mean={results['validation_accuracy_mean']:.4f}; "
          f"test accuracy={results['test']['accuracy']:.4f}; macro-F1={results['test']['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
