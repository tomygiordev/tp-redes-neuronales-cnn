"""Reproducible group-CV development, diagnostic experiments and final test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

from cnn import CLASS_NAMES, build_model, deduplicate, group_splits, select_development


def metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
            "cross_entropy": float(log_loss(labels, probabilities, labels=np.arange(10))),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=np.arange(10)).tolist(),
            "classification_report": classification_report(labels, predictions, labels=np.arange(10),
                                                            target_names=CLASS_NAMES, output_dict=True, zero_division=0)}


def predict(model, x, batch_size=128):
    # Direct batched calls avoid a second tf.data threadpool at inference.
    return np.concatenate([model(x[i:i + batch_size], training=False).numpy()
                           for i in range(0, len(x), batch_size)])


def shifted(x):
    result = np.zeros_like(x)
    result[:, 2:, 2:, :] = x[:, :-2, :-2, :]
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(output, folds, final_history, test_metrics, test_x, test_y, test_probs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for fold in folds:
        h = fold["history"]
        epochs = np.arange(1, len(h["loss"]) + 1)
        axes[0].plot(epochs, h["accuracy"], alpha=.55, label=f"F{fold['fold']} train")
        axes[0].plot(epochs, h["val_accuracy"], "--", label=f"F{fold['fold']} val")
        axes[1].plot(epochs, h["loss"], alpha=.55)
        axes[1].plot(epochs, h["val_loss"], "--")
    axes[0].set(title="Group CV: accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].set(title="Group CV: regularized loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(ncol=2, fontsize=7)
    fig.tight_layout(); fig.savefig(output / "cv_learning_curves.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = np.array(test_metrics["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title="Final held-out test: counts")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=7,
                    color="white" if cm[row, col] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output / "test_confusion_matrix.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(3, 6, figsize=(12, 7))
    for i, ax in enumerate(axes.flat):
        ax.imshow(test_x[i]); ax.axis("off")
        guess = int(test_probs[i].argmax())
        ax.set_title(f"True: {CLASS_NAMES[test_y[i]]}\nPred: {CLASS_NAMES[guess]}",
                     fontsize=9, color="green" if guess == test_y[i] else "crimson")
    fig.tight_layout(); fig.savefig(output / "sample_predictions.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-batch", type=int, default=2000, help="0 uses all development images")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.epochs < 1 or args.samples_per_batch < 0 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs, batch-size and threads must be positive; samples-per-batch must be nonnegative")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print("Loading CIFAR-10; checking exact duplicates before sampling...", flush=True)
    (x, y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    y, test_y = y.reshape(-1), test_y.reshape(-1)
    groups = np.repeat(np.arange(1, 6), 10000)
    x, y, test_x, test_y, groups, duplicate_stats = deduplicate(x, y, test_x, test_y, groups)
    x, y, groups, selected = select_development(x, y, groups, args.samples_per_batch, args.seed)
    splits = group_splits(x, y, groups)
    print(f"Development={len(x)}, test={len(test_x)}, duplicates={duplicate_stats}", flush=True)
    manifest = {"dataset": "CIFAR-10", "original_train_count": 50000, "original_test_count": 10000,
                "development_count": len(x), "test_count": len(test_x), "deduplication": duplicate_stats,
                "group_definition": "Original CIFAR-10 data_batch_1 through data_batch_5; not capture/source groups",
                "class_names": CLASS_NAMES, "seed": args.seed, "samples_per_batch": args.samples_per_batch,
                "groups": {str(g): int(np.sum(groups == g)) for g in np.unique(groups)},
                "development_pixels_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                "test_pixels_sha256": hashlib.sha256(test_x.tobytes()).hexdigest(),
                "selected_indices_into_deduplicated_train": selected.tolist()}
    write_json(args.output / "data_manifest.json", manifest)

    class EpochLog(tf.keras.callbacks.Callback):
        def __init__(self, name):
            super().__init__(); self.name = name

        def on_epoch_end(self, epoch, logs=None):
            print(f"{self.name} epoch {epoch + 1}: " + " ".join(f"{k}={v:.4f}" for k, v in (logs or {}).items()), flush=True)

    def dataset(images, labels, training):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if training:
            ds = ds.shuffle(len(images), seed=args.seed, reshuffle_each_iteration=True)
        ds = ds.batch(args.batch_size)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = args.threads
        return ds.with_options(options).prefetch(1)

    def fit(model, name, train_ids, val_ids=None, epochs=None):
        callbacks = [EpochLog(name)]
        if val_ids is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True))
        history = model.fit(dataset(x[train_ids], y[train_ids], True),
                            validation_data=dataset(x[val_ids], y[val_ids], False) if val_ids is not None else None,
                            epochs=epochs or args.epochs, verbose=0, callbacks=callbacks)
        return {key: [float(v) for v in values] for key, values in history.history.items()}

    folds = []
    oof = np.zeros((len(x), 10), dtype=np.float32)
    for fold, (train_ids, val_ids) in enumerate(splits, 1):
        tf.keras.backend.clear_session()
        model = build_model(args.seed + fold)
        history = fit(model, f"CV {fold}/5", train_ids, val_ids)
        val_probs = predict(model, x[val_ids])
        oof[val_ids] = val_probs
        entry = {"fold": fold, "train_groups": np.unique(groups[train_ids]).tolist(),
                 "validation_groups": np.unique(groups[val_ids]).tolist(), "train_count": len(train_ids),
                 "validation_count": len(val_ids), "best_epoch": int(np.argmin(history["val_loss"]) + 1),
                 "history": history, "train": metrics(y[train_ids], predict(model, x[train_ids])),
                 "validation": metrics(y[val_ids], val_probs)}
        folds.append(entry)
        write_json(args.output / "cv_progress.json", folds)
    fixed_epochs = max(1, int(np.median([f["best_epoch"] for f in folds])))
    # Diagnostics use only the same development holdout as fold 1.
    train_ids, val_ids = splits[0]
    diagnostic_results = {}
    for name, baseline, fraction in (("linear_baseline", True, 1.0), ("half_training_data", False, .5)):
        ids = train_ids
        if fraction < 1:
            local, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=args.seed).split(ids, y[ids]))
            ids = ids[local]
        tf.keras.backend.clear_session()
        model = build_model(args.seed + 1, baseline=baseline)
        history = fit(model, name, ids, val_ids)
        diagnostic_results[name] = {"train_count": len(ids), "validation_count": len(val_ids), "history": history,
                                    "train": metrics(y[ids], predict(model, x[ids])),
                                    "validation": metrics(y[val_ids], predict(model, x[val_ids]))}
    tf.keras.backend.clear_session()
    model = build_model(args.seed + 100)
    summary_lines = []
    model.summary(print_fn=lambda line, **kwargs: summary_lines.append(line))
    (args.output / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    final_history = fit(model, "Final development fit", np.arange(len(x)), epochs=fixed_epochs)
    model.save(args.output / "model.keras")
    reloaded = tf.keras.models.load_model(args.output / "model.keras")
    reload_error = float(np.max(np.abs(predict(model, x[:32]) - predict(reloaded, x[:32]))))
    # Final test becomes visible only after architecture and epoch decisions are frozen.
    test_probs = predict(reloaded, test_x)
    test_metrics = metrics(test_y, test_probs)
    shift_probs = predict(reloaded, shifted(test_x))
    fold_acc = [f["validation"]["accuracy"] for f in folds]
    results = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "versions": {"python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__},
               "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "elapsed_seconds": time.monotonic() - start, "folds": folds,
               "cv_accuracy_mean": float(np.mean(fold_acc)), "cv_accuracy_std": float(np.std(fold_acc, ddof=1)),
               "oof": metrics(y, oof), "diagnostics": diagnostic_results,
               "final_epochs": fixed_epochs, "final_history": final_history,
               "final_train": metrics(y, predict(reloaded, x)), "test": test_metrics,
               "robustness_shift_2px": metrics(test_y, shift_probs), "reload_max_abs_error": reload_error}
    write_json(args.output / "results.json", results)
    np.savez_compressed(args.output / "predictions.npz", development_labels=y, groups=groups,
                        oof_probabilities=oof, test_labels=test_y, test_probabilities=test_probs,
                        shifted_test_probabilities=shift_probs)
    plot_results(args.output, folds, final_history, test_metrics, test_x, test_y, test_probs)
    print(f"DONE: CV accuracy={results['cv_accuracy_mean']:.4f}; test accuracy={test_metrics['accuracy']:.4f}; "
          f"macro-F1={test_metrics['macro_f1']:.4f}; results={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
