"""Official GTSRB archives, shared preprocessing and track-disjoint splitting."""
from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from PIL import Image

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
ARCHIVES = {
    "GTSRB_Final_Training_Images.zip": None,
    "GTSRB_Final_Test_Images.zip": "c7e4e6327067d32654124b0fe9e82185",
    "GTSRB_Final_Test_GT.zip": "fe31e9c9270bbcd7b84b7f21a9d9d9e5",
}
EXPECTED_COUNTS = (39209, 12630)


def preprocess_image(image):
    return np.asarray(image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR))


def track_id(class_id, filename):
    stem = Path(filename).stem
    track, frame = stem.split("_")
    if not track.isdigit() or not frame.isdigit():
        raise ValueError(f"Unexpected training filename: {filename}")
    return f"{int(class_id):05d}/{track}"


def split_tracks(labels, groups, seed=42, validation_fraction=0.2):
    labels, groups = np.asarray(labels), np.asarray(groups)
    if len(labels) != len(groups) or not 0 < validation_fraction < 1:
        raise ValueError("Labels/groups must align and validation_fraction must be in (0, 1)")
    for group in np.unique(groups):
        if len(np.unique(labels[groups == group])) != 1:
            raise ValueError(f"Track {group} contains multiple classes")
    rng = np.random.default_rng(seed)
    validation_groups = []
    for label in np.unique(labels):
        tracks = np.unique(groups[labels == label])
        if len(tracks) < 2:
            raise ValueError(f"Class {label} needs at least two tracks")
        rng.shuffle(tracks)
        count = min(len(tracks) - 1, max(1, int(round(len(tracks) * validation_fraction))))
        validation_groups.extend(tracks[:count])
    is_validation = np.isin(groups, validation_groups)
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(archive, name):
    with archive.open(name) as source, Image.open(source) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        pixels = preprocess_image(rgb)
        resized_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    return pixels, digest, resized_digest


def load_gtsrb(cache=Path(".cache/gtsrb")):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive_hashes = {}
    for name, expected_md5 in ARCHIVES.items():
        destination = cache / name
        if not destination.exists():
            print(f"Downloading {name}", flush=True)
            temporary = destination.with_suffix(".download")
            urllib.request.urlretrieve(BASE_URL + name, temporary)
            temporary.replace(destination)
        if expected_md5 and file_hash(destination, "md5") != expected_md5:
            raise ValueError(f"Checksum mismatch: {destination}; remove it and download again")
        archive_hashes[name] = {"sha256": file_hash(destination), "published_md5": expected_md5}
    source_key = hashlib.sha256(str(archive_hashes).encode()).hexdigest()
    processed = cache / "rgb32_bilinear_v1.npz"
    if processed.exists():
        with np.load(processed, allow_pickle=False) as saved:
            if saved["source_key"].item() == source_key:
                return {key: saved[key] for key in saved.files if key != "source_key"}, archive_hashes
    arrays = {}
    for partition, archive_name in (("development", "GTSRB_Final_Training_Images.zip"),
                                    ("test", "GTSRB_Final_Test_Images.zip")):
        images, labels, groups, names, fingerprints, resized_fingerprints = [], [], [], [], [], []
        with zipfile.ZipFile(cache / archive_name) as archive:
            if partition == "development":
                rows = []
                for csv_name in sorted(n for n in archive.namelist() if n.endswith(".csv")):
                    text = archive.read(csv_name).decode("utf-8-sig")
                    rows.extend((str(Path(csv_name).parent).replace("\\", "/") + "/" + row["Filename"], row)
                                for row in csv.DictReader(io.StringIO(text), delimiter=";"))
            else:
                with zipfile.ZipFile(cache / "GTSRB_Final_Test_GT.zip") as ground_truth:
                    csv_name = next(n for n in ground_truth.namelist() if n.endswith(".csv"))
                    text = ground_truth.read(csv_name).decode("utf-8-sig")
                member_by_filename = {Path(n).name: n for n in archive.namelist() if n.endswith(".ppm")}
                rows = [(member_by_filename[row["Filename"]], row)
                        for row in csv.DictReader(io.StringIO(text), delimiter=";")]
            for name, row in rows:
                label = int(row["ClassId"])
                pixels, fingerprint, resized_fingerprint = _read_image(archive, name)
                images.append(pixels)
                labels.append(label)
                names.append(name)
                fingerprints.append(fingerprint)
                resized_fingerprints.append(resized_fingerprint)
                if partition == "development":
                    groups.append(track_id(label, row["Filename"]))
        arrays[f"{partition}_x"] = np.asarray(images, dtype=np.uint8)
        arrays[f"{partition}_y"] = np.asarray(labels, dtype=np.int64)
        arrays[f"{partition}_names"] = np.asarray(names)
        arrays[f"{partition}_fingerprints"] = np.asarray(fingerprints)
        arrays[f"{partition}_resized_fingerprints"] = np.asarray(resized_fingerprints)
        if groups:
            arrays["development_groups"] = np.asarray(groups)
    for partition, count in zip(("development", "test"), EXPECTED_COUNTS):
        if len(arrays[f"{partition}_y"]) != count or set(arrays[f"{partition}_y"]) != set(range(43)):
            raise ValueError(f"Unexpected {partition} counts/classes")
    np.savez_compressed(processed, source_key=source_key, **arrays)
    return arrays, archive_hashes


def exclude_test_overlap_tracks(data, train_ids, validation_ids):
    """Remove complete development tracks matching test pixels, without reading test labels.

    Split before calling: retained tracks preserve their original assignment.
    The raw cached arrays and the official test set remain unchanged.
    """
    direct_overlap = np.zeros(len(data["development_groups"]), dtype=bool)
    for key in ("fingerprints", "resized_fingerprints"):
        direct_overlap |= np.isin(data[f"development_{key}"], data[f"test_{key}"])
    excluded_groups = np.unique(data["development_groups"][direct_overlap])
    excluded_mask = np.isin(data["development_groups"], excluded_groups)
    excluded_ids = np.flatnonzero(excluded_mask)
    metadata = {"excluded_track_ids": excluded_groups.tolist(),
                "excluded_groups": len(excluded_groups),
                "excluded_images": len(excluded_ids),
                "exact_overlap_images": int(direct_overlap.sum()),
                "exclusion_rule": "Exclude complete development tracks with original RGB or resized RGB hashes present in the official test; no test labels used; original split assignments preserved"}
    return (train_ids[~excluded_mask[train_ids]], validation_ids[~excluded_mask[validation_ids]],
            excluded_ids, metadata)


def audit_partitions(data, train_ids, validation_ids):
    groups = data["development_groups"]
    group_overlap = sorted(set(groups[train_ids]) & set(groups[validation_ids]))
    result = {"train_validation_group_overlap": len(group_overlap)}
    for key in ("fingerprints", "resized_fingerprints"):
        partitions = {"train": set(data[f"development_{key}"][train_ids]),
                      "validation": set(data[f"development_{key}"][validation_ids]),
                      "test": set(data[f"test_{key}"])}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            result[f"{left}_{right}_{key}_overlap"] = len(partitions[left] & partitions[right])
    leakage = {key: value for key, value in result.items() if value}
    if leakage:
        raise ValueError(f"Cross-partition duplicates or shared tracks require investigation: {result}")
    return result


def write_manifest(path, data, train_ids, validation_ids, excluded_ids=()):
    excluded_ids = np.asarray(excluded_ids, dtype=int)
    splits = np.full(len(data["development_y"]), "train", dtype="U21")
    splits[validation_ids] = "validation"
    splits[excluded_ids] = "excluded_test_overlap"
    if sorted(np.concatenate((train_ids, validation_ids, excluded_ids)).tolist()) != list(range(len(splits))):
        raise ValueError("Train, validation and excluded indices must partition the development set exactly")
    with Path(path).open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("filename", "class_id", "track_id", "split", "rationale"))
        writer.writerows((name, label, group, split, "Track contains pixels matching official test" if split == "excluded_test_overlap" else "")
                         for name, label, group, split in zip(data["development_names"], data["development_y"], data["development_groups"], splits))
        writer.writerows((name, label, "", "test", "") for name, label in zip(data["test_names"], data["test_y"]))
    return file_hash(path)
