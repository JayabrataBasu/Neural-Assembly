#!/usr/bin/env python3
"""
Prepare the UCI Iris dataset for Neural Assembly framework.

Downloads the classic Iris dataset (150 samples, 4 features, 3 classes),
normalises features to [0,1], performs a stratified 80/20 train/val split,
and saves CSV files compatible with the assembly data loader.

Usage:
    python tools/prepare_iris.py
"""

import os
import csv
import random
import urllib.request

IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
OUT_DIR = "csv"
SEED = 42
TRAIN_RATIO = 0.8

SPECIES_MAP = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}


def download_iris():
    """Download iris.data from UCI and return rows as list of strings."""
    print(f"Downloading Iris dataset from {IRIS_URL} ...")
    try:
        with urllib.request.urlopen(IRIS_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        print(f"  Downloaded {len(lines)} samples")
        return lines
    except Exception as e:
        print(f"  Download failed ({e}), using embedded data")
        return _embedded_iris()


def _embedded_iris():
    """Fallback: return the full 150-row Iris dataset inline."""
    # sepal_length, sepal_width, petal_length, petal_width, class
    raw = """5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1.0,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5.0,3.0,1.6,0.2,Iris-setosa
5.0,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.2,Iris-setosa
5.0,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.6,1.4,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5.0,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5.0,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1.0,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4.0,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3.0,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3.0,5.0,1.7,Iris-versicolor
6.0,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1.0,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1.0,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6.0,2.7,5.1,1.6,Iris-versicolor
5.4,3.0,4.5,1.5,Iris-versicolor
6.0,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
5.5,2.5,4.0,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
5.8,2.6,4.0,1.2,Iris-versicolor
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3.0,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
6.5,3.0,5.8,2.2,Iris-virginica
7.6,3.0,6.6,2.1,Iris-virginica
4.9,2.5,4.5,1.7,Iris-virginica
7.3,2.9,6.3,1.8,Iris-virginica
6.7,2.5,5.8,1.8,Iris-virginica
7.2,3.6,6.1,2.5,Iris-virginica
6.5,3.2,5.1,2.0,Iris-virginica
6.4,2.7,5.3,1.9,Iris-virginica
6.8,3.0,5.5,2.1,Iris-virginica
5.7,2.5,5.0,2.0,Iris-virginica
5.8,2.8,5.1,2.4,Iris-virginica
6.4,3.2,5.3,2.3,Iris-virginica
6.5,3.0,5.5,1.8,Iris-virginica
7.7,3.8,6.7,2.2,Iris-virginica
7.7,2.6,6.9,2.3,Iris-virginica
6.0,2.2,5.0,1.5,Iris-virginica
6.9,3.2,5.7,2.3,Iris-virginica
5.6,2.8,4.9,2.0,Iris-virginica
7.7,2.8,6.7,2.0,Iris-virginica
6.3,2.7,4.9,1.8,Iris-virginica
6.7,3.3,5.7,2.1,Iris-virginica
7.2,3.2,6.0,1.8,Iris-virginica
6.2,2.8,4.8,1.8,Iris-virginica
6.1,3.0,4.9,1.8,Iris-virginica
6.4,2.8,5.6,2.1,Iris-virginica
7.2,3.0,5.8,1.6,Iris-virginica
7.4,2.8,6.1,1.9,Iris-virginica
7.9,3.8,6.4,2.0,Iris-virginica
6.4,2.8,5.6,2.2,Iris-virginica
6.3,2.8,5.1,1.5,Iris-virginica
6.1,2.6,5.6,1.4,Iris-virginica
7.7,3.0,6.1,2.3,Iris-virginica
6.3,3.4,5.6,2.4,Iris-virginica
6.4,3.1,5.5,1.8,Iris-virginica
6.0,3.0,4.8,1.8,Iris-virginica
6.9,3.1,5.4,2.1,Iris-virginica
6.7,3.1,5.6,2.4,Iris-virginica
6.9,3.1,5.1,2.3,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
6.8,3.2,5.9,2.3,Iris-virginica
6.7,3.3,5.7,2.5,Iris-virginica
6.7,3.0,5.2,2.3,Iris-virginica
6.3,2.5,5.0,1.9,Iris-virginica
6.5,3.0,5.2,2.0,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica"""
    return [l.strip() for l in raw.strip().splitlines()]


def parse_iris(lines):
    """Parse CSV lines into features (float) and labels (int)."""
    features, labels = [], []
    for line in lines:
        parts = line.split(",")
        if len(parts) != 5:
            continue
        feat = [float(x) for x in parts[:4]]
        species = parts[4].strip()
        if species not in SPECIES_MAP:
            continue
        features.append(feat)
        labels.append(SPECIES_MAP[species])
    return features, labels


def normalise(features):
    """Min-max normalise each feature column to [0, 1]."""
    n_feat = len(features[0])
    mins = [min(row[j] for row in features) for j in range(n_feat)]
    maxs = [max(row[j] for row in features) for j in range(n_feat)]
    normed = []
    for row in features:
        normed.append([
            (row[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] != mins[j] else 0.0
            for j in range(n_feat)
        ])
    return normed


def stratified_split(features, labels, train_ratio, seed):
    """Split data into train/val with equal class proportions."""
    random.seed(seed)
    # Group indices by class
    class_indices = {}
    for i, lab in enumerate(labels):
        class_indices.setdefault(lab, []).append(i)

    train_idx, val_idx = [], []
    for cls in sorted(class_indices.keys()):
        idx = class_indices[cls][:]
        random.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:])

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    return train_idx, val_idx


def save_csv(filepath, data, fmt="%.6f"):
    """Save 2D list as CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            if isinstance(row, (list, tuple)):
                writer.writerow([f"{v:.6f}" for v in row])
            else:
                writer.writerow([str(row)])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    lines = download_iris()
    features, labels = parse_iris(lines)
    print(f"  Parsed {len(features)} samples, {len(set(labels))} classes")

    # Normalise features
    features = normalise(features)

    # Stratified split
    train_idx, val_idx = stratified_split(features, labels, TRAIN_RATIO, SEED)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Report class distribution
    for name, idx_list in [("Train", train_idx), ("Val", val_idx)]:
        counts = {}
        for i in idx_list:
            counts[labels[i]] = counts.get(labels[i], 0) + 1
        dist = ", ".join(f"class {k}: {v}" for k, v in sorted(counts.items()))
        print(f"    {name}: {dist}")

    # Save
    train_feat = [features[i] for i in train_idx]
    train_lab = [[labels[i]] for i in train_idx]
    val_feat = [features[i] for i in val_idx]
    val_lab = [[labels[i]] for i in val_idx]

    save_csv(os.path.join(OUT_DIR, "iris_train.csv"), train_feat)
    save_csv(os.path.join(OUT_DIR, "iris_train_labels.csv"), [[l[0]] for l in train_lab])
    save_csv(os.path.join(OUT_DIR, "iris_val.csv"), val_feat)
    save_csv(os.path.join(OUT_DIR, "iris_val_labels.csv"), [[l[0]] for l in val_lab])

    print(f"\n✓ Iris dataset saved to {OUT_DIR}/iris_*.csv")
    print("  Files: iris_train.csv, iris_train_labels.csv, iris_val.csv, iris_val_labels.csv")
    print("  Use with configs/iris_config.ini")


if __name__ == "__main__":
    main()
