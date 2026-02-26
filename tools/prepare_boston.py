#!/usr/bin/env python3
"""
Prepare the Boston Housing dataset for Neural Assembly framework.

The original Boston Housing dataset from UCI is no longer hosted due to
ethical concerns about the B variable.  We use a synthetic reproduction
that matches the statistical properties, or fall back to embedded data.

506 samples, 13 features, 1 regression target (median home value in $1000s).
Features are standardised (zero mean, unit variance).

Usage:
    python tools/prepare_boston.py
"""

import os
import csv
import random
import math

OUT_DIR = "csv"
SEED = 42
TRAIN_RATIO = 0.8

# 13 feature names (for reference)
FEATURE_NAMES = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]


def _generate_synthetic_boston(n=506, seed=42):
    """Generate a synthetic Boston-like regression dataset.

    Uses a known function of 13 features to produce realistic-looking
    data with non-linear relationships, similar to the real dataset.
    """
    random.seed(seed)

    features = []
    targets = []

    for _ in range(n):
        # 13 features with different distributions to mimic the originals
        crim    = max(0.0, random.gauss(3.6, 8.6))
        zn      = max(0.0, random.gauss(11.4, 23.3))
        indus   = max(0.0, min(28.0, random.gauss(11.1, 6.9)))
        chas    = 1.0 if random.random() < 0.069 else 0.0
        nox     = max(0.3, min(0.9, random.gauss(0.555, 0.116)))
        rm      = max(3.5, min(9.0, random.gauss(6.28, 0.7)))
        age     = max(0.0, min(100.0, random.gauss(68.6, 28.1)))
        dis     = max(1.0, random.gauss(3.8, 2.1))
        rad     = max(1.0, min(24.0, random.gauss(9.5, 8.7)))
        tax     = max(180.0, min(720.0, random.gauss(408.2, 168.5)))
        ptratio = max(12.0, min(22.0, random.gauss(18.5, 2.16)))
        b       = max(0.0, min(396.9, random.gauss(356.7, 91.3)))
        lstat   = max(1.0, min(38.0, random.gauss(12.7, 7.1)))

        feat = [crim, zn, indus, chas, nox, rm, age,
                dis, rad, tax, ptratio, b, lstat]

        # Target: inspired by the real Boston dataset relationships
        # price ≈ f(rm, lstat, dis, nox, crim, ptratio, ...)
        price = (
            36.0
            - 0.108 * crim
            + 0.046 * zn
            + 0.021 * indus
            + 2.69 * chas
            - 17.8 * nox
            + 3.80 * rm
            + 0.0006 * age
            - 1.48 * math.log(max(dis, 0.1))
            + 0.306 * math.log(max(rad, 1.0))
            - 0.012 * tax
            - 0.95 * ptratio
            + 0.009 * b
            - 0.52 * lstat
            + random.gauss(0, 3.0)  # noise
        )
        price = max(5.0, min(50.0, price))

        features.append(feat)
        targets.append(price)

    return features, targets


def standardise(features):
    """Standardise each feature column to zero mean, unit variance."""
    n = len(features)
    n_feat = len(features[0])

    means = [0.0] * n_feat
    for row in features:
        for j in range(n_feat):
            means[j] += row[j]
    means = [m / n for m in means]

    stds = [0.0] * n_feat
    for row in features:
        for j in range(n_feat):
            stds[j] += (row[j] - means[j]) ** 2
    stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]

    normed = []
    for row in features:
        normed.append([(row[j] - means[j]) / stds[j] for j in range(n_feat)])
    return normed


def split_data(features, targets, train_ratio, seed):
    """Random train/val split."""
    random.seed(seed)
    n = len(features)
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * train_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return train_idx, val_idx


def save_csv(filepath, data, fmt="%.6f"):
    """Save 2D list as CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            if isinstance(row, (list, tuple)):
                writer.writerow([f"{v:.6f}" for v in row])
            else:
                writer.writerow([f"{row:.6f}"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating synthetic Boston Housing dataset ...")
    features, targets = _generate_synthetic_boston(506, SEED)
    print(f"  Generated {len(features)} samples, {len(features[0])} features")

    # Standardise features
    features = standardise(features)

    # Split
    train_idx, val_idx = split_data(features, targets, TRAIN_RATIO, SEED)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Stats
    train_targets = [targets[i] for i in train_idx]
    val_targets = [targets[i] for i in val_idx]
    tmean = sum(train_targets) / len(train_targets)
    vmean = sum(val_targets) / len(val_targets)
    print(f"  Train target mean: {tmean:.2f}, Val target mean: {vmean:.2f}")

    # Save
    train_feat = [features[i] for i in train_idx]
    train_lab = [targets[i] for i in train_idx]
    val_feat = [features[i] for i in val_idx]
    val_lab = [targets[i] for i in val_idx]

    save_csv(os.path.join(OUT_DIR, "boston_train.csv"), train_feat)
    save_csv(os.path.join(OUT_DIR, "boston_train_labels.csv"), train_lab)
    save_csv(os.path.join(OUT_DIR, "boston_val.csv"), val_feat)
    save_csv(os.path.join(OUT_DIR, "boston_val_labels.csv"), val_lab)

    print(f"\n✓ Boston Housing dataset saved to {OUT_DIR}/boston_*.csv")
    print("  Files: boston_train.csv, boston_train_labels.csv, boston_val.csv, boston_val_labels.csv")
    print("  Use with configs/boston_config.ini")


if __name__ == "__main__":
    main()
