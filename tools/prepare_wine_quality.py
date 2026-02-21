#!/usr/bin/env python3
"""
Download and prepare the UCI Wine Quality (Red Wine) dataset
for the Neural Assembly Framework.

Dataset: https://archive.ics.uci.edu/ml/datasets/wine+quality
- 1599 samples, 11 physicochemical features, quality scores 3-8
- Remapped to 6 classes (0-5) for cross-entropy training
- Stratified 80/20 train/val split
- Min-max normalization using train statistics
"""

import os
import sys
import urllib.request
import random
import math

# ── Configuration ──────────────────────────────────────────────────────────
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
SEED = 42
VAL_RATIO = 0.20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_DIR = os.path.join(PROJECT_DIR, "csv")

TRAIN_FEATURES = os.path.join(CSV_DIR, "wine_train.csv")
TRAIN_LABELS   = os.path.join(CSV_DIR, "wine_train_labels.csv")
VAL_FEATURES   = os.path.join(CSV_DIR, "wine_val.csv")
VAL_LABELS     = os.path.join(CSV_DIR, "wine_val_labels.csv")


def download_raw(url):
    """Download raw CSV text from UCI repository."""
    print(f"[INFO] Downloading from {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NeuralAssembly/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        print(f"[INFO] Downloaded {len(raw)} bytes")
        return raw
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("[INFO] Trying fallback URL ...")
        # Try alternative URL
        alt_url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/wine-quality-red.csv"
        try:
            req = urllib.request.Request(alt_url, headers={"User-Agent": "NeuralAssembly/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
            print(f"[INFO] Downloaded {len(raw)} bytes from fallback")
            return raw
        except Exception as e2:
            print(f"[ERROR] Fallback also failed: {e2}")
            print("[INFO] Generating synthetic wine-like dataset as last resort ...")
            return None


def parse_uci_csv(raw_text):
    """Parse UCI wine quality CSV (semicolon-delimited with header)."""
    lines = raw_text.strip().split("\n")
    header = lines[0]
    
    # Detect delimiter (semicolon for UCI, comma for GitHub mirror)
    delim = ";" if ";" in header else ","
    col_names = [c.strip().strip('"') for c in header.split(delim)]
    print(f"[INFO] Columns ({len(col_names)}): {col_names}")
    
    features = []
    labels = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.strip().split(delim)
        if len(parts) != len(col_names):
            print(f"[WARN] Skipping malformed line {i}: {len(parts)} fields")
            continue
        try:
            vals = [float(x.strip().strip('"')) for x in parts]
        except ValueError:
            print(f"[WARN] Skipping non-numeric line {i}")
            continue
        features.append(vals[:-1])   # first 11 columns are features
        labels.append(int(vals[-1]))  # last column is quality score
    
    print(f"[INFO] Parsed {len(features)} samples, {len(features[0])} features")
    return features, labels


def generate_synthetic_wine(n_samples=1599, n_features=11, n_classes=6):
    """Generate synthetic dataset mimicking wine quality distribution."""
    print(f"[INFO] Generating synthetic wine-like dataset: {n_samples} samples, "
          f"{n_features} features, {n_classes} classes")
    random.seed(SEED)
    
    # Approximate real wine quality distribution (classes 3-8 → 0-5)
    # Real distribution: 3→10, 4→53, 5→681, 6→638, 7→199, 8→18
    class_weights = [10, 53, 681, 638, 199, 18]
    total_w = sum(class_weights)
    class_probs = [w / total_w for w in class_weights]
    
    # Feature ranges mimicking real wine data
    feat_ranges = [
        (4.0, 16.0),    # fixed acidity
        (0.1, 1.6),     # volatile acidity
        (0.0, 1.0),     # citric acid
        (0.9, 16.0),    # residual sugar
        (0.01, 0.62),   # chlorides
        (1.0, 72.0),    # free sulfur dioxide
        (6.0, 289.0),   # total sulfur dioxide
        (0.99, 1.004),  # density
        (2.7, 4.1),     # pH
        (0.3, 2.0),     # sulphates
        (8.0, 15.0),    # alcohol
    ]
    
    features = []
    labels = []
    
    for _ in range(n_samples):
        # Pick class based on distribution
        r = random.random()
        cumulative = 0
        cls = 0
        for c, p in enumerate(class_probs):
            cumulative += p
            if r <= cumulative:
                cls = c
                break
        
        # Generate features with class-dependent bias
        feat = []
        for f_idx in range(n_features):
            lo, hi = feat_ranges[f_idx]
            mid = (lo + hi) / 2
            spread = (hi - lo) / 2
            # Add class-dependent shift
            shift = (cls - n_classes / 2) / n_classes * spread * 0.3
            val = mid + shift + random.gauss(0, spread * 0.3)
            val = max(lo, min(hi, val))
            feat.append(val)
        
        features.append(feat)
        labels.append(cls)
    
    print(f"[INFO] Generated {n_samples} synthetic samples")
    return features, labels


def remap_labels(labels):
    """Remap quality scores (3-8) to contiguous 0-based class indices (0-5)."""
    unique = sorted(set(labels))
    mapping = {v: i for i, v in enumerate(unique)}
    remapped = [mapping[l] for l in labels]
    n_classes = len(unique)
    
    print(f"[INFO] Label mapping: {mapping}")
    print(f"[INFO] Class distribution:")
    for orig, idx in mapping.items():
        count = remapped.count(idx)
        print(f"       class {idx} (quality={orig}): {count} samples ({100*count/len(remapped):.1f}%)")
    
    return remapped, n_classes


def stratified_split(features, labels, val_ratio, seed):
    """Stratified train/val split preserving class proportions."""
    random.seed(seed)
    n_classes = max(labels) + 1
    
    # Group indices by class
    buckets = [[] for _ in range(n_classes)]
    for i, l in enumerate(labels):
        buckets[l].append(i)
    
    train_idx, val_idx = [], []
    for cls_indices in buckets:
        random.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_ratio))
        val_idx.extend(cls_indices[:n_val])
        train_idx.extend(cls_indices[n_val:])
    
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    
    train_X = [features[i] for i in train_idx]
    train_y = [labels[i] for i in train_idx]
    val_X   = [features[i] for i in val_idx]
    val_y   = [labels[i] for i in val_idx]
    
    print(f"[INFO] Split: {len(train_X)} train, {len(val_X)} val")
    return train_X, train_y, val_X, val_y


def normalize_minmax(train_X, val_X):
    """Min-max normalize features using train statistics only."""
    n_feat = len(train_X[0])
    
    mins = [min(row[j] for row in train_X) for j in range(n_feat)]
    maxs = [max(row[j] for row in train_X) for j in range(n_feat)]
    
    def norm_row(row):
        out = []
        for j in range(n_feat):
            rng = maxs[j] - mins[j]
            if rng < 1e-12:
                out.append(0.0)
            else:
                out.append((row[j] - mins[j]) / rng)
        return out
    
    train_norm = [norm_row(r) for r in train_X]
    val_norm   = [norm_row(r) for r in val_X]
    
    print(f"[INFO] Min-max normalized {n_feat} features using train stats")
    return train_norm, val_norm


def save_csv(features, labels, feat_path, label_path):
    """Save features and labels as framework-compatible CSV files."""
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    
    with open(feat_path, "w") as f:
        for row in features:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    
    with open(label_path, "w") as f:
        for l in labels:
            f.write(f"{l}\n")
    
    print(f"[INFO] Saved {len(features)} rows → {feat_path}")
    print(f"[INFO] Saved {len(labels)} labels → {label_path}")


def main():
    print("=" * 60)
    print("UCI Wine Quality Dataset Preparation")
    print("=" * 60)
    
    # Step 1: Download or generate
    raw = download_raw(URL)
    if raw is not None:
        features, labels = parse_uci_csv(raw)
    else:
        features, labels = generate_synthetic_wine()
    
    # Step 2: Remap labels to 0-based
    labels, n_classes = remap_labels(labels)
    
    # Step 3: Stratified split
    train_X, train_y, val_X, val_y = stratified_split(
        features, labels, VAL_RATIO, SEED
    )
    
    # Step 4: Normalize
    train_X, val_X = normalize_minmax(train_X, val_X)
    
    # Step 5: Save
    save_csv(train_X, train_y, TRAIN_FEATURES, TRAIN_LABELS)
    save_csv(val_X, val_y, VAL_FEATURES, VAL_LABELS)
    
    # Summary
    print()
    print("=" * 60)
    print(f"Dataset ready: {n_classes} classes, {len(train_X[0])} features")
    print(f"  Train: {len(train_X)} samples → {TRAIN_FEATURES}")
    print(f"  Val:   {len(val_X)} samples → {VAL_FEATURES}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
