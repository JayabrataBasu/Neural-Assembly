#!/usr/bin/env python3
"""
Prepare the CIFAR-10 dataset for Neural Assembly framework.

Downloads the CIFAR-10 Python batch files from the University of Toronto,
flattens 32×32×3 images to 3072-element vectors, normalises pixel values
to [0,1], and saves CSV files.

CIFAR-10: 60,000 images total — 50,000 train, 10,000 test.
10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Usage:
    python tools/prepare_cifar10.py [--subset N]

    --subset N:  Only use first N samples per split (useful for quick tests).
                 Default: full dataset (50000 train, 10000 test).
"""

import os
import csv
import sys
import struct
import tarfile
import hashlib
import urllib.request
import pickle

OUT_DIR = "csv"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
TAR_FILE = "cifar-10-python.tar.gz"
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def download_cifar10(cache_dir="."):
    """Download and extract CIFAR-10 if not already present."""
    tar_path = os.path.join(cache_dir, TAR_FILE)
    extract_dir = os.path.join(cache_dir, "cifar-10-batches-py")

    if os.path.isdir(extract_dir):
        print(f"  Found existing {extract_dir}")
        return extract_dir

    if not os.path.isfile(tar_path):
        print(f"  Downloading CIFAR-10 from {URL} ...")
        urllib.request.urlretrieve(URL, tar_path)
        print(f"  Downloaded {os.path.getsize(tar_path) / 1e6:.1f} MB")

    # Verify MD5
    md5 = hashlib.md5(open(tar_path, "rb").read()).hexdigest()
    if md5 != EXPECTED_MD5:
        print(f"  WARNING: MD5 mismatch ({md5} != {EXPECTED_MD5})")

    print(f"  Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=cache_dir)

    return extract_dir


def load_batch(filepath):
    """Load a single CIFAR-10 batch file."""
    with open(filepath, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]       # numpy uint8 array (N, 3072)
    labels = batch[b"labels"]   # list of ints
    return data, labels


def load_cifar10(extract_dir, subset=None):
    """Load all CIFAR-10 train and test data."""
    # Training batches
    train_data_list, train_labels = [], []
    for i in range(1, 6):
        batch_file = os.path.join(extract_dir, f"data_batch_{i}")
        data, labels = load_batch(batch_file)
        train_data_list.append(data)
        train_labels.extend(labels)
    
    # Concatenate (requires numpy for the uint8 arrays from pickle)
    import numpy as np
    train_data = np.concatenate(train_data_list, axis=0)

    # Test batch
    test_file = os.path.join(extract_dir, "test_batch")
    test_data, test_labels = load_batch(test_file)

    if subset:
        train_data = train_data[:subset]
        train_labels = train_labels[:subset]
        test_data = test_data[:subset]
        test_labels = test_labels[:subset]

    return train_data, train_labels, test_data, test_labels


def save_data_csv(filepath, data):
    """Save normalised pixel data as CSV (float values 0-1)."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            # Normalise uint8 [0,255] to float [0,1]
            writer.writerow([f"{v / 255.0:.6f}" for v in row])


def save_labels_csv(filepath, labels):
    """Save integer labels as CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([str(label)])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parse args
    subset = None
    if "--subset" in sys.argv:
        idx = sys.argv.index("--subset")
        if idx + 1 < len(sys.argv):
            subset = int(sys.argv[idx + 1])
            print(f"Using subset of {subset} samples per split")

    print("Preparing CIFAR-10 dataset ...")
    extract_dir = download_cifar10()
    train_data, train_labels, test_data, test_labels = load_cifar10(
        extract_dir, subset=subset
    )

    print(f"  Train: {len(train_labels)} samples")
    print(f"  Test:  {len(test_labels)} samples")
    print(f"  Features: {train_data.shape[1]} (32×32×3 flattened)")
    print(f"  Classes: {len(CLASS_NAMES)}")

    # Class distribution
    from collections import Counter
    train_dist = Counter(train_labels)
    print(f"  Train class dist: {dict(sorted(train_dist.items()))}")

    # Save
    print("  Saving CSV files (this may take a moment for 50k samples) ...")
    save_data_csv(os.path.join(OUT_DIR, "cifar10_train.csv"), train_data)
    save_labels_csv(os.path.join(OUT_DIR, "cifar10_train_labels.csv"), train_labels)
    save_data_csv(os.path.join(OUT_DIR, "cifar10_test.csv"), test_data)
    save_labels_csv(os.path.join(OUT_DIR, "cifar10_test_labels.csv"), test_labels)

    print(f"\n✓ CIFAR-10 dataset saved to {OUT_DIR}/cifar10_*.csv")
    print("  Files: cifar10_train.csv, cifar10_train_labels.csv,")
    print("         cifar10_test.csv, cifar10_test_labels.csv")
    print("  Use with configs/cifar10_config.ini")
    if not subset:
        print("  Note: CSV files are large (~750 MB). Use --subset N for quick tests.")


if __name__ == "__main__":
    main()
