"""
Native dataset loaders for PyNeural.

Provides torch-style dataset loading for standard benchmarks:
  - MNIST (IDX binary format)
  - CIFAR-10 (pickle format)
  - Iris, Wine, Boston (UCI tabular)

Each dataset supports automatic download, caching, and train/test splits.

Example:
    >>> from pyneural.datasets import MNIST, CIFAR10
    >>> mnist = MNIST(root='./data', train=True, download=True)
    >>> x, y = mnist[0]
    >>> loader = DataLoader(mnist, batch_size=64, shuffle=True)
"""

from __future__ import annotations

import gzip
import hashlib
import os
import struct
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .tensor import Tensor


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VisionDataset:
    """Base class for datasets with transform support."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.root = Path(root).resolve()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data: List = []
        self.targets: List[int] = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __repr__(self) -> str:
        split = "Train" if self.train else "Test"
        return f"{self.__class__.__name__}({split}, size={len(self)})"


class TabularDataset:
    """Base class for tabular (feature vector) datasets."""

    def __init__(self):
        self.data: List[List[float]] = []
        self.targets: List = []
        self.feature_names: List[str] = []
        self.target_names: List[str] = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[float], int]:
        return self.data[idx], self.targets[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self)}, features={len(self.feature_names)})"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, expected_md5: Optional[str] = None) -> None:
    """Download a file if it doesn't exist, with optional MD5 verification."""
    if dest.exists():
        if expected_md5 is not None:
            md5 = hashlib.md5(dest.read_bytes()).hexdigest()
            if md5 == expected_md5:
                return
        else:
            return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, str(dest))

    if expected_md5 is not None:
        md5 = hashlib.md5(dest.read_bytes()).hexdigest()
        if md5 != expected_md5:
            dest.unlink()
            raise RuntimeError(
                f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {md5}"
            )


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

_MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}


def _read_idx_images(path: Path) -> List[List[float]]:
    """Read IDX3-UBYTE image file and return as list of flat float lists."""
    with gzip.open(str(path), "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} for IDX3 image file")
        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]
        data = f.read(num_images * rows * cols)

    images = []
    pixels_per = rows * cols
    for i in range(num_images):
        offset = i * pixels_per
        img = [data[offset + j] / 255.0 for j in range(pixels_per)]
        images.append(img)
    return images


def _read_idx_labels(path: Path) -> List[int]:
    """Read IDX1-UBYTE label file and return as list of ints."""
    with gzip.open(str(path), "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} for IDX1 label file")
        num_items = struct.unpack(">I", f.read(4))[0]
        data = f.read(num_items)
    return [int(b) for b in data]


class MNIST(VisionDataset):
    """
    MNIST handwritten digits dataset.

    Loads from the official IDX binary format (.idx3-ubyte / .idx1-ubyte).

    Args:
        root: Root directory where ``MNIST/`` folder will be created.
        train: If True, load training set (60k); else test set (10k).
        transform: Optional transform applied to each image (flat float list).
        target_transform: Optional transform applied to each label.
        download: If True, download dataset if not found locally.

    Example:
        >>> mnist = MNIST(root='./data', train=True, download=True)
        >>> x, y = mnist[0]  # x is a list of 784 floats, y is int
        >>> len(mnist)
        60000
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, train, transform, target_transform, download)
        self._folder = self.root / "MNIST" / "raw"

        if download:
            self._download()

        if train:
            img_path = self._folder / "train-images-idx3-ubyte.gz"
            lbl_path = self._folder / "train-labels-idx1-ubyte.gz"
        else:
            img_path = self._folder / "t10k-images-idx3-ubyte.gz"
            lbl_path = self._folder / "t10k-labels-idx1-ubyte.gz"

        if not img_path.exists():
            raise FileNotFoundError(
                f"MNIST data not found at {self._folder}. "
                "Use download=True or manually place the files."
            )

        self.data = _read_idx_images(img_path)
        self.targets = _read_idx_labels(lbl_path)

    def _download(self) -> None:
        """Download MNIST dataset files."""
        self._folder.mkdir(parents=True, exist_ok=True)
        for key, (filename, md5) in _MNIST_FILES.items():
            dest = self._folder / filename
            if dest.exists():
                continue
            for mirror in _MNIST_MIRRORS:
                try:
                    _download_file(mirror + filename, dest, md5)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"Failed to download {filename} from any mirror")


# ---------------------------------------------------------------------------
# FashionMNIST
# ---------------------------------------------------------------------------

_FASHION_MNIST_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

_FASHION_MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", None),
    "train_labels": ("train-labels-idx1-ubyte.gz", None),
    "test_images": ("t10k-images-idx3-ubyte.gz", None),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", None),
}


class FashionMNIST(VisionDataset):
    """
    Fashion-MNIST dataset (Zalando).

    Same format as MNIST but with clothing item images.

    Args:
        root: Root directory where ``FashionMNIST/`` folder will be created.
        train: If True, load training set (60k); else test set (10k).
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
        download: If True, download if not found locally.
    """

    CLASS_NAMES = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ]

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, train, transform, target_transform, download)
        self._folder = self.root / "FashionMNIST" / "raw"

        if download:
            self._download()

        if train:
            img_path = self._folder / "train-images-idx3-ubyte.gz"
            lbl_path = self._folder / "train-labels-idx1-ubyte.gz"
        else:
            img_path = self._folder / "t10k-images-idx3-ubyte.gz"
            lbl_path = self._folder / "t10k-labels-idx1-ubyte.gz"

        if not img_path.exists():
            raise FileNotFoundError(
                f"FashionMNIST data not found at {self._folder}. "
                "Use download=True or manually place the files."
            )

        self.data = _read_idx_images(img_path)
        self.targets = _read_idx_labels(lbl_path)

    def _download(self) -> None:
        self._folder.mkdir(parents=True, exist_ok=True)
        for key, (filename, md5) in _FASHION_MNIST_FILES.items():
            dest = self._folder / filename
            if dest.exists():
                continue
            _download_file(_FASHION_MNIST_URL + filename, dest, md5)


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"


class CIFAR10(VisionDataset):
    """
    CIFAR-10 dataset (Krizhevsky, 2009).

    10 classes, 32x32 RGB images. Stored in Python pickle format.

    Args:
        root: Root directory where ``cifar-10-batches-py/`` will be created.
        train: If True, load training set (50k); else test set (10k).
        transform: Optional transform for images.
        target_transform: Optional transform for labels.
        download: If True, download and extract if not found.
    """

    CLASS_NAMES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, train, transform, target_transform, download)
        self._folder = self.root / "cifar-10-batches-py"

        if download:
            self._download()

        if not self._folder.exists():
            raise FileNotFoundError(
                f"CIFAR-10 data not found at {self._folder}. "
                "Use download=True or manually place the files."
            )

        self._load()

    def _download(self) -> None:
        import tarfile
        archive = self.root / "cifar-10-python.tar.gz"
        _download_file(_CIFAR10_URL, archive, _CIFAR10_MD5)
        if not self._folder.exists():
            with tarfile.open(str(archive), "r:gz") as tar:
                tar.extractall(str(self.root), filter="data")

    def _load(self) -> None:
        import pickle

        if self.train:
            batches = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batches = ["test_batch"]

        all_data = []
        all_labels = []
        for batch_name in batches:
            batch_path = self._folder / batch_name
            with open(str(batch_path), "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            # data is (N, 3072) uint8; labels is list of ints
            raw = batch[b"data"]
            all_data.extend(raw.tolist() if hasattr(raw, "tolist") else list(raw))
            all_labels.extend(batch[b"labels"])

        # Normalize to [0, 1] floats
        self.data = [[v / 255.0 for v in sample] for sample in all_data]
        self.targets = all_labels


# ---------------------------------------------------------------------------
# Iris
# ---------------------------------------------------------------------------

_IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


class Iris(TabularDataset):
    """
    Iris flower dataset (Fisher, 1936).

    4 features, 3 classes, 150 samples.

    Args:
        root: Directory to cache the downloaded data.
        download: If True, download if not found.
    """

    FEATURE_NAMES = [
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ]
    TARGET_NAMES = ["setosa", "versicolor", "virginica"]

    def __init__(self, root: str = "./data", download: bool = False):
        super().__init__()
        self.feature_names = self.FEATURE_NAMES
        self.target_names = self.TARGET_NAMES
        self._root = Path(root).resolve()
        self._file = self._root / "iris" / "iris.data"

        if download:
            self._download()

        if not self._file.exists():
            raise FileNotFoundError(
                f"Iris data not found at {self._file}. Use download=True."
            )

        self._load()

    def _download(self) -> None:
        _download_file(_IRIS_URL, self._file)

    def _load(self) -> None:
        label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        with open(str(self._file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 5:
                    continue
                features = [float(x) for x in parts[:4]]
                label = label_map.get(parts[4], -1)
                if label < 0:
                    continue
                self.data.append(features)
                self.targets.append(label)


# ---------------------------------------------------------------------------
# Wine Quality
# ---------------------------------------------------------------------------

_WINE_RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
_WINE_WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


class WineQuality(TabularDataset):
    """
    UCI Wine Quality dataset.

    11 features, quality score (3-9). Available in red/white variants.

    Args:
        root: Directory for caching.
        color: 'red' or 'white'.
        download: If True, download if not found.
    """

    FEATURE_NAMES = [
        "fixed_acidity", "volatile_acidity", "citric_acid",
        "residual_sugar", "chlorides", "free_sulfur_dioxide",
        "total_sulfur_dioxide", "density", "pH",
        "sulphates", "alcohol",
    ]

    def __init__(self, root: str = "./data", color: str = "red", download: bool = False):
        super().__init__()
        self.feature_names = self.FEATURE_NAMES
        self._root = Path(root).resolve()
        self.color = color
        url = _WINE_RED_URL if color == "red" else _WINE_WHITE_URL
        self._file = self._root / "wine" / f"winequality-{color}.csv"
        self._url = url

        if download:
            self._download()

        if not self._file.exists():
            raise FileNotFoundError(
                f"Wine data not found at {self._file}. Use download=True."
            )
        self._load()

    def _download(self) -> None:
        _download_file(self._url, self._file)

    def _load(self) -> None:
        with open(str(self._file)) as f:
            header = f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) < 12:
                    continue
                features = [float(x) for x in parts[:11]]
                target = int(float(parts[11]))
                self.data.append(features)
                self.targets.append(target)


# ---------------------------------------------------------------------------
# Utility: train/test split
# ---------------------------------------------------------------------------

def train_test_split(
    dataset,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple:
    """
    Split a dataset into train and test portions.

    Args:
        dataset: Any dataset with .data and .targets attributes.
        test_size: Fraction of data to use for testing (0.0 to 1.0).
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.

    Returns:
        (train_data, test_data, train_targets, test_targets) tuples.
    """
    import random as _random

    n = len(dataset.data)
    indices = list(range(n))

    if shuffle:
        rng = _random.Random(seed)
        rng.shuffle(indices)

    split = int(n * (1.0 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_data = [dataset.data[i] for i in train_idx]
    test_data = [dataset.data[i] for i in test_idx]
    train_targets = [dataset.targets[i] for i in train_idx]
    test_targets = [dataset.targets[i] for i in test_idx]

    return train_data, test_data, train_targets, test_targets
