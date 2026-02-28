"""
Data transforms for PyNeural.

Provides composable, C-backed data transforms for preprocessing:
  - Normalize (z-score normalisation per feature)
  - MinMaxScale (scale to [0,1] per feature)
  - Compose (chain transforms)

All transforms operate on flat double* arrays via ctypes.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Sequence

from .core import _lib


def _make_f64(values):
    """Create a ctypes double array from a Python list."""
    n = len(values)
    return (ctypes.c_double * n)(*values)


def _f64_ptr(arr):
    """Get c_void_p from a ctypes array."""
    return ctypes.cast(arr, ctypes.c_void_p)


def compute_stats(data: List[float], num_features: int):
    """
    Compute per-feature statistics from a flat data array.

    Args:
        data: Flat list of floats, row-major [batch_size × num_features].
        num_features: Number of features per sample.

    Returns:
        dict with keys: 'mean', 'std', 'min', 'max' — each a list[float]
        of length num_features.
    """
    n = len(data)
    if n == 0 or num_features <= 0:
        raise ValueError("data must be non-empty and num_features > 0")
    batch_size = n // num_features
    if batch_size * num_features != n:
        raise ValueError(
            f"data length {n} not divisible by num_features {num_features}"
        )

    data_arr = _make_f64(data)
    mean_arr = (ctypes.c_double * num_features)()
    std_arr  = (ctypes.c_double * num_features)()
    min_arr  = (ctypes.c_double * num_features)()
    max_arr  = (ctypes.c_double * num_features)()

    result = _lib.transform_compute_stats(
        _f64_ptr(data_arr), batch_size, num_features,
        _f64_ptr(mean_arr), _f64_ptr(std_arr),
        _f64_ptr(min_arr), _f64_ptr(max_arr),
    )
    if result != 0:
        raise RuntimeError("transform_compute_stats failed")

    return {
        'mean': [float(mean_arr[i]) for i in range(num_features)],
        'std':  [float(std_arr[i])  for i in range(num_features)],
        'min':  [float(min_arr[i])  for i in range(num_features)],
        'max':  [float(max_arr[i])  for i in range(num_features)],
    }


class Normalize:
    """
    Z-score normalisation: (x - mean) / (std + eps).

    If mean/std are not provided, they are computed from the first
    call to __call__ (fit-on-first-use).

    Args:
        mean: Per-feature means (list of float). None = auto-compute.
        std:  Per-feature standard deviations. None = auto-compute.
        eps:  Small constant for numerical stability.
    """

    def __init__(self, mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps
        self._fitted = mean is not None and std is not None

    def fit(self, data: List[float], num_features: int) -> "Normalize":
        """Compute mean/std from data."""
        stats = compute_stats(data, num_features)
        self.mean = stats['mean']
        self.std = stats['std']
        self._fitted = True
        return self

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        """
        Apply z-score normalisation.

        Args:
            data: Flat list, row-major [batch_size × num_features].
            num_features: Features per sample.

        Returns:
            Normalised data as list of float.
        """
        if not self._fitted:
            self.fit(data, num_features)

        n = len(data)
        batch_size = n // num_features

        data_arr = _make_f64(data)
        out_arr  = (ctypes.c_double * n)()
        mean_arr = _make_f64(self.mean)
        std_arr  = _make_f64(self.std)

        result = _lib.transform_normalize(
            _f64_ptr(data_arr), _f64_ptr(out_arr),
            batch_size, num_features,
            _f64_ptr(mean_arr), _f64_ptr(std_arr),
            self.eps,
        )
        if result != 0:
            raise RuntimeError("transform_normalize failed")

        return [float(out_arr[i]) for i in range(n)]

    def inverse(self, data: List[float], num_features: int) -> List[float]:
        """Undo normalisation (inverse transform)."""
        if not self._fitted:
            raise RuntimeError("Cannot inverse before fitting")

        n = len(data)
        batch_size = n // num_features

        data_arr = _make_f64(data)
        out_arr  = (ctypes.c_double * n)()
        mean_arr = _make_f64(self.mean)
        std_arr  = _make_f64(self.std)

        result = _lib.transform_unnormalize(
            _f64_ptr(data_arr), _f64_ptr(out_arr),
            batch_size, num_features,
            _f64_ptr(mean_arr), _f64_ptr(std_arr),
            self.eps,
        )
        if result != 0:
            raise RuntimeError("transform_unnormalize failed")

        return [float(out_arr[i]) for i in range(n)]

    def __repr__(self) -> str:
        return f"Normalize(fitted={self._fitted}, eps={self.eps})"


class MinMaxScale:
    """
    Min-max scaling to [0, 1]: (x - min) / (max - min + eps).

    Args:
        min_val: Per-feature minimums. None = auto-compute.
        max_val: Per-feature maximums. None = auto-compute.
        eps: Small constant for numerical stability.
    """

    def __init__(self, min_val: Optional[List[float]] = None,
                 max_val: Optional[List[float]] = None,
                 eps: float = 1e-8):
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps
        self._fitted = min_val is not None and max_val is not None

    def fit(self, data: List[float], num_features: int) -> "MinMaxScale":
        """Compute min/max from data."""
        stats = compute_stats(data, num_features)
        self.min_val = stats['min']
        self.max_val = stats['max']
        self._fitted = True
        return self

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        """
        Apply min-max scaling.

        Args:
            data: Flat list, row-major [batch_size × num_features].
            num_features: Features per sample.

        Returns:
            Scaled data as list of float in [0, 1].
        """
        if not self._fitted:
            self.fit(data, num_features)

        n = len(data)
        batch_size = n // num_features

        data_arr = _make_f64(data)
        out_arr  = (ctypes.c_double * n)()
        min_arr  = _make_f64(self.min_val)
        max_arr  = _make_f64(self.max_val)

        result = _lib.transform_minmax(
            _f64_ptr(data_arr), _f64_ptr(out_arr),
            batch_size, num_features,
            _f64_ptr(min_arr), _f64_ptr(max_arr),
            self.eps,
        )
        if result != 0:
            raise RuntimeError("transform_minmax failed")

        return [float(out_arr[i]) for i in range(n)]

    def __repr__(self) -> str:
        return f"MinMaxScale(fitted={self._fitted}, eps={self.eps})"


class Compose:
    """
    Compose multiple transforms into a pipeline.

    Each transform must be callable with signature:
        transform(data: List[float], num_features: int) -> List[float]

    Example:
        >>> t = Compose([Normalize(), MinMaxScale()])
        >>> out = t(data, num_features=4)
    """

    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        for t in self.transforms:
            data = t(data, num_features)
        return data

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"
