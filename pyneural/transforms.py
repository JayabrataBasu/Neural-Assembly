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


# ---------------------------------------------------------------------------
# Image Augmentation Transforms
# ---------------------------------------------------------------------------

import math
import random as _random


class ToTensor:
    """
    Convert a flat list of pixel values to a normalized float list.

    If values are in [0, 255], scales them to [0.0, 1.0].
    """

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        max_val = max(data) if data else 1.0
        if max_val > 1.0:
            return [v / 255.0 for v in data]
        return list(data)

    def __repr__(self) -> str:
        return "ToTensor()"


class RandomHorizontalFlip:
    """
    Randomly flip an image horizontally.

    Operates on a flat pixel list assumed to be (C, H, W) layout.

    Args:
        p: Probability of flipping (default: 0.5).
        height: Image height.
        width: Image width.
        channels: Number of channels (default: 1).
    """

    def __init__(self, p: float = 0.5, height: int = 28, width: int = 28,
                 channels: int = 1):
        self.p = p
        self.height = height
        self.width = width
        self.channels = channels

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        if _random.random() >= self.p:
            return data

        result = list(data)
        for c in range(self.channels):
            for h in range(self.height):
                for w in range(self.width // 2):
                    idx1 = c * self.height * self.width + h * self.width + w
                    idx2 = c * self.height * self.width + h * self.width + (self.width - 1 - w)
                    result[idx1], result[idx2] = result[idx2], result[idx1]
        return result

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """
    Randomly flip an image vertically.

    Args:
        p: Probability of flipping (default: 0.5).
        height: Image height.
        width: Image width.
        channels: Number of channels (default: 1).
    """

    def __init__(self, p: float = 0.5, height: int = 28, width: int = 28,
                 channels: int = 1):
        self.p = p
        self.height = height
        self.width = width
        self.channels = channels

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        if _random.random() >= self.p:
            return data

        result = list(data)
        for c in range(self.channels):
            for h in range(self.height // 2):
                for w in range(self.width):
                    idx1 = c * self.height * self.width + h * self.width + w
                    idx2 = c * self.height * self.width + (self.height - 1 - h) * self.width + w
                    result[idx1], result[idx2] = result[idx2], result[idx1]
        return result

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomCrop:
    """
    Randomly crop an image and pad back to original size.

    Args:
        height: Image height.
        width: Image width.
        padding: Number of pixels to pad on each side.
        channels: Number of channels (default: 1).
        fill: Fill value for padding (default: 0.0).
    """

    def __init__(self, height: int = 28, width: int = 28, padding: int = 4,
                 channels: int = 1, fill: float = 0.0):
        self.height = height
        self.width = width
        self.padding = padding
        self.channels = channels
        self.fill = fill

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        pad = self.padding
        h, w, c = self.height, self.width, self.channels
        padded_h = h + 2 * pad
        padded_w = w + 2 * pad

        # Create padded image
        padded = [self.fill] * (c * padded_h * padded_w)
        for ch in range(c):
            for row in range(h):
                for col in range(w):
                    src = ch * h * w + row * w + col
                    dst = ch * padded_h * padded_w + (row + pad) * padded_w + (col + pad)
                    padded[dst] = data[src]

        # Random crop from padded
        top = _random.randint(0, 2 * pad)
        left = _random.randint(0, 2 * pad)

        result = [0.0] * (c * h * w)
        for ch in range(c):
            for row in range(h):
                for col in range(w):
                    src = ch * padded_h * padded_w + (top + row) * padded_w + (left + col)
                    dst = ch * h * w + row * w + col
                    result[dst] = padded[src]

        return result

    def __repr__(self) -> str:
        return f"RandomCrop(padding={self.padding})"


class RandomRotation:
    """
    Randomly rotate an image by a small angle.

    Uses bilinear interpolation. Operates on (C, H, W) flat layout.

    Args:
        degrees: Maximum rotation angle in degrees (± degrees).
        height: Image height.
        width: Image width.
        channels: Number of channels (default: 1).
        fill: Fill value for empty pixels after rotation.
    """

    def __init__(self, degrees: float = 15.0, height: int = 28, width: int = 28,
                 channels: int = 1, fill: float = 0.0):
        self.degrees = degrees
        self.height = height
        self.width = width
        self.channels = channels
        self.fill = fill

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        angle = _random.uniform(-self.degrees, self.degrees)
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        h, w, c = self.height, self.width, self.channels
        cx, cy = w / 2.0, h / 2.0
        result = [self.fill] * (c * h * w)

        for ch in range(c):
            for row in range(h):
                for col in range(w):
                    # Map destination to source (inverse rotation)
                    dx = col - cx
                    dy = row - cy
                    src_x = cos_a * dx + sin_a * dy + cx
                    src_y = -sin_a * dx + cos_a * dy + cy

                    # Bilinear interpolation
                    x0 = int(math.floor(src_x))
                    y0 = int(math.floor(src_y))
                    x1 = x0 + 1
                    y1 = y0 + 1
                    fx = src_x - x0
                    fy = src_y - y0

                    def _pixel(r, cc):
                        if 0 <= r < h and 0 <= cc < w:
                            return data[ch * h * w + r * w + cc]
                        return self.fill

                    val = (
                        _pixel(y0, x0) * (1 - fx) * (1 - fy) +
                        _pixel(y0, x1) * fx * (1 - fy) +
                        _pixel(y1, x0) * (1 - fx) * fy +
                        _pixel(y1, x1) * fx * fy
                    )
                    result[ch * h * w + row * w + col] = val

        return result

    def __repr__(self) -> str:
        return f"RandomRotation(degrees={self.degrees})"


class RandomNoise:
    """
    Add random Gaussian noise to data.

    Args:
        std: Standard deviation of the noise.
    """

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        return [v + _random.gauss(0, self.std) for v in data]

    def __repr__(self) -> str:
        return f"RandomNoise(std={self.std})"


class RandomErasing:
    """
    Randomly erase a rectangular region of the image (Zhong et al., 2020).

    Args:
        p: Probability of erasing.
        scale: Range of proportion of image area to erase.
        ratio: Range of aspect ratio of erased region.
        height: Image height.
        width: Image width.
        channels: Number of channels.
        fill: Fill value for erased region.
    """

    def __init__(self, p: float = 0.5, scale: tuple = (0.02, 0.33),
                 ratio: tuple = (0.3, 3.3), height: int = 28, width: int = 28,
                 channels: int = 1, fill: float = 0.0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.height = height
        self.width = width
        self.channels = channels
        self.fill = fill

    def __call__(self, data: List[float], num_features: int) -> List[float]:
        if _random.random() >= self.p:
            return data

        h, w = self.height, self.width
        area = h * w
        result = list(data)

        for _ in range(10):  # try up to 10 times
            target_area = _random.uniform(self.scale[0], self.scale[1]) * area
            aspect = _random.uniform(self.ratio[0], self.ratio[1])
            eh = int(round(math.sqrt(target_area * aspect)))
            ew = int(round(math.sqrt(target_area / aspect)))
            if eh < h and ew < w:
                top = _random.randint(0, h - eh)
                left = _random.randint(0, w - ew)
                for c in range(self.channels):
                    for r in range(top, top + eh):
                        for col in range(left, left + ew):
                            idx = c * h * w + r * w + col
                            result[idx] = self.fill
                return result

        return result

    def __repr__(self) -> str:
        return f"RandomErasing(p={self.p})"
