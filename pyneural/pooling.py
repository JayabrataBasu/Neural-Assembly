"""
pyneural.pooling — AvgPool2D and nearest-neighbor Upsample2D wrappers.
"""

from __future__ import annotations

import ctypes
from typing import Tuple

from .core import _lib


class AvgPool2D:
    """C-backed AvgPool2D (float64, NCHW)."""

    def __init__(self, kernel_size, stride=None, padding: int = 0):
        if isinstance(kernel_size, int):
            self.pool_h = self.pool_w = kernel_size
        else:
            self.pool_h, self.pool_w = kernel_size

        self.stride = self.pool_h if stride is None else stride
        self.padding = padding
        self._last_shape = None

    def output_shape(self, in_h: int, in_w: int) -> Tuple[int, int]:
        oh = _lib.conv2d_output_size(in_h, self.pool_h, self.stride, self.padding)
        ow = _lib.conv2d_output_size(in_w, self.pool_w, self.stride, self.padding)
        return (oh, ow)

    def forward(self, data, batch: int, channels: int, in_h: int, in_w: int):
        oh, ow = self.output_shape(in_h, in_w)
        out_size = batch * channels * oh * ow
        out_buf = (ctypes.c_double * out_size)()

        rc = _lib.avgpool2d_forward(
            ctypes.cast(data, ctypes.c_void_p),
            batch, channels, in_h, in_w,
            self.pool_h, self.pool_w,
            self.stride, self.padding,
            ctypes.cast(out_buf, ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("avgpool2d forward failed")

        self._last_shape = (batch, channels, in_h, in_w, oh, ow)
        return out_buf, oh, ow

    def backward(self, grad_output, shape=None):
        if shape is None:
            shape = self._last_shape
        if shape is None:
            raise RuntimeError("backward called before forward")

        batch, channels, in_h, in_w, _, _ = shape
        in_size = batch * channels * in_h * in_w
        grad_in = (ctypes.c_double * in_size)()

        rc = _lib.avgpool2d_backward(
            ctypes.cast(grad_output, ctypes.c_void_p),
            batch, channels, in_h, in_w,
            self.pool_h, self.pool_w,
            self.stride, self.padding,
            ctypes.cast(grad_in, ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("avgpool2d backward failed")

        return grad_in

    def __repr__(self) -> str:
        return (
            f"AvgPool2D(kernel_size=({self.pool_h}, {self.pool_w}), "
            f"stride={self.stride}, padding={self.padding})"
        )


class Upsample2D:
    """C-backed nearest-neighbor upsampling (float64, NCHW)."""

    def __init__(self, scale_factor):
        if isinstance(scale_factor, int):
            self.scale_h = self.scale_w = scale_factor
        else:
            self.scale_h, self.scale_w = scale_factor
        self._last_shape = None

    def output_shape(self, in_h: int, in_w: int) -> Tuple[int, int]:
        return (in_h * self.scale_h, in_w * self.scale_w)

    def forward(self, data, batch: int, channels: int, in_h: int, in_w: int):
        oh, ow = self.output_shape(in_h, in_w)
        out_size = batch * channels * oh * ow
        out_buf = (ctypes.c_double * out_size)()

        rc = _lib.upsample2d_nearest_forward(
            ctypes.cast(data, ctypes.c_void_p),
            batch, channels, in_h, in_w,
            self.scale_h, self.scale_w,
            ctypes.cast(out_buf, ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("upsample2d_nearest forward failed")

        self._last_shape = (batch, channels, in_h, in_w, oh, ow)
        return out_buf, oh, ow

    def backward(self, grad_output, shape=None):
        if shape is None:
            shape = self._last_shape
        if shape is None:
            raise RuntimeError("backward called before forward")

        batch, channels, in_h, in_w, _, _ = shape
        in_size = batch * channels * in_h * in_w
        grad_in = (ctypes.c_double * in_size)()

        rc = _lib.upsample2d_nearest_backward(
            ctypes.cast(grad_output, ctypes.c_void_p),
            batch, channels, in_h, in_w,
            self.scale_h, self.scale_w,
            ctypes.cast(grad_in, ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("upsample2d_nearest backward failed")

        return grad_in

    def __repr__(self) -> str:
        return f"Upsample2D(scale_factor=({self.scale_h}, {self.scale_w}))"
