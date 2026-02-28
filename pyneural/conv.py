"""
pyneural.conv — Conv2D and MaxPool2D layers.

Thin wrappers around the C implementations in conv2d.c.
All the heavy lifting (im2col, GEMM, pooling) happens in C;
Python just marshals arguments and manages memory lifetimes.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Tuple

from .core import _lib


# ── Conv2D ───────────────────────────────────────────────────────────

class Conv2D:
    """
    2D convolution layer (C-backed, im2col + GEMM).

    Data layout is NCHW: [batch, channels, height, width].
    All computation uses float64.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size:  (kH, kW) or a single int for square kernels.
        stride:       Convolution stride (default 1).
        padding:      Zero-padding added to both sides (default 0).
        bias:         Whether to include a learnable bias (default True).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride: int = 1,
                 padding: int = 0, bias: bool = True):
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        self._ptr = _lib.conv2d_layer_create(
            in_channels, out_channels, kh, kw, stride, padding,
            1 if bias else 0,
        )
        if not self._ptr:
            raise MemoryError("Failed to create Conv2D layer")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Cache from last forward for backward
        self._last_input = None
        self._last_shape = None

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.conv2d_layer_free(self._ptr)
            self._ptr = None

    def output_shape(self, in_h: int, in_w: int) -> Tuple[int, int]:
        """Compute (out_h, out_w) for given spatial input size."""
        oh = _lib.conv2d_layer_out_h(self._ptr, in_h)
        ow = _lib.conv2d_layer_out_w(self._ptr, in_w)
        return (oh, ow)

    def forward(self, data, batch: int, in_h: int, in_w: int):
        """
        Forward pass.

        Args:
            data:   ctypes pointer or (c_double * N) buffer,
                    NCHW layout [batch, in_c, in_h, in_w].
            batch:  Batch size.
            in_h:   Input height.
            in_w:   Input width.

        Returns:
            (output_buf, out_h, out_w) — output is a ctypes double array.
        """
        oh, ow = self.output_shape(in_h, in_w)
        out_size = batch * self.out_channels * oh * ow
        out_buf = (ctypes.c_double * out_size)()

        inp_ptr = ctypes.cast(data, ctypes.c_void_p)
        out_ptr = ctypes.cast(out_buf, ctypes.c_void_p)

        rc = _lib.conv2d_layer_forward(self._ptr, inp_ptr, batch, in_h, in_w, out_ptr)
        if rc != 0:
            raise RuntimeError("conv2d forward failed")

        # Stash for backward
        self._last_input = data
        self._last_shape = (batch, in_h, in_w, oh, ow)

        return out_buf, oh, ow

    def backward(self, grad_output, input_data=None, shape=None):
        """
        Backward pass — computes grad_input, grad_weight, grad_bias.

        Args:
            grad_output:  ctypes buffer [batch, out_c, out_h, out_w].
            input_data:   Original input (defaults to cached from forward).
            shape:        (batch, in_h, in_w) if not using cached values.

        Returns:
            (grad_input, grad_weight, grad_bias) as ctypes arrays.
        """
        if input_data is None:
            input_data = self._last_input
        if shape is None and self._last_shape:
            batch, in_h, in_w = self._last_shape[0], self._last_shape[1], self._last_shape[2]
        else:
            batch, in_h, in_w = shape

        in_size = batch * self.in_channels * in_h * in_w
        wt_size = _lib.conv2d_layer_weight_size(self._ptr)

        grad_in = (ctypes.c_double * in_size)()
        grad_wt = (ctypes.c_double * wt_size)()
        grad_bi = (ctypes.c_double * self.out_channels)() if self.has_bias else None

        rc = _lib.conv2d_layer_backward(
            self._ptr,
            ctypes.cast(input_data, ctypes.c_void_p),
            ctypes.cast(grad_output, ctypes.c_void_p),
            batch, in_h, in_w,
            ctypes.cast(grad_in, ctypes.c_void_p),
            ctypes.cast(grad_wt, ctypes.c_void_p),
            ctypes.cast(grad_bi, ctypes.c_void_p) if grad_bi else None,
        )
        if rc != 0:
            raise RuntimeError("conv2d backward failed")

        return grad_in, grad_wt, grad_bi

    @property
    def weight_ptr(self):
        """Raw pointer to the weight buffer (double*)."""
        return _lib.conv2d_layer_weight(self._ptr)

    @property
    def bias_ptr(self):
        """Raw pointer to the bias buffer (double*), or None."""
        if not self.has_bias:
            return None
        return _lib.conv2d_layer_bias(self._ptr)

    @property
    def weight_size(self) -> int:
        """Total number of weight parameters."""
        return _lib.conv2d_layer_weight_size(self._ptr)

    def __repr__(self) -> str:
        return (
            f"Conv2D({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding})"
        )


# ── MaxPool2D ────────────────────────────────────────────────────────

class MaxPool2D:
    """
    2D max pooling (C-backed).

    Stores the argmax indices from the forward pass for use in backward.

    Args:
        kernel_size: (pH, pW) or a single int for square pooling.
        stride:      Pooling stride.  Defaults to kernel_size.
        padding:     Zero-padding (default 0).
    """

    def __init__(self, kernel_size, stride=None, padding: int = 0):
        if isinstance(kernel_size, int):
            self.pool_h = self.pool_w = kernel_size
        else:
            self.pool_h, self.pool_w = kernel_size

        if stride is None:
            self.stride = self.pool_h  # standard default
        else:
            self.stride = stride

        self.padding = padding

        # Cached from forward for backward use
        self._mask = None
        self._last_shape = None

    def output_shape(self, in_h: int, in_w: int) -> Tuple[int, int]:
        """Compute (out_h, out_w) for given spatial input."""
        oh = _lib.conv2d_output_size(in_h, self.pool_h, self.stride, self.padding)
        ow = _lib.conv2d_output_size(in_w, self.pool_w, self.stride, self.padding)
        return (oh, ow)

    def forward(self, data, batch: int, channels: int,
                in_h: int, in_w: int):
        """
        Forward pass.

        Args:
            data:     ctypes buffer, NCHW [batch, channels, in_h, in_w].
            batch:    Batch size.
            channels: Number of channels.
            in_h:     Input height.
            in_w:     Input width.

        Returns:
            (output_buf, out_h, out_w).
        """
        oh, ow = self.output_shape(in_h, in_w)
        out_size = batch * channels * oh * ow
        out_buf = (ctypes.c_double * out_size)()
        mask_buf = (ctypes.c_int64 * out_size)()

        rc = _lib.maxpool2d_forward(
            ctypes.cast(data, ctypes.c_void_p),
            batch, channels, in_h, in_w,
            self.pool_h, self.pool_w, self.stride, self.padding,
            ctypes.cast(out_buf, ctypes.c_void_p),
            ctypes.cast(mask_buf, ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("maxpool2d forward failed")

        self._mask = mask_buf
        self._last_shape = (batch, channels, in_h, in_w, oh, ow)

        return out_buf, oh, ow

    def backward(self, grad_output, mask=None, shape=None):
        """
        Backward pass — scatter gradients back through max indices.

        Args:
            grad_output: ctypes buffer [batch, channels, out_h, out_w].
            mask:        Argmax indices (defaults to cached from forward).
            shape:       (batch, channels, in_h, in_w, out_h, out_w).

        Returns:
            grad_input as a ctypes double array.
        """
        if mask is None:
            mask = self._mask
        if shape is None:
            shape = self._last_shape
        if mask is None or shape is None:
            raise RuntimeError("backward called before forward")

        batch, channels, in_h, in_w, oh, ow = shape
        in_size = batch * channels * in_h * in_w
        grad_in = (ctypes.c_double * in_size)()

        rc = _lib.maxpool2d_backward(
            ctypes.cast(grad_output, ctypes.c_void_p),
            ctypes.cast(mask, ctypes.c_void_p),
            batch, channels, in_h, in_w, oh, ow,
            ctypes.cast(grad_in, ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError("maxpool2d backward failed")

        return grad_in

    def __repr__(self) -> str:
        return (
            f"MaxPool2D(kernel_size=({self.pool_h}, {self.pool_w}), "
            f"stride={self.stride}, padding={self.padding})"
        )


# ── Utility ──────────────────────────────────────────────────────────

def calc_output_size(input_dim: int, kernel_dim: int,
                     stride: int = 1, padding: int = 0) -> int:
    """Compute output spatial dimension for conv or pool."""
    return _lib.conv2d_output_size(input_dim, kernel_dim, stride, padding)
