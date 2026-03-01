"""
pyneural.tensor_ops — C-backed tensor utility operations.

Current wrappers target the Batch 12 2D APIs:
  - concat
  - split
  - pad
  - transpose

All buffers are float64 and represented as flat row-major lists.
"""

from __future__ import annotations

import ctypes
from typing import List, Tuple

from .core import _lib


def _make_f64(values: List[float]):
    return (ctypes.c_double * len(values))(*values)


def _to_list(arr, n: int) -> List[float]:
    return [float(arr[i]) for i in range(n)]


def concat_2d(a: List[float], b: List[float],
              rows_a: int, cols_a: int,
              rows_b: int, cols_b: int,
              axis: int = 0) -> Tuple[List[float], int, int]:
    """Concatenate two 2D arrays along `axis` (0 or 1)."""
    a_arr = _make_f64(a)
    b_arr = _make_f64(b)

    max_rows = rows_a + rows_b
    max_cols = cols_a + cols_b
    out_cap = max_rows * max_cols
    out_arr = (ctypes.c_double * out_cap)()

    out_rows = ctypes.c_int64()
    out_cols = ctypes.c_int64()

    rc = _lib.tensor_concat_2d(
        ctypes.cast(a_arr, ctypes.c_void_p),
        ctypes.cast(b_arr, ctypes.c_void_p),
        rows_a, cols_a,
        rows_b, cols_b,
        axis,
        ctypes.cast(out_arr, ctypes.c_void_p),
        ctypes.byref(out_rows),
        ctypes.byref(out_cols),
    )
    if rc != 0:
        raise RuntimeError("tensor_concat_2d failed")

    n = out_rows.value * out_cols.value
    return _to_list(out_arr, n), out_rows.value, out_cols.value


def split_2d(data: List[float], rows: int, cols: int,
             axis: int, split_index: int):
    """Split a 2D array into two outputs along `axis` at `split_index`."""
    in_arr = _make_f64(data)

    out_a = (ctypes.c_double * (rows * cols))()
    out_b = (ctypes.c_double * (rows * cols))()

    rows_a = ctypes.c_int64()
    cols_a = ctypes.c_int64()
    rows_b = ctypes.c_int64()
    cols_b = ctypes.c_int64()

    rc = _lib.tensor_split_2d(
        ctypes.cast(in_arr, ctypes.c_void_p),
        rows, cols, axis, split_index,
        ctypes.cast(out_a, ctypes.c_void_p),
        ctypes.cast(out_b, ctypes.c_void_p),
        ctypes.byref(rows_a), ctypes.byref(cols_a),
        ctypes.byref(rows_b), ctypes.byref(cols_b),
    )
    if rc != 0:
        raise RuntimeError("tensor_split_2d failed")

    n_a = rows_a.value * cols_a.value
    n_b = rows_b.value * cols_b.value

    return (
        _to_list(out_a, n_a), rows_a.value, cols_a.value,
        _to_list(out_b, n_b), rows_b.value, cols_b.value,
    )


def pad_2d(data: List[float], in_rows: int, in_cols: int,
           pad_top: int = 0, pad_bottom: int = 0,
           pad_left: int = 0, pad_right: int = 0,
           pad_value: float = 0.0) -> Tuple[List[float], int, int]:
    """Pad a 2D array with constant value."""
    in_arr = _make_f64(data)

    out_rows_max = in_rows + pad_top + pad_bottom
    out_cols_max = in_cols + pad_left + pad_right
    out_arr = (ctypes.c_double * (out_rows_max * out_cols_max))()

    out_rows = ctypes.c_int64()
    out_cols = ctypes.c_int64()

    rc = _lib.tensor_pad_2d(
        ctypes.cast(in_arr, ctypes.c_void_p),
        in_rows, in_cols,
        pad_top, pad_bottom, pad_left, pad_right,
        float(pad_value),
        ctypes.cast(out_arr, ctypes.c_void_p),
        ctypes.byref(out_rows), ctypes.byref(out_cols),
    )
    if rc != 0:
        raise RuntimeError("tensor_pad_2d failed")

    n = out_rows.value * out_cols.value
    return _to_list(out_arr, n), out_rows.value, out_cols.value


def transpose_2d(data: List[float], rows: int, cols: int) -> Tuple[List[float], int, int]:
    """Transpose a 2D array (rows, cols) -> (cols, rows)."""
    in_arr = _make_f64(data)
    out_arr = (ctypes.c_double * (rows * cols))()

    rc = _lib.tensor_transpose2d_array(
        ctypes.cast(in_arr, ctypes.c_void_p),
        rows, cols,
        ctypes.cast(out_arr, ctypes.c_void_p),
    )
    if rc != 0:
        raise RuntimeError("tensor_transpose2d_array failed")

    return _to_list(out_arr, rows * cols), cols, rows
