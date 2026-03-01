"""
pyneural.attention — ctypes wrappers for attention and transformer block.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Tuple

from .core import _lib


def _make_f64(values: List[float]):
    return (ctypes.c_double * len(values))(*values)


def _to_list(arr, n: int) -> List[float]:
    return [float(arr[i]) for i in range(n)]


def scaled_dot_product_attention(
    q: List[float],
    k: List[float],
    v: List[float],
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    d_k: int,
    d_v: int,
    mask: Optional[List[float]] = None,
    return_weights: bool = False,
):
    """
    Run scaled dot-product attention.

    Flat layout convention for q/k/v/out:
      [batch, heads, seq, dim]
    Flat layout for optional mask/weights:
      [batch, heads, seq_q, seq_kv]
    """
    q_arr = _make_f64(q)
    k_arr = _make_f64(k)
    v_arr = _make_f64(v)

    out_n = batch * heads * seq_q * d_v
    out_arr = (ctypes.c_double * out_n)()

    mask_arr = _make_f64(mask) if mask is not None else None
    w_n = batch * heads * seq_q * seq_kv
    w_arr = (ctypes.c_double * w_n)() if return_weights else None

    rc = _lib.attention_scaled_dot_product(
        ctypes.cast(q_arr, ctypes.c_void_p),
        ctypes.cast(k_arr, ctypes.c_void_p),
        ctypes.cast(v_arr, ctypes.c_void_p),
        ctypes.cast(mask_arr, ctypes.c_void_p) if mask_arr is not None else None,
        batch, heads, seq_q, seq_kv, d_k, d_v,
        ctypes.cast(out_arr, ctypes.c_void_p),
        ctypes.cast(w_arr, ctypes.c_void_p) if w_arr is not None else None,
    )
    if rc != 0:
        raise RuntimeError("attention_scaled_dot_product failed")

    out = _to_list(out_arr, out_n)
    if not return_weights:
        return out
    return out, _to_list(w_arr, w_n)


def transformer_block(
    x: List[float],
    batch: int,
    seq_len: int,
    d_model: int,
    d_ff: int,
    w_q: List[float],
    w_k: List[float],
    w_v: List[float],
    w_o: List[float],
    b_q: Optional[List[float]] = None,
    b_k: Optional[List[float]] = None,
    b_v: Optional[List[float]] = None,
    b_o: Optional[List[float]] = None,
    w1: Optional[List[float]] = None,
    b1: Optional[List[float]] = None,
    w2: Optional[List[float]] = None,
    b2: Optional[List[float]] = None,
    eps: float = 1e-5,
) -> List[float]:
    """
    Run a single-head Transformer encoder block forward pass.

    Weight shapes expected (flat row-major):
      w_q, w_k, w_v, w_o: [d_model, d_model]
      w1: [d_model, d_ff]
      w2: [d_ff, d_model]
      biases: [d_model] for q/k/v/o and b2, [d_ff] for b1
    """
    if w1 is None or w2 is None:
        raise ValueError("w1 and w2 are required")

    x_arr = _make_f64(x)
    wq_arr = _make_f64(w_q)
    wk_arr = _make_f64(w_k)
    wv_arr = _make_f64(w_v)
    wo_arr = _make_f64(w_o)
    w1_arr = _make_f64(w1)
    w2_arr = _make_f64(w2)

    bq_arr = _make_f64(b_q) if b_q is not None else None
    bk_arr = _make_f64(b_k) if b_k is not None else None
    bv_arr = _make_f64(b_v) if b_v is not None else None
    bo_arr = _make_f64(b_o) if b_o is not None else None
    b1_arr = _make_f64(b1) if b1 is not None else None
    b2_arr = _make_f64(b2) if b2 is not None else None

    out_n = batch * seq_len * d_model
    out_arr = (ctypes.c_double * out_n)()

    rc = _lib.transformer_block_forward(
        ctypes.cast(x_arr, ctypes.c_void_p),
        batch, seq_len, d_model, d_ff,
        ctypes.cast(wq_arr, ctypes.c_void_p),
        ctypes.cast(wk_arr, ctypes.c_void_p),
        ctypes.cast(wv_arr, ctypes.c_void_p),
        ctypes.cast(wo_arr, ctypes.c_void_p),
        ctypes.cast(bq_arr, ctypes.c_void_p) if bq_arr is not None else None,
        ctypes.cast(bk_arr, ctypes.c_void_p) if bk_arr is not None else None,
        ctypes.cast(bv_arr, ctypes.c_void_p) if bv_arr is not None else None,
        ctypes.cast(bo_arr, ctypes.c_void_p) if bo_arr is not None else None,
        ctypes.cast(w1_arr, ctypes.c_void_p),
        ctypes.cast(b1_arr, ctypes.c_void_p) if b1_arr is not None else None,
        ctypes.cast(w2_arr, ctypes.c_void_p),
        ctypes.cast(b2_arr, ctypes.c_void_p) if b2_arr is not None else None,
        float(eps),
        ctypes.cast(out_arr, ctypes.c_void_p),
    )
    if rc != 0:
        raise RuntimeError("transformer_block_forward failed")

    return _to_list(out_arr, out_n)
