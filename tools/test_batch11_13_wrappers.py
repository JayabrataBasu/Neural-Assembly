#!/usr/bin/env python3
"""
Tests for Batch 11/12/13 Python ctypes wrappers.

Covers:
- AvgPool2D + Upsample2D wrappers
- Tensor ops wrappers (concat/split/pad/transpose)
- Attention + Transformer wrappers
"""

import math
import os
import sys
import ctypes

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyneural as pn


def assert_close(a, b, eps=1e-6, msg=""):
    if abs(a - b) > eps:
        raise AssertionError(msg or f"Expected {a} ~= {b}")


def test_pooling_wrappers():
    print("Testing AvgPool2D/Upsample2D wrappers...", end=" ")

    # Input: N=1, C=1, H=2, W=2
    # [[1,2],[3,4]]
    x = (ctypes.c_double * 4)(1.0, 2.0, 3.0, 4.0)

    avg = pn.AvgPool2D(kernel_size=2, stride=1, padding=0)
    out, oh, ow = avg.forward(x, batch=1, channels=1, in_h=2, in_w=2)
    assert oh == 1 and ow == 1
    assert_close(out[0], 2.5)

    # Backward: single grad 1.0 distributes equally over 4 inputs
    go = (ctypes.c_double * 1)(1.0)
    gi = avg.backward(go)
    for i in range(4):
        assert_close(gi[i], 0.25)

    up = pn.Upsample2D(scale_factor=2)
    up_out, uh, uw = up.forward(x, batch=1, channels=1, in_h=2, in_w=2)
    assert uh == 4 and uw == 4

    # Check corners map correctly with nearest-neighbor
    # Output 4x4 should replicate each source pixel as 2x2 block.
    assert_close(up_out[0], 1.0)        # (0,0)
    assert_close(up_out[1], 1.0)        # (0,1)
    assert_close(up_out[2], 2.0)        # (0,2)
    assert_close(up_out[10], 4.0)       # near bottom-right block

    # Backward: ones on 4x4 should sum 4x into each source pixel
    go2 = (ctypes.c_double * (4 * 4))(*([1.0] * 16))
    gi2 = up.backward(go2)
    for i in range(4):
        assert_close(gi2[i], 4.0)

    print("PASSED")


def test_tensor_ops_wrappers():
    print("Testing tensor ops wrappers...", end=" ")

    # concat axis=0
    c0, r0, k0 = pn.concat_2d([1, 2, 3, 4], [5, 6, 7, 8], 2, 2, 2, 2, axis=0)
    assert (r0, k0) == (4, 2)
    assert c0 == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    # concat axis=1
    c1, r1, k1 = pn.concat_2d([1, 2, 3, 4], [10, 20, 30, 40], 2, 2, 2, 2, axis=1)
    assert (r1, k1) == (2, 4)
    assert c1 == [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0]

    # split axis=1 at col 1
    a, ra, ca, b, rb, cb = pn.split_2d([1, 2, 3, 4, 5, 6], 2, 3, axis=1, split_index=1)
    assert (ra, ca) == (2, 1)
    assert (rb, cb) == (2, 2)
    assert a == [1.0, 4.0]
    assert b == [2.0, 3.0, 5.0, 6.0]

    # pad
    p, pr, pc = pn.pad_2d([1, 2, 3, 4], 2, 2, pad_top=1, pad_bottom=1, pad_left=1, pad_right=1, pad_value=0.0)
    assert (pr, pc) == (4, 4)
    # center block should match original
    assert p[5] == 1.0 and p[6] == 2.0 and p[9] == 3.0 and p[10] == 4.0

    # transpose
    t, tr, tc = pn.transpose_2d([1, 2, 3, 4, 5, 6], 2, 3)
    assert (tr, tc) == (3, 2)
    assert t == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

    print("PASSED")


def test_attention_transformer_wrappers():
    print("Testing attention/transformer wrappers...", end=" ")

    # Small attention test: batch=1, heads=1, seq_q=2, seq_kv=2, d_k=1, d_v=2
    q = [1.0, 0.0]                 # [2,1]
    k = [1.0, 0.0]                 # [2,1]
    v = [1.0, 2.0, 3.0, 4.0]       # [2,2]

    out, w = pn.scaled_dot_product_attention(
        q, k, v,
        batch=1, heads=1,
        seq_q=2, seq_kv=2,
        d_k=1, d_v=2,
        return_weights=True,
    )
    assert len(out) == 4
    assert len(w) == 4

    # Softmax rows sum to ~1
    assert_close(w[0] + w[1], 1.0, eps=1e-6)
    assert_close(w[2] + w[3], 1.0, eps=1e-6)

    # Transformer block smoke + finite check
    batch, seq_len, d_model, d_ff = 1, 2, 2, 4
    x = [0.1, -0.2, 0.3, 0.4]      # [1,2,2]

    # Use zeros for all projections/feedforward (valid, deterministic, finite)
    z_dm_dm = [0.0] * (d_model * d_model)
    z_dm = [0.0] * d_model
    z_dm_ff = [0.0] * (d_model * d_ff)
    z_ff_dm = [0.0] * (d_ff * d_model)
    z_ff = [0.0] * d_ff

    y = pn.transformer_block(
        x=x,
        batch=batch,
        seq_len=seq_len,
        d_model=d_model,
        d_ff=d_ff,
        w_q=z_dm_dm,
        w_k=z_dm_dm,
        w_v=z_dm_dm,
        w_o=z_dm_dm,
        b_q=z_dm,
        b_k=z_dm,
        b_v=z_dm,
        b_o=z_dm,
        w1=z_dm_ff,
        b1=z_ff,
        w2=z_ff_dm,
        b2=z_dm,
        eps=1e-5,
    )

    assert len(y) == batch * seq_len * d_model
    for val in y:
        if not math.isfinite(val):
            raise AssertionError("transformer output contains non-finite value")

    print("PASSED")


def main():
    print("=" * 64)
    print("Batch 11/12/13 Wrapper Tests")
    print("=" * 64)

    pn.init()

    tests = [
        test_pooling_wrappers,
        test_tensor_ops_wrappers,
        test_attention_transformer_wrappers,
    ]

    passed = 0
    failed = 0

    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 64)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 64)

    pn.shutdown()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
