#!/usr/bin/env python3
"""
Test suite for Conv2D and MaxPool2D layers.

These are both backed by C (conv2d.c) with thin Python wrappers.
Covers construction, forward/backward numerics, pooling, edge cases,
utility functions, and repr strings.

~50 tests.
"""

import sys, os, ctypes, math

# Make sure we can import pyneural from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyneural import Conv2D, MaxPool2D, calc_output_size
from pyneural.core import _lib

passed = 0
failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


def approx(a, b, tol=1e-9):
    """Good enough for doubles."""
    return abs(a - b) < tol


# Helper: create a ctypes double array from a flat Python list
def make_buf(values):
    buf = (ctypes.c_double * len(values))(*values)
    return buf


# Helper: read values from a raw c_void_p into a list of doubles
def read_ptr(ptr, count):
    dp = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    return [dp[i] for i in range(count)]


# Helper: write values into a raw c_void_p
def write_ptr(ptr, values):
    dp = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    for i, v in enumerate(values):
        dp[i] = v


# Helper: read ctypes array into a list
def to_list(buf, count=None):
    if count is None:
        count = len(buf)
    return [buf[i] for i in range(count)]


# ── Section 1: calc_output_size ──────────────────────────────────

print("--- calc_output_size ---")

check(calc_output_size(5, 3, 1, 0) == 3,
      "5x5 input, 3x3 kernel, stride=1, pad=0 → 3")

check(calc_output_size(5, 3, 1, 1) == 5,
      "5x5 input, 3x3 kernel, stride=1, pad=1 → 5 (same padding)")

check(calc_output_size(7, 3, 2, 0) == 3,
      "7x7 input, 3x3 kernel, stride=2, pad=0 → 3")

check(calc_output_size(28, 5, 1, 2) == 28,
      "28x28 input, 5x5 kernel, stride=1, pad=2 → 28")

check(calc_output_size(4, 2, 2, 0) == 2,
      "4x4 input, 2x2 kernel, stride=2 → 2")

check(calc_output_size(1, 1, 1, 0) == 1,
      "1x1 input, 1x1 kernel → 1")

check(calc_output_size(3, 5, 1, 0) == -1,
      "input smaller than kernel → -1 (invalid)")


# ── Section 2: Conv2D construction ───────────────────────────────

print("\n--- Conv2D construction ---")

c1 = Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
check(c1.in_channels == 3, "in_channels stored")
check(c1.out_channels == 16, "out_channels stored")
check(c1.kernel_size == (3, 3), "kernel_size tuple from int")
check(c1.stride == 1, "stride stored")
check(c1.padding == 1, "padding stored")
check(c1.has_bias is True, "bias defaults to True")

c2 = Conv2D(1, 8, kernel_size=(5, 3), stride=2, padding=0, bias=False)
check(c2.kernel_size == (5, 3), "rectangular kernel from tuple")
check(c2.has_bias is False, "bias=False honoured")
check(c2.bias_ptr is None, "bias_ptr returns None when no bias")

# Weight size check
check(c1.weight_size == 16 * 3 * 3 * 3, "weight_size = oc*ic*kh*kw")
check(c2.weight_size == 8 * 1 * 5 * 3, "rectangular weight_size")

# Weight pointer is not null
check(c1.weight_ptr is not None and c1.weight_ptr != 0,
      "weight_ptr is valid")

# repr
rep = repr(c1)
check("Conv2D" in rep and "16" in rep and "(3, 3)" in rep,
      "repr contains key info")

del c2  # make sure destructor doesn't crash


# ── Section 3: Conv2D output_shape ───────────────────────────────

print("\n--- Conv2D output_shape ---")

# c1: 3→16, 3x3 kernel, stride=1, pad=1  →  same spatial size
oh, ow = c1.output_shape(8, 8)
check(oh == 8 and ow == 8, "pad=1 stride=1 3x3 → same size (8x8)")

oh, ow = c1.output_shape(32, 32)
check(oh == 32 and ow == 32, "pad=1 stride=1 3x3 → same size (32x32)")

c3 = Conv2D(1, 4, kernel_size=3, stride=2, padding=0)
oh, ow = c3.output_shape(7, 7)
check(oh == 3 and ow == 3, "7x7, k=3, s=2, p=0 → 3x3")

# 1x1 conv — should preserve spatial dims
c_1x1 = Conv2D(8, 16, kernel_size=1)
oh, ow = c_1x1.output_shape(10, 10)
check(oh == 10 and ow == 10, "1x1 conv preserves spatial dims")


# ── Section 4: Conv2D forward numerics ───────────────────────────

print("\n--- Conv2D forward (numerical) ---")

# Tiny case: 1 channel, 1 filter, 2x2 kernel, no padding, stride=1
# Input (1, 1, 3, 3):
#   1 2 3
#   4 5 6
#   7 8 9
# Kernel: [[w0, w1], [w2, w3]]  — we'll set to [[1, 0], [0, 1]]
# Expected output (1, 1, 2, 2):
#   1*1+2*0+4*0+5*1 = 6   |   2*1+3*0+5*0+6*1 = 8
#   4*1+5*0+7*0+8*1 = 12  |   5*1+6*0+8*0+9*1 = 14

conv_tiny = Conv2D(1, 1, kernel_size=2, stride=1, padding=0, bias=True)

# Set weight to [1, 0, 0, 1] and bias to [0]
write_ptr(conv_tiny.weight_ptr, [1.0, 0.0, 0.0, 1.0])
bp = conv_tiny.bias_ptr
write_ptr(bp, [0.0])

inp = make_buf([1, 2, 3, 4, 5, 6, 7, 8, 9])
out, oh, ow = conv_tiny.forward(inp, batch=1, in_h=3, in_w=3)

check(oh == 2 and ow == 2, "tiny conv: output shape is 2x2")

vals = to_list(out)
check(approx(vals[0], 6.0) and approx(vals[1], 8.0) and
      approx(vals[2], 12.0) and approx(vals[3], 14.0),
      "tiny conv: correct output values [6, 8, 12, 14]")

# Now add a bias of 0.5
write_ptr(bp, [0.5])
out2, _, _ = conv_tiny.forward(inp, batch=1, in_h=3, in_w=3)
vals2 = to_list(out2)
check(approx(vals2[0], 6.5) and approx(vals2[3], 14.5),
      "tiny conv with bias=0.5: output shifted correctly")


# ── Section 5: Conv2D forward with padding ───────────────────────

print("\n--- Conv2D forward (with padding) ---")

# 1 channel, 1 filter, 3x3 kernel, pad=1, stride=1 on 3x3 input
# Should give 3x3 output (same padding)
conv_pad = Conv2D(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

# Set all-ones kernel: every output is sum of the 3x3 neighbourhood
write_ptr(conv_pad.weight_ptr, [1.0] * 9)

inp3 = make_buf([1, 1, 1, 1, 1, 1, 1, 1, 1])  # 3x3 of ones
out3, oh3, ow3 = conv_pad.forward(inp3, batch=1, in_h=3, in_w=3)

check(oh3 == 3 and ow3 == 3, "padded conv: output is 3x3")

vals3 = to_list(out3)
# Center pixel sees all 9 ones → 9
# Corner pixel (0,0) sees 4 ones + 5 zeros → 4
# Edge pixel (0,1) sees 6 ones + 3 zeros → 6
check(approx(vals3[4], 9.0), "padded conv: centre = 9 (sees all 9)")
check(approx(vals3[0], 4.0), "padded conv: corner = 4")
check(approx(vals3[1], 6.0), "padded conv: edge = 6")


# ── Section 6: Conv2D forward multi-channel / multi-filter ───────

print("\n--- Conv2D forward (multi-channel) ---")

# 2 input channels, 2 output channels, 1x1 kernel, no padding
# This is basically a matrix multiply across channels at each pixel.
conv_mc = Conv2D(2, 2, kernel_size=1, stride=1, padding=0, bias=False)

# Weight layout: out_c × (in_c * kh * kw) = 2 × 2
# Filter 0: [1, 0]  → just picks channel 0
# Filter 1: [0, 1]  → just picks channel 1
write_ptr(conv_mc.weight_ptr, [1.0, 0.0, 0.0, 1.0])

# Input: batch=1, 2 channels, 2x2 each
# ch0: [[1, 2], [3, 4]]
# ch1: [[5, 6], [7, 8]]
inp_mc = make_buf([1, 2, 3, 4, 5, 6, 7, 8])
out_mc, oh_mc, ow_mc = conv_mc.forward(inp_mc, batch=1, in_h=2, in_w=2)

check(oh_mc == 2 and ow_mc == 2, "1x1 multi-channel: output 2x2")

vals_mc = to_list(out_mc)
# Filter 0 output = channel 0: [1,2,3,4]
# Filter 1 output = channel 1: [5,6,7,8]
check(approx(vals_mc[0], 1.0) and approx(vals_mc[3], 4.0),
      "1x1 conv filter 0 picks channel 0")
check(approx(vals_mc[4], 5.0) and approx(vals_mc[7], 8.0),
      "1x1 conv filter 1 picks channel 1")


# ── Section 7: Conv2D forward batch > 1 ─────────────────────────

print("\n--- Conv2D forward (batched) ---")

# Reuse conv_tiny (1→1, 2x2 kernel = [1,0,0,1], bias=0)
write_ptr(conv_tiny.bias_ptr, [0.0])
write_ptr(conv_tiny.weight_ptr, [1.0, 0.0, 0.0, 1.0])

# Batch of 2: first sample = [1..9], second = [10..18]
inp_b = make_buf(list(range(1, 10)) + list(range(10, 19)))
out_b, oh_b, ow_b = conv_tiny.forward(inp_b, batch=2, in_h=3, in_w=3)

vals_b = to_list(out_b)
# Sample 0: [6, 8, 12, 14]
# Sample 1: [10+15, 11+16, 13+18, 14+19] = [25, 27, 31, 33]
# Actually let me recalculate. Second sample input:
#  10 11 12
#  13 14 15
#  16 17 18
# Kernel [1,0;0,1]:
#  out[0,0] = 10*1 + 11*0 + 13*0 + 14*1 = 24
#  out[0,1] = 11*1 + 12*0 + 14*0 + 15*1 = 26
#  out[1,0] = 13*1 + 14*0 + 16*0 + 17*1 = 30
#  out[1,1] = 14*1 + 15*0 + 17*0 + 18*1 = 32
check(approx(vals_b[0], 6.0) and approx(vals_b[3], 14.0),
      "batch: sample 0 correct")
check(approx(vals_b[4], 24.0) and approx(vals_b[7], 32.0),
      "batch: sample 1 correct")


# ── Section 8: Conv2D backward (numerical gradient check) ───────

print("\n--- Conv2D backward ---")

# Use the tiny conv: 1→1, 2x2 kernel=[1,0,0,1], bias=0
# Input: [1..9] → output: [6, 8, 12, 14]
# grad_output = all ones [1, 1, 1, 1]
write_ptr(conv_tiny.weight_ptr, [1.0, 0.0, 0.0, 1.0])
write_ptr(conv_tiny.bias_ptr, [0.0])

inp_bk = make_buf([1, 2, 3, 4, 5, 6, 7, 8, 9])
# Do forward first (to cache input)
conv_tiny.forward(inp_bk, batch=1, in_h=3, in_w=3)

grad_out = make_buf([1.0, 1.0, 1.0, 1.0])  # ones
grad_in, grad_wt, grad_bi = conv_tiny.backward(grad_out)

# grad_bias: sum of grad_output = 4.0
gi_bi = to_list(grad_bi)
check(approx(gi_bi[0], 4.0), "backward: grad_bias = sum(grad_out) = 4")

# grad_weight: grad_out_reshaped @ col^T
# For kernel [1,0,0,1] and grad_out all ones:
# weight has shape (1, 4) = (out_c, in_c*kh*kw)
# grad_weight[j] = sum over spatial of (grad_out[s] * col[j, s])
# The im2col columns for input [1..9] with k=2,s=1,p=0:
#   Patch at (0,0): 1, 2, 4, 5
#   Patch at (0,1): 2, 3, 5, 6
#   Patch at (1,0): 4, 5, 7, 8
#   Patch at (1,1): 5, 6, 8, 9
# So col (rows = in_c*kh*kw=4, cols = spatial_out=4):
#   row 0: 1, 2, 4, 5
#   row 1: 2, 3, 5, 6
#   row 2: 4, 5, 7, 8
#   row 3: 5, 6, 8, 9
# grad_weight = grad_out (1×4 = [1,1,1,1]) @ col^T
# = [1,1,1,1] @ col  (sum each col row)
# = [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9] = [12, 16, 24, 28]
gi_wt = to_list(grad_wt)
check(approx(gi_wt[0], 12.0) and approx(gi_wt[1], 16.0) and
      approx(gi_wt[2], 24.0) and approx(gi_wt[3], 28.0),
      "backward: grad_weight values are correct")

# grad_input: col2im(weight^T @ grad_out_reshaped)
# weight^T (4×1) @ grad_out (1×4) = outer product → 4×4 matrix
# weight = [1,0,0,1], so weight^T = [[1],[0],[0],[1]]
# col_scratch[j,s] = weight^T[j,:] @ grad_out[s] for each s
# = [[1,1,1,1],[0,0,0,0],[0,0,0,0],[1,1,1,1]] (each row of col_scratch)
# Then col2im scatters these back:
# Patch (0,0) → pos 0,1,3,4 get +[1,0,0,1]
# Patch (0,1) → pos 1,2,4,5 get +[1,0,0,1]
# Patch (1,0) → pos 3,4,6,7 get +[1,0,0,1]
# Patch (1,1) → pos 4,5,7,8 get +[1,0,0,1]
# So:
# pos0(0,0): 1
# pos1(0,1): 0+1 = 1
# pos2(0,2): 0
# pos3(1,0): 0+1 = 1
# pos4(1,1): 1+0+0+1 = 2
# pos5(1,2): 1+0 = 1
# pos6(2,0): 0
# pos7(2,1): 1+0 = 1
# pos8(2,2): 1
gi_in = to_list(grad_in)
expected_gi = [1, 1, 0, 1, 2, 1, 0, 1, 1]
gi_match = all(approx(gi_in[i], expected_gi[i]) for i in range(9))
check(gi_match, "backward: grad_input values are correct")


# ── Section 9: Conv2D backward no-bias ──────────────────────────

print("\n--- Conv2D backward (no bias) ---")

conv_nb = Conv2D(1, 1, kernel_size=2, stride=1, padding=0, bias=False)
write_ptr(conv_nb.weight_ptr, [1.0, 1.0, 1.0, 1.0])

inp_nb = make_buf([1, 2, 3, 4, 5, 6, 7, 8, 9])
conv_nb.forward(inp_nb, batch=1, in_h=3, in_w=3)

grad_nb = make_buf([1.0, 1.0, 1.0, 1.0])
gi_nb, gw_nb, gb_nb = conv_nb.backward(grad_nb)

check(gb_nb is None, "no-bias conv: grad_bias is None")
check(gi_nb is not None and gw_nb is not None,
      "no-bias conv: grad_input and grad_weight still computed")


# ── Section 10: MaxPool2D construction ───────────────────────────

print("\n--- MaxPool2D construction ---")

pool1 = MaxPool2D(kernel_size=2)
check(pool1.pool_h == 2 and pool1.pool_w == 2, "pool_h/w from int")
check(pool1.stride == 2, "stride defaults to kernel_size")
check(pool1.padding == 0, "padding defaults to 0")

pool2 = MaxPool2D(kernel_size=(3, 2), stride=1, padding=1)
check(pool2.pool_h == 3 and pool2.pool_w == 2, "rectangular pool kernel")
check(pool2.stride == 1, "explicit stride")
check(pool2.padding == 1, "explicit padding")

rep_p = repr(pool1)
check("MaxPool2D" in rep_p and "2" in rep_p, "MaxPool2D repr")


# ── Section 11: MaxPool2D output_shape ───────────────────────────

print("\n--- MaxPool2D output_shape ---")

oh_p, ow_p = pool1.output_shape(4, 4)
check(oh_p == 2 and ow_p == 2, "4x4, pool=2, stride=2 → 2x2")

oh_p2, ow_p2 = pool1.output_shape(6, 8)
check(oh_p2 == 3 and ow_p2 == 4, "6x8, pool=2, stride=2 → 3x4")


# ── Section 12: MaxPool2D forward numerics ───────────────────────

print("\n--- MaxPool2D forward (numerical) ---")

# 1 sample, 1 channel, 4x4 input, pool=2, stride=2
# Input:
#   1  3  2  4
#   5  7  6  8
#   9 11 10 12
#  13 15 14 16
# Expected 2x2 output: max of each 2x2 block
#   top-left: max(1,3,5,7)=7     top-right: max(2,4,6,8)=8
#   bot-left: max(9,11,13,15)=15  bot-right: max(10,12,14,16)=16

pool = MaxPool2D(kernel_size=2, stride=2)
inp_pool = make_buf([1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16])
out_pool, oh_pool, ow_pool = pool.forward(inp_pool, batch=1, channels=1,
                                          in_h=4, in_w=4)

check(oh_pool == 2 and ow_pool == 2, "pool forward: output shape 2x2")

vp = to_list(out_pool)
check(approx(vp[0], 7.0) and approx(vp[1], 8.0) and
      approx(vp[2], 15.0) and approx(vp[3], 16.0),
      "pool forward: correct max values [7, 8, 15, 16]")


# ── Section 13: MaxPool2D forward with multiple channels ────────

print("\n--- MaxPool2D forward (multi-channel) ---")

# 1 sample, 2 channels, 4x4 each, pool=2, stride=2
# ch0: same as above (max=[7,8,15,16])
# ch1: all 100s → max=[100,100,100,100]
ch0 = [1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16]
ch1 = [100.0] * 16
inp_mc_pool = make_buf(ch0 + ch1)

out_mc_pool, _, _ = pool.forward(inp_mc_pool, batch=1, channels=2,
                                 in_h=4, in_w=4)
vp_mc = to_list(out_mc_pool)
check(approx(vp_mc[0], 7.0) and approx(vp_mc[3], 16.0),
      "multi-ch pool: channel 0 correct")
check(approx(vp_mc[4], 100.0) and approx(vp_mc[7], 100.0),
      "multi-ch pool: channel 1 correct")


# ── Section 14: MaxPool2D backward ──────────────────────────────

print("\n--- MaxPool2D backward ---")

# Re-run the simple 4x4→2x2 pool (from section 12, single channel)
pool_bk = MaxPool2D(kernel_size=2, stride=2)
inp_bk_pool = make_buf([1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16])
pool_bk.forward(inp_bk_pool, batch=1, channels=1, in_h=4, in_w=4)

# grad_output = [10, 20, 30, 40]
grad_pool = make_buf([10.0, 20.0, 30.0, 40.0])
grad_in_pool = pool_bk.backward(grad_pool)

gi_pool = to_list(grad_in_pool)

# The max positions were:
#   block (0,0): 7 at position (1,1) → flat index 5
#   block (0,1): 8 at position (1,3) → flat index 7
#   block (1,0): 15 at position (3,1) → flat index 13
#   block (1,1): 16 at position (3,3) → flat index 15
# So grad_input should be all zeros except:
#   index 5 = 10, index 7 = 20, index 13 = 30, index 15 = 40
expected_pool_grad = [0]*16
expected_pool_grad[5] = 10.0
expected_pool_grad[7] = 20.0
expected_pool_grad[13] = 30.0
expected_pool_grad[15] = 40.0

pool_grad_ok = all(approx(gi_pool[i], expected_pool_grad[i]) for i in range(16))
check(pool_grad_ok, "pool backward: gradients routed to max positions")


# ── Section 15: MaxPool2D backward before forward ───────────────

print("\n--- MaxPool2D backward edge cases ---")

pool_err = MaxPool2D(kernel_size=2)
try:
    pool_err.backward(make_buf([1.0]))
    check(False, "backward before forward should raise")
except RuntimeError:
    check(True, "backward before forward raises RuntimeError")


# ── Section 16: Conv2D + MaxPool2D pipeline ─────────────────────

print("\n--- Conv2D → MaxPool2D pipeline ---")

# A quick smoke test: conv then pool
# 1 channel, 4 filters, 3x3 kernel, pad=1, stride=1 → same spatial size
# Then pool 2x2, stride 2 → halves spatial
conv_pipe = Conv2D(1, 4, kernel_size=3, stride=1, padding=1, bias=True)
pool_pipe = MaxPool2D(kernel_size=2, stride=2)

# Input: 1 sample, 1 channel, 8x8
import random
random.seed(42)
inp_pipe_data = [random.gauss(0, 1) for _ in range(1 * 1 * 8 * 8)]
inp_pipe = make_buf(inp_pipe_data)

# Conv forward
out_conv, oh_c, ow_c = conv_pipe.forward(inp_pipe, batch=1, in_h=8, in_w=8)
check(oh_c == 8 and ow_c == 8, "pipeline: conv output 8x8 (same pad)")

# Pool forward on conv output
out_pool_pipe, oh_p_pipe, ow_p_pipe = pool_pipe.forward(
    out_conv, batch=1, channels=4, in_h=8, in_w=8)
check(oh_p_pipe == 4 and ow_p_pipe == 4, "pipeline: pool output 4x4")

# Make sure values aren't all zero (conv has random weights, not degenerate)
pool_vals = to_list(out_pool_pipe)
any_nonzero = any(abs(v) > 1e-10 for v in pool_vals)
check(any_nonzero, "pipeline: output has non-trivial values")


# ── Section 17: Large batch / multi-filter consistency ───────────

print("\n--- larger batch test ---")

# 3 channels → 8 filters, 3x3, pad=1, batch of 4, 6x6 input
conv_big = Conv2D(3, 8, kernel_size=3, stride=1, padding=1)
in_sz = 4 * 3 * 6 * 6  # batch*c*h*w
inp_big = make_buf([float(i % 17) / 17.0 for i in range(in_sz)])

out_big, oh_bg, ow_bg = conv_big.forward(inp_big, batch=4, in_h=6, in_w=6)
check(oh_bg == 6 and ow_bg == 6, "large: output spatial matches input (same pad)")

out_total = 4 * 8 * 6 * 6
vals_big = to_list(out_big, out_total)
check(len(vals_big) == out_total, "large: output has correct total size")

# Make sure backward doesn't crash
grad_big = make_buf([1.0] * out_total)
gi_big, gw_big, gb_big = conv_big.backward(grad_big)
check(len(to_list(gi_big)) == in_sz, "large: grad_input has correct size")
check(len(to_list(gw_big)) == conv_big.weight_size,
      "large: grad_weight has correct size")
check(len(to_list(gb_big)) == 8, "large: grad_bias has 8 elements")


# ── Section 18: Stride > 1 convolution ──────────────────────────

print("\n--- stride > 1 conv ---")

conv_s2 = Conv2D(1, 1, kernel_size=3, stride=2, padding=0, bias=False)
write_ptr(conv_s2.weight_ptr, [1.0] * 9)

# 5x5 input of ones → output should be (5-3)/2+1 = 2x2, each value = 9
inp_s2 = make_buf([1.0] * 25)
out_s2, oh_s2, ow_s2 = conv_s2.forward(inp_s2, batch=1, in_h=5, in_w=5)
check(oh_s2 == 2 and ow_s2 == 2, "stride=2: output is 2x2")

vs2 = to_list(out_s2)
check(all(approx(v, 9.0) for v in vs2), "stride=2: all outputs = 9 (sum of ones)")


# ── Section 19: Conv2D weight initialisation sanity ──────────────

print("\n--- weight init sanity ---")

# Weights should be ~U(-bound, bound) where bound = sqrt(2/fan_in)
conv_init = Conv2D(16, 32, kernel_size=3)
fan_in = 16 * 3 * 3
bound = math.sqrt(2.0 / fan_in)

wts = read_ptr(conv_init.weight_ptr, conv_init.weight_size)
max_w = max(abs(w) for w in wts)
check(max_w < bound * 1.1,  # a bit of tolerance for the LCG mapping
      f"weights within Kaiming bounds (max |w|={max_w:.4f}, bound={bound:.4f})")

# Check that weights aren't all zero (init actually happened)
mean_abs = sum(abs(w) for w in wts) / len(wts)
check(mean_abs > 0.001, f"weights aren't degenerate (mean |w| = {mean_abs:.5f})")


# ── Section 20: MaxPool2D with stride != kernel_size ─────────────

print("\n--- pool with stride != kernel ---")

# 1 channel, 4x4 input, pool_h=3, pool_w=3, stride=1, pad=0
# Output: (4-3)/1+1=2, so 2x2
pool_overlap = MaxPool2D(kernel_size=3, stride=1)
inp_ov = make_buf([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
out_ov, oh_ov, ow_ov = pool_overlap.forward(inp_ov, batch=1, channels=1,
                                             in_h=4, in_w=4)
check(oh_ov == 2 and ow_ov == 2, "overlapping pool: output 2x2")

vov = to_list(out_ov)
# top-left 3x3 block: max(1..11) = 11
# top-right 3x3 block starting at col 1: max of [[2,3,4],[6,7,8],[10,11,12]] = 12
# bot-left: max of rows 1-3, cols 0-2: [[5,6,7],[9,10,11],[13,14,15]] = 15
# bot-right: max of rows 1-3, cols 1-3: [[6,7,8],[10,11,12],[14,15,16]] = 16
check(approx(vov[0], 11.0) and approx(vov[1], 12.0) and
      approx(vov[2], 15.0) and approx(vov[3], 16.0),
      "overlapping pool: correct max values")


# ── Section 21: MaxPool2D batched ────────────────────────────────

print("\n--- pool batched ---")

# 2 samples, 1 channel, 4x4, pool=2 stride=2
inp_pb = make_buf(
    [1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16] +
    [16, 14, 15, 13, 12, 10, 11, 9, 8, 6, 7, 5, 4, 2, 3, 1]
)
pool_batch = MaxPool2D(2, stride=2)
out_pb, _, _ = pool_batch.forward(inp_pb, batch=2, channels=1, in_h=4, in_w=4)
vpb = to_list(out_pb)

# Sample 0: [7, 8, 15, 16] (same as before)
# Sample 1: max of each 2x2 block of the reversed-ish input
# Block (0,0): max(16,14,12,10)=16
# Block (0,1): max(15,13,11,9)=15
# Block (1,0): max(8,6,4,2)=8
# Block (1,1): max(7,5,3,1)=7
check(approx(vpb[0], 7.0) and approx(vpb[3], 16.0),
      "batched pool: sample 0 correct")
check(approx(vpb[4], 16.0) and approx(vpb[5], 15.0) and
      approx(vpb[6], 8.0) and approx(vpb[7], 7.0),
      "batched pool: sample 1 correct")


# ── Section 22: Conv2D accessor functions ────────────────────────

print("\n--- accessor functions ---")

ca = Conv2D(3, 16, kernel_size=(5, 7), stride=2, padding=3)
# Access through the C layer via _lib calls directly
check(_lib.conv2d_layer_in_channels(ca._ptr) == 3, "accessor: in_channels")
check(_lib.conv2d_layer_out_channels(ca._ptr) == 16, "accessor: out_channels")
check(_lib.conv2d_layer_kernel_h(ca._ptr) == 5, "accessor: kernel_h")
check(_lib.conv2d_layer_kernel_w(ca._ptr) == 7, "accessor: kernel_w")
check(_lib.conv2d_layer_stride(ca._ptr) == 2, "accessor: stride")
check(_lib.conv2d_layer_padding(ca._ptr) == 3, "accessor: padding")


# ── Section 23: Numerical gradient check for Conv2D ──────────────

print("\n--- finite-difference gradient check ---")

# Verify backward with finite differences for a small network
eps = 1e-5
conv_fd = Conv2D(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
write_ptr(conv_fd.weight_ptr, [0.5, -0.3, 0.2, 0.8])
write_ptr(conv_fd.bias_ptr, [0.1])

inp_fd = make_buf([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

# Forward: get output, compute a scalar loss = sum(output)
out_fd, _, _ = conv_fd.forward(inp_fd, batch=1, in_h=3, in_w=3)
base_loss = sum(to_list(out_fd))

# Analytic gradients via backward with grad_output = all ones
grad_ones = make_buf([1.0, 1.0, 1.0, 1.0])
gi_fd, gw_fd, gb_fd = conv_fd.backward(grad_ones)

# Check weight gradients via finite diff
weights_orig = [0.5, -0.3, 0.2, 0.8]
fd_gw = []
for i in range(4):
    w_plus = list(weights_orig)
    w_plus[i] += eps
    write_ptr(conv_fd.weight_ptr, w_plus)
    out_p, _, _ = conv_fd.forward(inp_fd, batch=1, in_h=3, in_w=3)
    loss_p = sum(to_list(out_p))
    fd_gw.append((loss_p - base_loss) / eps)

# Restore
write_ptr(conv_fd.weight_ptr, weights_orig)

analytic_gw = to_list(gw_fd)
gw_close = all(abs(analytic_gw[i] - fd_gw[i]) < 1e-4 for i in range(4))
check(gw_close, "finite-diff: weight gradients match analytic")

# Check bias gradient via finite diff
write_ptr(conv_fd.bias_ptr, [0.1 + eps])
out_bp, _, _ = conv_fd.forward(inp_fd, batch=1, in_h=3, in_w=3)
loss_bp = sum(to_list(out_bp))
fd_gb = (loss_bp - base_loss) / eps
write_ptr(conv_fd.bias_ptr, [0.1])

check(abs(to_list(gb_fd)[0] - fd_gb) < 1e-4,
      "finite-diff: bias gradient matches analytic")

# Check input gradients via finite diff
fd_gi = []
inp_orig = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(9):
    perturbed = list(inp_orig)
    perturbed[i] += eps
    inp_p = make_buf(perturbed)
    out_ip, _, _ = conv_fd.forward(inp_p, batch=1, in_h=3, in_w=3)
    loss_ip = sum(to_list(out_ip))
    fd_gi.append((loss_ip - base_loss) / eps)

analytic_gi = to_list(gi_fd)
gi_close = all(abs(analytic_gi[i] - fd_gi[i]) < 1e-4 for i in range(9))
check(gi_close, "finite-diff: input gradients match analytic")


# ── Section 24: Conv2D with no-bias backward finite diff ────────

print("\n--- no-bias finite diff ---")

conv_fd_nb = Conv2D(1, 1, kernel_size=2, stride=1, padding=0, bias=False)
write_ptr(conv_fd_nb.weight_ptr, [0.5, -0.3, 0.2, 0.8])

inp_fd_nb = make_buf([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
out_fd_nb, _, _ = conv_fd_nb.forward(inp_fd_nb, batch=1, in_h=3, in_w=3)
base_nb = sum(to_list(out_fd_nb))

grad_ones_nb = make_buf([1.0, 1.0, 1.0, 1.0])
gi_nb2, gw_nb2, gb_nb2 = conv_fd_nb.backward(grad_ones_nb)

check(gb_nb2 is None, "no-bias FD: grad_bias is None")

# Quick FD check on one weight
wt_nb_orig = [0.5, -0.3, 0.2, 0.8]
wt_nb_pert = list(wt_nb_orig)
wt_nb_pert[0] += eps
write_ptr(conv_fd_nb.weight_ptr, wt_nb_pert)
out_nb_p, _, _ = conv_fd_nb.forward(inp_fd_nb, batch=1, in_h=3, in_w=3)
fd_nb_g0 = (sum(to_list(out_nb_p)) - base_nb) / eps

check(abs(to_list(gw_nb2)[0] - fd_nb_g0) < 1e-4,
      "no-bias FD: weight grad[0] matches")


# ── Summary ──────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Conv2D / MaxPool2D tests:  {passed} passed, {failed} failed")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
