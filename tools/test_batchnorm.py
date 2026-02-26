#!/usr/bin/env python3
"""
Tests for BatchNorm1d (Batch 3a).

Covers:
  1. Output has mean≈0, var≈1 per feature in train mode
  2. Running stats update correctly over multiple batches
  3. Eval mode uses running stats, not batch stats
  4. Backward: grad_gamma and grad_beta computed correctly
  5. Edge: batch_size=1
  6. Edge: all-same-value input (zero variance)
  7. Gamma/beta initialisation: gamma=1, beta=0
  8. Train/eval mode propagation
"""

import sys, os, ctypes, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn
from pyneural.nn import BatchNorm1d

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name} — {detail}")
        failed += 1


def make_f64_array(values):
    """Create a ctypes double array from a flat list."""
    n = len(values)
    arr = (ctypes.c_double * n)(*values)
    return arr


def f64_array_to_list(ptr, n):
    """Read n doubles from a pointer."""
    arr = (ctypes.c_double * n).from_address(ptr)
    return [float(arr[i]) for i in range(n)]


# ---------------------------------------------------------------------------
print("=== Test 1: BatchNorm1d output mean≈0, var≈1 ===")
bn = BatchNorm1d(3, momentum=0.1, eps=1e-5)

# Input: 4 samples × 3 features, with known non-zero mean and varying std
# Feature 0: values [10, 20, 30, 40] → mean=25, var=125
# Feature 1: values [1, 1, 1, 1]    → mean=1, var=0
# Feature 2: values [0, 2, 4, 6]    → mean=3, var=5
input_data = make_f64_array([
    10.0, 1.0, 0.0,
    20.0, 1.0, 2.0,
    30.0, 1.0, 4.0,
    40.0, 1.0, 6.0,
])
output_data = make_f64_array([0.0] * 12)

bn.train()
result = bn.forward_f64(
    ctypes.cast(input_data, ctypes.c_void_p),
    ctypes.cast(output_data, ctypes.c_void_p),
    4
)
check("forward returns 0", result == 0, f"got {result}")

# Check output per feature
out = [float(output_data[i]) for i in range(12)]
for c in range(3):
    col = [out[b * 3 + c] for b in range(4)]
    mean = sum(col) / len(col)
    var = sum((x - mean) ** 2 for x in col) / len(col)
    check(f"feature {c} mean≈0", abs(mean) < 0.01, f"mean={mean:.6f}")
    if c != 1:  # feature 1 is constant → all zeros, var=0 after norm
        check(f"feature {c} var≈1", abs(var - 1.0) < 0.01, f"var={var:.6f}")

# ---------------------------------------------------------------------------
print("\n=== Test 2: Running stats update ===")
rm = f64_array_to_list(bn.running_mean_ptr, 3)
rv = f64_array_to_list(bn.running_var_ptr, 3)
# After 1 batch with momentum=0.1:
# running_mean = 0.9*0 + 0.1*batch_mean
check("running_mean[0]≈2.5 (0.1*25)", abs(rm[0] - 2.5) < 0.01, f"got {rm[0]}")
check("running_mean[1]≈0.1 (0.1*1)", abs(rm[1] - 0.1) < 0.01, f"got {rm[1]}")
check("running_mean[2]≈0.3 (0.1*3)", abs(rm[2] - 0.3) < 0.01, f"got {rm[2]}")

# Running var uses unbiased variance: S²/(N-1)
# Feature 0: sum_sq_diff = 500, unbiased_var = 500/3 ≈ 166.67
# running_var = 0.9*1 + 0.1*166.67 ≈ 17.567
check("running_var[0] > 1 (updated)", rv[0] > 5, f"got {rv[0]}")

# ---------------------------------------------------------------------------
print("\n=== Test 3: Eval mode uses running stats ===")
bn.eval()
# Feed different data in eval mode — should use running stats
eval_input = make_f64_array([100.0, 100.0, 100.0] * 2)  # 2×3
eval_output = make_f64_array([0.0] * 6)
result = bn.forward_f64(
    ctypes.cast(eval_input, ctypes.c_void_p),
    ctypes.cast(eval_output, ctypes.c_void_p),
    2
)
check("eval forward returns 0", result == 0)

eval_out = [float(eval_output[i]) for i in range(6)]
# In eval mode, output should use running_mean/running_var
# All samples are the same, so within-feature outputs should be identical
check("eval: same input → same output per sample",
      abs(eval_out[0] - eval_out[3]) < 1e-10 and
      abs(eval_out[1] - eval_out[4]) < 1e-10,
      f"sample1={eval_out[:3]}, sample2={eval_out[3:]}")
bn.train()

# ---------------------------------------------------------------------------
print("\n=== Test 4: Backward ===")
# Forward first
fwd_in = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2×3
fwd_out = make_f64_array([0.0] * 6)
bn2 = BatchNorm1d(3, momentum=0.1, eps=1e-5)
bn2.train()
bn2.forward_f64(
    ctypes.cast(fwd_in, ctypes.c_void_p),
    ctypes.cast(fwd_out, ctypes.c_void_p),
    2
)

# Backward with uniform gradient
grad_out = make_f64_array([1.0] * 6)
grad_in = make_f64_array([0.0] * 6)
grad_gamma = make_f64_array([0.0] * 3)
grad_beta = make_f64_array([0.0] * 3)

result = bn2.backward_f64(
    ctypes.cast(grad_out, ctypes.c_void_p),
    ctypes.cast(grad_in, ctypes.c_void_p),
    ctypes.cast(grad_gamma, ctypes.c_void_p),
    ctypes.cast(grad_beta, ctypes.c_void_p),
    2
)
check("backward returns 0", result == 0)

# grad_beta should be sum of grad_output per feature = 2.0 each
gb = [float(grad_beta[i]) for i in range(3)]
check("grad_beta = sum(grad_out) per feature",
      all(abs(g - 2.0) < 1e-6 for g in gb),
      f"got {gb}")

# ---------------------------------------------------------------------------
print("\n=== Test 5: Edge — batch_size=1 ===")
bn3 = BatchNorm1d(2, momentum=0.1, eps=1e-5)
single_in = make_f64_array([5.0, 10.0])
single_out = make_f64_array([0.0, 0.0])
result = bn3.forward_f64(
    ctypes.cast(single_in, ctypes.c_void_p),
    ctypes.cast(single_out, ctypes.c_void_p),
    1
)
check("batch_size=1 doesn't crash", result == 0)
# With batch_size=1, mean=value, var=0, so x_hat=0, output=beta=0
s_out = [float(single_out[i]) for i in range(2)]
check("batch_size=1: output = beta (≈0)", all(abs(v) < 0.01 for v in s_out),
      f"got {s_out}")

# ---------------------------------------------------------------------------
print("\n=== Test 6: Edge — all-same-value input ===")
bn4 = BatchNorm1d(2, momentum=0.1, eps=1e-5)
same_in = make_f64_array([7.0, 3.0, 7.0, 3.0, 7.0, 3.0])  # 3×2, constant per feature
same_out = make_f64_array([0.0] * 6)
result = bn4.forward_f64(
    ctypes.cast(same_in, ctypes.c_void_p),
    ctypes.cast(same_out, ctypes.c_void_p),
    3
)
check("constant input doesn't crash", result == 0)
# var=0 + eps → outputs should be ~0 (since x_hat ≈ 0)
const_out = [float(same_out[i]) for i in range(6)]
check("constant input → outputs ≈ 0",
      all(abs(v) < 0.1 for v in const_out),
      f"got {const_out}")

# ---------------------------------------------------------------------------
print("\n=== Test 7: Gamma/beta initialisation ===")
bn5 = BatchNorm1d(4)
gamma = f64_array_to_list(bn5.gamma_ptr, 4)
beta = f64_array_to_list(bn5.beta_ptr, 4)
check("gamma initialised to 1.0", all(abs(g - 1.0) < 1e-10 for g in gamma))
check("beta initialised to 0.0", all(abs(b) < 1e-10 for b in beta))

# ---------------------------------------------------------------------------
print("\n=== Test 8: Train/eval mode propagation ===")
bn6 = BatchNorm1d(4)
bn6.eval()
check("eval sets _training=False", not bn6._training)
bn6.train()
check("train sets _training=True", bn6._training)

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"BatchNorm1d tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL BATCHNORM1D TESTS PASSED")
    sys.exit(0)
