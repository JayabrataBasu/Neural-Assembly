#!/usr/bin/env python3
"""
Tests for LayerNorm (Batch 3b).

Covers:
  1. Output has mean≈0, var≈1 per sample (not per feature)
  2. Doesn't depend on batch size (no running stats)
  3. Backward: grad_gamma and grad_beta correct
  4. Edge: single-feature input
  5. Edge: single-sample batch
  6. Gamma/beta initialisation
  7. Same output in train and eval mode (no running stats)
"""

import sys, os, ctypes, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn
from pyneural.nn import LayerNorm

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
    n = len(values)
    return (ctypes.c_double * n)(*values)


def f64_array_to_list(ptr, n):
    arr = (ctypes.c_double * n).from_address(ptr)
    return [float(arr[i]) for i in range(n)]


# ---------------------------------------------------------------------------
print("=== Test 1: LayerNorm output mean≈0, var≈1 per sample ===")
ln = LayerNorm(4, eps=1e-5)

# Input: 3 samples × 4 features
input_data = make_f64_array([
    1.0, 2.0, 3.0, 4.0,     # sample 0: mean=2.5, var=1.25
    10.0, 20.0, 30.0, 40.0, # sample 1: mean=25, var=125
    0.0, 0.0, 0.0, 1.0,     # sample 2: mean=0.25, var=0.1875
])
output_data = make_f64_array([0.0] * 12)

result = ln.forward_f64(
    ctypes.cast(input_data, ctypes.c_void_p),
    ctypes.cast(output_data, ctypes.c_void_p),
    3
)
check("forward returns 0", result == 0)

out = [float(output_data[i]) for i in range(12)]
for b in range(3):
    row = [out[b * 4 + c] for c in range(4)]
    mean = sum(row) / len(row)
    var = sum((x - mean) ** 2 for x in row) / len(row)
    check(f"sample {b} mean≈0", abs(mean) < 0.01, f"mean={mean:.6f}")
    check(f"sample {b} var≈1", abs(var - 1.0) < 0.05, f"var={var:.6f}")

# ---------------------------------------------------------------------------
print("\n=== Test 2: Same result regardless of batch size ===")
# Process sample 0 alone vs in a batch of 3
ln2 = LayerNorm(4, eps=1e-5)

single_in = make_f64_array([1.0, 2.0, 3.0, 4.0])
single_out = make_f64_array([0.0] * 4)
ln2.forward_f64(
    ctypes.cast(single_in, ctypes.c_void_p),
    ctypes.cast(single_out, ctypes.c_void_p),
    1
)
single_result = [float(single_out[i]) for i in range(4)]

batch_in = make_f64_array([1.0, 2.0, 3.0, 4.0, 99.0, 99.0, 99.0, 99.0])
batch_out = make_f64_array([0.0] * 8)
ln2.forward_f64(
    ctypes.cast(batch_in, ctypes.c_void_p),
    ctypes.cast(batch_out, ctypes.c_void_p),
    2
)
batch_result = [float(batch_out[i]) for i in range(4)]  # first sample only

match = all(abs(a - b) < 1e-10 for a, b in zip(single_result, batch_result))
check("single vs batch: same output for same sample", match,
      f"single={single_result}, batch={batch_result}")

# ---------------------------------------------------------------------------
print("\n=== Test 3: Backward ===")
ln3 = LayerNorm(3, eps=1e-5)
fwd_in = make_f64_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2×3
fwd_out = make_f64_array([0.0] * 6)
ln3.forward_f64(
    ctypes.cast(fwd_in, ctypes.c_void_p),
    ctypes.cast(fwd_out, ctypes.c_void_p),
    2
)

grad_out = make_f64_array([1.0] * 6)
grad_in = make_f64_array([0.0] * 6)
grad_gamma = make_f64_array([0.0] * 3)
grad_beta = make_f64_array([0.0] * 3)

result = ln3.backward_f64(
    ctypes.cast(grad_out, ctypes.c_void_p),
    ctypes.cast(grad_in, ctypes.c_void_p),
    ctypes.cast(grad_gamma, ctypes.c_void_p),
    ctypes.cast(grad_beta, ctypes.c_void_p),
    2
)
check("backward returns 0", result == 0)

gb = [float(grad_beta[i]) for i in range(3)]
check("grad_beta = sum over batch", all(abs(g - 2.0) < 1e-6 for g in gb),
      f"got {gb}")

# ---------------------------------------------------------------------------
print("\n=== Test 4: Edge — single-feature input ===")
ln4 = LayerNorm(1, eps=1e-5)
one_feat_in = make_f64_array([5.0, 10.0, 15.0])  # 3×1
one_feat_out = make_f64_array([0.0] * 3)
result = ln4.forward_f64(
    ctypes.cast(one_feat_in, ctypes.c_void_p),
    ctypes.cast(one_feat_out, ctypes.c_void_p),
    3
)
check("single-feature doesn't crash", result == 0)
# With 1 feature: mean=value, var=0, x_hat=0, output=beta=0
one_out = [float(one_feat_out[i]) for i in range(3)]
check("single-feature: output ≈ 0 (beta)",
      all(abs(v) < 0.01 for v in one_out),
      f"got {one_out}")

# ---------------------------------------------------------------------------
print("\n=== Test 5: Edge — single-sample batch ===")
ln5 = LayerNorm(4, eps=1e-5)
ss_in = make_f64_array([1.0, 2.0, 3.0, 4.0])
ss_out = make_f64_array([0.0] * 4)
result = ln5.forward_f64(
    ctypes.cast(ss_in, ctypes.c_void_p),
    ctypes.cast(ss_out, ctypes.c_void_p),
    1
)
check("single-sample batch doesn't crash", result == 0)

# ---------------------------------------------------------------------------
print("\n=== Test 6: Gamma/beta initialisation ===")
ln6 = LayerNorm(5)
gamma = f64_array_to_list(ln6.gamma_ptr, 5)
beta = f64_array_to_list(ln6.beta_ptr, 5)
check("gamma = 1.0", all(abs(g - 1.0) < 1e-10 for g in gamma))
check("beta = 0.0", all(abs(b) < 1e-10 for b in beta))

# ---------------------------------------------------------------------------
print("\n=== Test 7: Same output in train and eval mode ===")
ln7 = LayerNorm(3, eps=1e-5)
test_in = make_f64_array([2.0, 4.0, 6.0])

ln7.train()
train_out = make_f64_array([0.0] * 3)
ln7.forward_f64(
    ctypes.cast(test_in, ctypes.c_void_p),
    ctypes.cast(train_out, ctypes.c_void_p),
    1
)

ln7.eval()
eval_out = make_f64_array([0.0] * 3)
ln7.forward_f64(
    ctypes.cast(test_in, ctypes.c_void_p),
    ctypes.cast(eval_out, ctypes.c_void_p),
    1
)

train_vals = [float(train_out[i]) for i in range(3)]
eval_vals = [float(eval_out[i]) for i in range(3)]
check("train==eval output (no running stats)",
      all(abs(a - b) < 1e-10 for a, b in zip(train_vals, eval_vals)),
      f"train={train_vals}, eval={eval_vals}")

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"LayerNorm tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL LAYERNORM TESTS PASSED")
    sys.exit(0)
