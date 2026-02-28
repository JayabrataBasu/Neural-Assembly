#!/usr/bin/env python3
"""
Tests for LabelSmoothingCrossEntropy (Batch 4a).

Covers:
  1. smoothing=0 → equivalent to standard CE
  2. smoothing=1 → uniform distribution loss
  3. Correct gradient shape & values
  4. Gradient sums to ~0 per sample (conservation)
  5. Loss is lower-bounded (non-negative)
  6. Invalid parameters rejected
  7. Single-class edge case
  8. Large batch
  9. backward without forward → error
 10. Python list convenience API
 11. Gradient numerical check (finite differences)
"""

import sys, os, ctypes, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn
from pyneural.nn import LabelSmoothingCrossEntropy

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


def make_f64(values):
    return (ctypes.c_double * len(values))(*values)


def make_i64(values):
    return (ctypes.c_int64 * len(values))(*values)


def softmax(logits, K):
    """Python softmax for verification."""
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def standard_ce(logits_flat, targets, K):
    """Python standard cross-entropy (no smoothing)."""
    N = len(targets)
    total = 0.0
    for i in range(N):
        row = logits_flat[i * K:(i + 1) * K]
        sm = softmax(row, K)
        total -= math.log(max(sm[targets[i]], 1e-15))
    return total / N


# ---------------------------------------------------------------------------
print("=== Test 1: smoothing=0 → standard CE ===")
loss_fn = LabelSmoothingCrossEntropy(num_classes=3, smoothing=0.0)

logits_list = [2.0, 1.0, 0.1, 0.5, 2.5, 0.3]  # 2 samples × 3 classes
targets_list = [0, 1]

loss_smooth = loss_fn(logits_list, targets_list)
loss_std = standard_ce(logits_list, targets_list, 3)
check("smooth=0 ≈ standard CE", abs(loss_smooth - loss_std) < 1e-10,
      f"smooth={loss_smooth:.10f}, std={loss_std:.10f}")

# ---------------------------------------------------------------------------
print("\n=== Test 2: smoothing=1 → uniform target ===")
loss_fn1 = LabelSmoothingCrossEntropy(num_classes=3, smoothing=1.0)
loss_uniform = loss_fn1(logits_list, targets_list)

# With uniform target (1/K for all classes), loss = -mean(sum(1/K * log_softmax))
# which is the entropy of softmax output
K = 3
total = 0.0
for i in range(2):
    row = logits_list[i * K:(i + 1) * K]
    mx = max(row)
    lse = math.log(sum(math.exp(x - mx) for x in row))
    for c in range(K):
        log_soft = (row[c] - mx) - lse
        total -= (1.0 / K) * log_soft
total /= 2.0
check("smooth=1 ≈ uniform-target CE", abs(loss_uniform - total) < 1e-10,
      f"got={loss_uniform:.10f}, expected={total:.10f}")

# ---------------------------------------------------------------------------
print("\n=== Test 3: smoothing reduces loss for correct class ===")
loss_fn_lo = LabelSmoothingCrossEntropy(num_classes=3, smoothing=0.0)
loss_fn_hi = LabelSmoothingCrossEntropy(num_classes=3, smoothing=0.3)
loss_lo = loss_fn_lo(logits_list, targets_list)
loss_hi = loss_fn_hi(logits_list, targets_list)
# With smoothing, we spread probability mass, so loss for correct logits increases
check("smoothing > 0 increases loss vs. smooth=0", loss_hi > loss_lo,
      f"lo={loss_lo:.6f}, hi={loss_hi:.6f}")

# ---------------------------------------------------------------------------
print("\n=== Test 4: Gradient shape and conservation ===")
loss_fn4 = LabelSmoothingCrossEntropy(num_classes=3, smoothing=0.1)
logits_arr = make_f64(logits_list)
targets_arr = make_i64(targets_list)
grad_arr = make_f64([0.0] * 6)

loss_fn4.forward_f64(
    ctypes.cast(logits_arr, ctypes.c_void_p),
    ctypes.cast(targets_arr, ctypes.c_void_p),
    2
)
loss_fn4.backward_f64(ctypes.cast(grad_arr, ctypes.c_void_p))

grads = [float(grad_arr[i]) for i in range(6)]
check("grad length = 6", len(grads) == 6)

# Each sample's gradients sum ≈ 0 (softmax - smoothed_target both sum to 1)
for s in range(2):
    gsum = sum(grads[s * 3:(s + 1) * 3])
    check(f"sample {s} grad sum ≈ 0", abs(gsum) < 1e-12, f"sum={gsum}")

# ---------------------------------------------------------------------------
print("\n=== Test 5: Loss is non-negative ===")
loss_fn5 = LabelSmoothingCrossEntropy(num_classes=4, smoothing=0.2)
loss_val = loss_fn5([1, 2, 3, 4, 5, 6, 7, 8], [2, 3])
check("loss >= 0", loss_val >= 0.0, f"loss={loss_val}")

# ---------------------------------------------------------------------------
print("\n=== Test 6: Invalid parameters rejected ===")
caught = False
try:
    LabelSmoothingCrossEntropy(num_classes=3, smoothing=-0.1)
except ValueError:
    caught = True
check("smoothing < 0 raises ValueError", caught)

caught = False
try:
    LabelSmoothingCrossEntropy(num_classes=3, smoothing=1.5)
except ValueError:
    caught = True
check("smoothing > 1 raises ValueError", caught)

caught = False
try:
    LabelSmoothingCrossEntropy(num_classes=0)
except ValueError:
    caught = True
check("num_classes=0 raises ValueError", caught)

# ---------------------------------------------------------------------------
print("\n=== Test 7: Single-class edge case ===")
loss_fn7 = LabelSmoothingCrossEntropy(num_classes=1, smoothing=0.1)
loss_1c = loss_fn7([5.0], [0])
# single class: softmax=1.0, log_softmax=0.0, loss=0.0
check("1-class loss ≈ 0", abs(loss_1c) < 1e-10, f"loss={loss_1c}")

# ---------------------------------------------------------------------------
print("\n=== Test 8: Larger batch (100 samples) ===")
import random
random.seed(42)
N, K = 100, 5
big_logits = [random.gauss(0, 1) for _ in range(N * K)]
big_targets = [random.randint(0, K - 1) for _ in range(N)]
loss_fn8 = LabelSmoothingCrossEntropy(num_classes=K, smoothing=0.05)
loss_big = loss_fn8(big_logits, big_targets)
check("100-sample loss finite", math.isfinite(loss_big), f"loss={loss_big}")
check("100-sample loss > 0", loss_big > 0)

# ---------------------------------------------------------------------------
print("\n=== Test 9: backward before forward → error ===")
loss_fn9 = LabelSmoothingCrossEntropy(num_classes=3, smoothing=0.1)
caught = False
try:
    grad_tmp = make_f64([0.0] * 6)
    loss_fn9.backward_f64(ctypes.cast(grad_tmp, ctypes.c_void_p))
except RuntimeError:
    caught = True
check("backward before forward raises RuntimeError", caught)

# ---------------------------------------------------------------------------
print("\n=== Test 10: Numerical gradient check ===")
loss_fn10 = LabelSmoothingCrossEntropy(num_classes=3, smoothing=0.15)
base_logits = [1.0, 2.0, 0.5]
tgt10 = [1]
eps = 1e-5

# Analytical gradient
la = make_f64(base_logits)
ta = make_i64(tgt10)
ga = make_f64([0.0] * 3)
loss_fn10.forward_f64(
    ctypes.cast(la, ctypes.c_void_p),
    ctypes.cast(ta, ctypes.c_void_p), 1
)
loss_fn10.backward_f64(ctypes.cast(ga, ctypes.c_void_p))
analytical = [float(ga[i]) for i in range(3)]

# Numerical gradient via central differences
numerical = []
for c in range(3):
    # +eps
    perturbed = list(base_logits)
    perturbed[c] += eps
    loss_p = loss_fn10.forward(perturbed, tgt10)
    # -eps
    perturbed = list(base_logits)
    perturbed[c] -= eps
    loss_m = loss_fn10.forward(perturbed, tgt10)
    numerical.append((loss_p - loss_m) / (2 * eps))

max_diff = max(abs(a - n) for a, n in zip(analytical, numerical))
check("numerical grad matches analytical", max_diff < 1e-6,
      f"max_diff={max_diff:.2e}, analytical={analytical}, numerical={numerical}")

# ---------------------------------------------------------------------------
print("\n=== Test 11: repr ===")
r = repr(LabelSmoothingCrossEntropy(num_classes=10, smoothing=0.2))
check("repr contains class name", "LabelSmoothingCrossEntropy" in r)
check("repr contains smoothing", "0.2" in r)

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"LabelSmoothingCE tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL LABEL SMOOTHING TESTS PASSED")
    sys.exit(0)
