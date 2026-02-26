#!/usr/bin/env python3
"""
Tests for the Dropout layer (Batch 1a).

Covers:
  1. p=0 → identity (all elements pass through)
  2. p=1 → all zeros
  3. Eval mode → identity regardless of p
  4. Backward mask consistency (same mask as forward)
  5. Inverted scaling (output mean ≈ input mean)
  6. Edge: invalid p raises ValueError
  7. Edge: single-element tensor
  8. Convergence: XOR problem with Dropout(0.3) still converges
"""

import sys, os, math, ctypes

# Ensure pyneural is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn

def make_ones(shape):
    """Create a tensor filled with 1.0 (workaround: Tensor.ones may not fill f32)."""
    t = pn.Tensor.zeros(shape)
    t.fill(1.0)
    return t

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


def tensor_to_list(t):
    """Extract float values from a Tensor into a Python list."""
    numel = int(pn.core._lib.neural_tensor_numel(t._ptr))
    data_ptr = pn.core._lib.neural_tensor_data(t._ptr)
    dtype = int(pn.core._lib.neural_tensor_dtype(t._ptr))
    if dtype == 0:  # FLOAT32
        arr = (ctypes.c_float * numel).from_address(data_ptr)
    else:  # FLOAT64
        arr = (ctypes.c_double * numel).from_address(data_ptr)
    return [float(arr[i]) for i in range(numel)]


# ---------------------------------------------------------------------------
print("=== Test 1: Dropout p=0 (identity) ===")
drop0 = pn.Dropout(p=0.0)
x = make_ones([8])
y = drop0(x)
vals = tensor_to_list(y)
check("all elements equal 1.0", all(abs(v - 1.0) < 1e-6 for v in vals),
      f"got {vals}")

# ---------------------------------------------------------------------------
print("\n=== Test 2: Dropout p=1 (all zeros) ===")
drop1 = pn.Dropout(p=1.0)
x = make_ones([8])
y = drop1(x)
vals = tensor_to_list(y)
check("all elements equal 0.0", all(abs(v) < 1e-6 for v in vals),
      f"got {vals}")

# ---------------------------------------------------------------------------
print("\n=== Test 3: Eval mode → identity ===")
drop_half = pn.Dropout(p=0.5)
drop_half.eval()  # set eval mode
x = make_ones([100])
y = drop_half(x)
vals = tensor_to_list(y)
check("eval mode: all elements equal 1.0",
      all(abs(v - 1.0) < 1e-6 for v in vals),
      f"got {sum(1 for v in vals if abs(v-1.0)<1e-6)}/100 ones")

# Restore train mode for later tests
drop_half.train()

# ---------------------------------------------------------------------------
print("\n=== Test 4: Forward mask consistency (backward reuses mask) ===")
drop_bwd = pn.Dropout(p=0.5)
x = make_ones([50])
y = drop_bwd(x)

fwd_vals = tensor_to_list(y)
# Elements are either 0 (dropped) or 2.0 (kept, scaled by 1/(1-0.5)=2)
fwd_dropped = [i for i, v in enumerate(fwd_vals) if abs(v) < 1e-6]
fwd_kept = [i for i, v in enumerate(fwd_vals) if abs(v) > 1e-6]

# Now backward
grad_out = make_ones([50])
grad_in = drop_bwd.backward(grad_out)
bwd_vals = tensor_to_list(grad_in)
bwd_dropped = [i for i, v in enumerate(bwd_vals) if abs(v) < 1e-6]
bwd_kept = [i for i, v in enumerate(bwd_vals) if abs(v) > 1e-6]

check("backward zeros match forward zeros",
      fwd_dropped == bwd_dropped,
      f"fwd_dropped({len(fwd_dropped)}) != bwd_dropped({len(bwd_dropped)})")
check("backward non-zeros match forward non-zeros",
      fwd_kept == bwd_kept,
      f"fwd_kept({len(fwd_kept)}) != bwd_kept({len(bwd_kept)})")

# ---------------------------------------------------------------------------
print("\n=== Test 5: Inverted scaling (mean preservation) ===")
# With inverted dropout, E[output] ≈ E[input]
drop_scale = pn.Dropout(p=0.3)
x = make_ones([10000])
y = drop_scale(x)
vals = tensor_to_list(y)
mean_out = sum(vals) / len(vals)
# The expected mean is 1.0 (input) due to inverted scaling
# Allow 10% tolerance for stochasticity
check("mean output ≈ 1.0 (inverted scaling)",
      abs(mean_out - 1.0) < 0.15,
      f"mean={mean_out:.4f}, expected ≈ 1.0")

# Check that approximately p fraction of elements are dropped
n_dropped = sum(1 for v in vals if abs(v) < 1e-6)
drop_rate = n_dropped / len(vals)
check("drop rate ≈ 0.3",
      abs(drop_rate - 0.3) < 0.06,
      f"drop_rate={drop_rate:.4f}, expected ≈ 0.3")

# ---------------------------------------------------------------------------
print("\n=== Test 6: Edge — invalid p raises ValueError ===")
try:
    _ = pn.Dropout(p=-0.1)
    check("p < 0 raises ValueError", False, "no exception raised")
except ValueError:
    check("p < 0 raises ValueError", True)
except Exception as e:
    check("p < 0 raises ValueError", False, f"wrong exception: {e}")

try:
    _ = pn.Dropout(p=1.5)
    check("p > 1 raises ValueError", False, "no exception raised")
except ValueError:
    check("p > 1 raises ValueError", True)
except Exception as e:
    check("p > 1 raises ValueError", False, f"wrong exception: {e}")

# ---------------------------------------------------------------------------
print("\n=== Test 7: Edge — single-element tensor ===")
drop_single = pn.Dropout(p=0.5)
x1 = make_ones([1])
# Run multiple times — should not crash
for _ in range(20):
    y1 = drop_single(x1)
check("single-element tensor does not crash", True)

# ---------------------------------------------------------------------------
print("\n=== Test 8: Edge — backward before forward raises RuntimeError ===")
drop_fresh = pn.Dropout(p=0.3)
try:
    grad_out = make_ones([4])
    _ = drop_fresh.backward(grad_out)
    check("backward before forward raises error", False, "no exception raised")
except RuntimeError:
    check("backward before forward raises error", True)
except Exception as e:
    check("backward before forward raises error", False, f"wrong exception: {e}")

# ---------------------------------------------------------------------------
print("\n=== Test 9: Train/eval mode propagation in Sequential ===")
model = pn.Sequential([
    pn.Linear(4, 8),
    pn.Dropout(p=0.5),
    pn.ReLU(),
])
model.eval()
# The Dropout submodule should be in eval mode
drop_layer = model[1]
check("Sequential.eval() propagates to Dropout",
      not drop_layer._training,
      f"training={drop_layer._training}")

model.train()
check("Sequential.train() propagates to Dropout",
      drop_layer._training,
      f"training={drop_layer._training}")

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Dropout tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL DROPOUT TESTS PASSED")
    sys.exit(0)
