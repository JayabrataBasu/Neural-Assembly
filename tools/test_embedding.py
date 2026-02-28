#!/usr/bin/env python3
"""
Tests for Embedding layer (Batch 5b).

Covers:
  1. Forward: correct lookup (known weights)
  2. Forward: Python list API
  3. Backward: gradient accumulation
  4. Backward: duplicate indices accumulate
  5. Edge: single embedding, single dim
  6. Edge: large vocabulary
  7. Invalid index → error
  8. Invalid constructor args → error
  9. backward before forward → error
 10. Weight pointer accessible
 11. repr
"""

import sys, os, ctypes, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyneural.nn import Embedding

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


def make_i64(values):
    return (ctypes.c_int64 * len(values))(*values)


def make_f64(values):
    return (ctypes.c_double * len(values))(*values)


def f64_read(ptr, n):
    arr = (ctypes.c_double * n).from_address(ptr)
    return [float(arr[i]) for i in range(n)]


def f64_write(ptr, values):
    arr = (ctypes.c_double * len(values)).from_address(ptr)
    for i, v in enumerate(values):
        arr[i] = v


# ---------------------------------------------------------------------------
print("=== Test 1: Forward with known weights ===")
emb = Embedding(4, 3)  # 4 words × 3 dims

# Set weights to known values: row i = [i*10+1, i*10+2, i*10+3]
wptr = emb.weight_ptr
for i in range(4):
    for d in range(3):
        val = i * 10.0 + d + 1
        arr = (ctypes.c_double * 1).from_address(wptr + (i * 3 + d) * 8)
        arr[0] = val

# Lookup indices [2, 0, 3]
result = emb([2, 0, 3])
expected = [21.0, 22.0, 23.0, 1.0, 2.0, 3.0, 31.0, 32.0, 33.0]
check("forward [2,0,3] matches known weights",
      all(abs(a - b) < 1e-10 for a, b in zip(result, expected)),
      f"got {result}")

# ---------------------------------------------------------------------------
print("\n=== Test 2: Forward output length ===")
emb2 = Embedding(10, 5)
out2 = emb2([0, 3, 7])
check("output length = 3 * 5 = 15", len(out2) == 15)

# ---------------------------------------------------------------------------
print("\n=== Test 3: Backward gradient accumulation ===")
emb3 = Embedding(3, 2)
# Set weights to known: [0,1], [2,3], [4,5]
wptr3 = emb3.weight_ptr
f64_write(wptr3, [0, 1, 2, 3, 4, 5])

# Forward with indices [1, 2]
indices = make_i64([1, 2])
out_arr = make_f64([0.0] * 4)
emb3.forward_f64(ctypes.cast(indices, ctypes.c_void_p), 2,
                  ctypes.cast(out_arr, ctypes.c_void_p))

# Backward: grad_output = [1,1, 2,2]
grad_out = make_f64([1.0, 1.0, 2.0, 2.0])
grad_w = make_f64([0.0] * 6)  # 3×2 zeroed
emb3.backward_f64(
    ctypes.cast(grad_out, ctypes.c_void_p),
    ctypes.cast(grad_w, ctypes.c_void_p),
)
gw = [float(grad_w[i]) for i in range(6)]
# Row 0: untouched → [0, 0]
# Row 1: gets grad_out[0] = [1, 1]
# Row 2: gets grad_out[1] = [2, 2]
expected_gw = [0, 0, 1, 1, 2, 2]
check("backward grad_weight correct",
      all(abs(a - b) < 1e-10 for a, b in zip(gw, expected_gw)),
      f"got {gw}")

# ---------------------------------------------------------------------------
print("\n=== Test 4: Backward with duplicate indices ===")
emb4 = Embedding(3, 2)
indices4 = make_i64([1, 1, 1])
out4 = make_f64([0.0] * 6)
emb4.forward_f64(ctypes.cast(indices4, ctypes.c_void_p), 3,
                  ctypes.cast(out4, ctypes.c_void_p))

grad_out4 = make_f64([1.0, 0.0, 0.0, 2.0, 3.0, 3.0])
grad_w4 = make_f64([0.0] * 6)
emb4.backward_f64(
    ctypes.cast(grad_out4, ctypes.c_void_p),
    ctypes.cast(grad_w4, ctypes.c_void_p),
)
gw4 = [float(grad_w4[i]) for i in range(6)]
# Row 1 accumulates: [1+0+3, 0+2+3] = [4, 5]
check("duplicate idx row 0 = [0,0]", gw4[0] == 0.0 and gw4[1] == 0.0)
check("duplicate idx row 1 accumulated [4,5]",
      abs(gw4[2] - 4.0) < 1e-10 and abs(gw4[3] - 5.0) < 1e-10,
      f"got [{gw4[2]}, {gw4[3]}]")
check("duplicate idx row 2 = [0,0]", gw4[4] == 0.0 and gw4[5] == 0.0)

# ---------------------------------------------------------------------------
print("\n=== Test 5: Edge — single embedding, single dim ===")
emb5 = Embedding(1, 1)
f64_write(emb5.weight_ptr, [42.0])
out5 = emb5([0])
check("1×1 embedding forward", abs(out5[0] - 42.0) < 1e-10, f"got {out5}")

# ---------------------------------------------------------------------------
print("\n=== Test 6: Edge — large vocabulary ===")
emb6 = Embedding(10000, 8)
out6 = emb6([0, 5000, 9999])
check("large vocab output length", len(out6) == 24)
check("large vocab output finite", all(math.isfinite(v) for v in out6))

# ---------------------------------------------------------------------------
print("\n=== Test 7: Invalid index → error ===")
emb7 = Embedding(5, 3)
caught = False
try:
    emb7([-1])
except Exception:
    caught = True
check("negative index raises error", caught)

caught = False
try:
    emb7([5])  # out of range (valid: 0-4)
except Exception:
    caught = True
check("out-of-range index raises error", caught)

# ---------------------------------------------------------------------------
print("\n=== Test 8: Invalid constructor args ===")
caught = False
try:
    Embedding(0, 5)
except ValueError:
    caught = True
check("num_embeddings=0 raises ValueError", caught)

caught = False
try:
    Embedding(5, 0)
except ValueError:
    caught = True
check("embedding_dim=0 raises ValueError", caught)

# ---------------------------------------------------------------------------
print("\n=== Test 9: backward before forward ===")
emb9 = Embedding(3, 2)
caught = False
try:
    gout = make_f64([0.0] * 4)
    gw9 = make_f64([0.0] * 6)
    emb9.backward_f64(ctypes.cast(gout, ctypes.c_void_p),
                       ctypes.cast(gw9, ctypes.c_void_p))
except RuntimeError:
    caught = True
check("backward before forward raises RuntimeError", caught)

# ---------------------------------------------------------------------------
print("\n=== Test 10: Weight pointer accessible ===")
emb10 = Embedding(5, 4)
wptr10 = emb10.weight_ptr
check("weight_ptr is non-null", wptr10 is not None and wptr10 != 0)
weights = f64_read(wptr10, 20)
check("weights are all finite", all(math.isfinite(w) for w in weights))

# ---------------------------------------------------------------------------
print("\n=== Test 11: repr ===")
r = repr(Embedding(100, 32))
check("repr contains Embedding", "Embedding" in r)
check("repr contains 100", "100" in r)
check("repr contains 32", "32" in r)

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Embedding tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL EMBEDDING TESTS PASSED")
    sys.exit(0)
