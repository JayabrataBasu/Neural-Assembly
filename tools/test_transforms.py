#!/usr/bin/env python3
"""
Tests for Data Transforms (Batch 5a).

Covers:
  1. compute_stats: mean, std, min, max
  2. Normalize: z-score output mean≈0, std≈1
  3. Normalize: inverse round-trip
  4. Normalize: auto-fit on first call
  5. MinMaxScale: output in [0, 1]
  6. MinMaxScale: known values
  7. MinMaxScale: constant feature (all same value)
  8. Compose: chains transforms
  9. Single-sample edge case
 10. Single-feature edge case
 11. Pre-fitted Normalize with custom mean/std
 12. Empty / invalid inputs
"""

import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyneural.transforms import Normalize, MinMaxScale, Compose, compute_stats

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


# ---------------------------------------------------------------------------
print("=== Test 1: compute_stats ===")
# 3 samples × 2 features
data = [
    1.0, 10.0,
    2.0, 20.0,
    3.0, 30.0,
]
stats = compute_stats(data, num_features=2)

check("mean[0] = 2.0", abs(stats['mean'][0] - 2.0) < 1e-10)
check("mean[1] = 20.0", abs(stats['mean'][1] - 20.0) < 1e-10)
# std = sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = sqrt(2/3) ≈ 0.8165
check("std[0] ≈ 0.8165", abs(stats['std'][0] - math.sqrt(2.0 / 3.0)) < 1e-6,
      f"got {stats['std'][0]}")
check("min[0] = 1.0", abs(stats['min'][0] - 1.0) < 1e-10)
check("max[0] = 3.0", abs(stats['max'][0] - 3.0) < 1e-10)
check("min[1] = 10.0", abs(stats['min'][1] - 10.0) < 1e-10)
check("max[1] = 30.0", abs(stats['max'][1] - 30.0) < 1e-10)

# ---------------------------------------------------------------------------
print("\n=== Test 2: Normalize → z-score mean≈0, var≈1 ===")
import random
random.seed(42)
N, F = 200, 3
raw = [random.gauss(50, 10) for _ in range(N * F)]

norm = Normalize()
out = norm(raw, F)

# Check per-feature stats of output
for f in range(F):
    vals = [out[i * F + f] for i in range(N)]
    m = sum(vals) / N
    v = sum((x - m) ** 2 for x in vals) / N
    check(f"feature {f} mean≈0", abs(m) < 0.05, f"mean={m:.6f}")
    check(f"feature {f} var≈1", abs(v - 1.0) < 0.05, f"var={v:.6f}")

# ---------------------------------------------------------------------------
print("\n=== Test 3: Normalize inverse round-trip ===")
reconstructed = norm.inverse(out, F)
max_err = max(abs(a - b) for a, b in zip(raw, reconstructed))
check("round-trip max error < 1e-8", max_err < 1e-8, f"max_err={max_err}")

# ---------------------------------------------------------------------------
print("\n=== Test 4: Normalize auto-fit on first call ===")
norm2 = Normalize()
check("not fitted initially", not norm2._fitted)
_ = norm2(data, 2)
check("fitted after first call", norm2._fitted)
check("mean stored", norm2.mean is not None)
check("std stored", norm2.std is not None)

# ---------------------------------------------------------------------------
print("\n=== Test 5: MinMaxScale output in [0, 1] ===")
scaler = MinMaxScale()
scaled = scaler(raw, F)
check("all values >= 0", all(v >= -1e-10 for v in scaled))
check("all values <= 1", all(v <= 1.0 + 1e-10 for v in scaled))

# ---------------------------------------------------------------------------
print("\n=== Test 6: MinMaxScale known values ===")
# data: [0, 100, 50, 100, 100, 100]  → 3 samples × 2 features
# feature 0: [0, 50, 100] → min=0, max=100 → [0, 0.5, 1.0]
# feature 1: [100, 100, 100] → constant → all ≈ 0 (due to eps)
simple_data = [0.0, 100.0, 50.0, 100.0, 100.0, 100.0]
scaler2 = MinMaxScale()
scaled2 = scaler2(simple_data, 2)
check("feat 0 sample 0 ≈ 0.0", abs(scaled2[0]) < 1e-6)
check("feat 0 sample 1 ≈ 0.5", abs(scaled2[2] - 0.5) < 1e-6)
check("feat 0 sample 2 ≈ 1.0", abs(scaled2[4] - 1.0) < 1e-6)

# ---------------------------------------------------------------------------
print("\n=== Test 7: MinMaxScale constant feature ===")
const_data = [5.0, 5.0, 5.0]  # 3 samples × 1 feature, all same
scaler3 = MinMaxScale()
scaled3 = scaler3(const_data, 1)
# (5 - 5) / (5 - 5 + eps) ≈ 0
check("constant feature → ≈ 0", all(abs(v) < 1e-4 for v in scaled3),
      f"got {scaled3}")

# ---------------------------------------------------------------------------
print("\n=== Test 8: Compose chains transforms ===")
# Compose Normalize + MinMaxScale — results should be valid
pipeline = Compose([Normalize(), MinMaxScale()])
composed = pipeline(raw, F)
check("Compose result has correct length", len(composed) == N * F)
check("Compose result all finite", all(math.isfinite(v) for v in composed))

# ---------------------------------------------------------------------------
print("\n=== Test 9: Single-sample ===")
single = [1.0, 2.0, 3.0]
stats_s = compute_stats(single, 3)
check("single-sample mean", stats_s['mean'] == [1.0, 2.0, 3.0])
check("single-sample std=0", all(s == 0.0 for s in stats_s['std']))

# ---------------------------------------------------------------------------
print("\n=== Test 10: Single-feature ===")
one_feat = [10.0, 20.0, 30.0]
norm_1f = Normalize()
out_1f = norm_1f(one_feat, 1)
m_1f = sum(out_1f) / 3
check("single-feature mean≈0", abs(m_1f) < 1e-6)

# ---------------------------------------------------------------------------
print("\n=== Test 11: Pre-fitted Normalize ===")
custom_norm = Normalize(mean=[0.0], std=[2.0])
check("pre-fitted is fitted", custom_norm._fitted)
out_custom = custom_norm([4.0, 6.0], 1)
# (4 - 0) / (2 + 1e-8) ≈ 2.0, (6 - 0) / (2 + 1e-8) ≈ 3.0
check("custom mean/std: [4]→≈2.0", abs(out_custom[0] - 2.0) < 1e-6)
check("custom mean/std: [6]→≈3.0", abs(out_custom[1] - 3.0) < 1e-6)

# ---------------------------------------------------------------------------
print("\n=== Test 12: Invalid inputs ===")
caught = False
try:
    compute_stats([], 2)
except ValueError:
    caught = True
check("empty data raises ValueError", caught)

caught = False
try:
    compute_stats([1, 2, 3], 2)  # 3 not divisible by 2
except ValueError:
    caught = True
check("non-divisible length raises ValueError", caught)

# ---------------------------------------------------------------------------
print("\n=== Test 13: repr ===")
check("Normalize repr", "fitted" in repr(Normalize()))
check("MinMaxScale repr", "fitted" in repr(MinMaxScale()))
check("Compose repr", "Compose" in repr(Compose([Normalize()])))

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Transform tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL TRANSFORM TESTS PASSED")
    sys.exit(0)
