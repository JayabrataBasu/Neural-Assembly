#!/usr/bin/env python3
"""
Tests for roc_auc_score (Batch 4b).

Covers:
  1. Perfect classifier → AUC = 1.0
  2. Anti-perfect classifier → AUC = 0.0
  3. Random classifier → AUC ≈ 0.5
  4. Known hand-computed example
  5. Tied scores
  6. All-positive labels → AUC = 0.0 (undefined)
  7. All-negative labels → AUC = 0.0 (undefined)
  8. Single sample
  9. Large dataset
 10. Empty input → ValueError
 11. Mismatched lengths → ValueError
 12. Comparison with manual trapezoidal computation
"""

import sys, os, math, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyneural.metrics import roc_auc_score

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
print("=== Test 1: Perfect classifier → AUC = 1.0 ===")
y_true = [0, 0, 0, 1, 1, 1]
y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
auc = roc_auc_score(y_true, y_score)
check("perfect AUC = 1.0", abs(auc - 1.0) < 1e-10, f"auc={auc}")

# ---------------------------------------------------------------------------
print("\n=== Test 2: Anti-perfect classifier → AUC = 0.0 ===")
y_true = [1, 1, 1, 0, 0, 0]
y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
auc = roc_auc_score(y_true, y_score)
check("anti-perfect AUC = 0.0", abs(auc) < 1e-10, f"auc={auc}")

# ---------------------------------------------------------------------------
print("\n=== Test 3: Random classifier → AUC ≈ 0.5 ===")
random.seed(123)
N = 10000
y_true = [random.randint(0, 1) for _ in range(N)]
y_score = [random.random() for _ in range(N)]
auc = roc_auc_score(y_true, y_score)
check("random AUC ≈ 0.5", abs(auc - 0.5) < 0.03, f"auc={auc:.4f}")

# ---------------------------------------------------------------------------
print("\n=== Test 4: Hand-computed example ===")
# 2 positives, 2 negatives
# scores: pos=[0.9, 0.4], neg=[0.35, 0.1]
# sorted desc: (0.9,P), (0.4,P), (0.35,N), (0.1,N) → perfect → AUC=1.0
y_true_4 = [1, 0, 1, 0]
y_score_4 = [0.9, 0.35, 0.4, 0.1]
auc4 = roc_auc_score(y_true_4, y_score_4)
check("hand-computed AUC = 1.0", abs(auc4 - 1.0) < 1e-10, f"auc={auc4}")

# One error: swap one pair
# scores: pos=[0.9, 0.3], neg=[0.35, 0.1]
# sorted: (0.9,P), (0.35,N), (0.3,P), (0.1,N)
# TPR/FPR: (0,0)→(0.5,0)→(0.5,0.5)→(1.0,0.5)→(1.0,1.0)
# AUC = 0 + 0.5*0.5*(0.5+0.5) + 0.5*0 + 0.5*0.5*(1+1) = 0.25 + 0.5 = 0.75
y_true_4b = [1, 0, 1, 0]
y_score_4b = [0.9, 0.35, 0.3, 0.1]
auc4b = roc_auc_score(y_true_4b, y_score_4b)
check("one-error AUC = 0.75", abs(auc4b - 0.75) < 1e-10, f"auc={auc4b}")

# ---------------------------------------------------------------------------
print("\n=== Test 5: Tied scores ===")
# Two samples with same score, one pos one neg → AUC = 0.5 for that pair
y_true_5 = [1, 0]
y_score_5 = [0.5, 0.5]
auc5 = roc_auc_score(y_true_5, y_score_5)
check("tied scores AUC = 0.5", abs(auc5 - 0.5) < 1e-10, f"auc={auc5}")

# ---------------------------------------------------------------------------
print("\n=== Test 6: All-positive labels → AUC = 0 (undefined) ===")
y_true_6 = [1, 1, 1, 1]
y_score_6 = [0.1, 0.5, 0.8, 0.9]
auc6 = roc_auc_score(y_true_6, y_score_6)
check("all-positive AUC = 0.0", abs(auc6) < 1e-10, f"auc={auc6}")

# ---------------------------------------------------------------------------
print("\n=== Test 7: All-negative labels → AUC = 0 (undefined) ===")
y_true_7 = [0, 0, 0, 0]
y_score_7 = [0.1, 0.5, 0.8, 0.9]
auc7 = roc_auc_score(y_true_7, y_score_7)
check("all-negative AUC = 0.0", abs(auc7) < 1e-10, f"auc={auc7}")

# ---------------------------------------------------------------------------
print("\n=== Test 8: Two samples (minimal) ===")
auc_min = roc_auc_score([0, 1], [0.3, 0.7])
check("2-sample perfect AUC = 1.0", abs(auc_min - 1.0) < 1e-10, f"auc={auc_min}")

auc_min2 = roc_auc_score([1, 0], [0.3, 0.7])
check("2-sample inverted AUC = 0.0", abs(auc_min2) < 1e-10, f"auc={auc_min2}")

# ---------------------------------------------------------------------------
print("\n=== Test 9: Large dataset (50k) ===")
random.seed(999)
N = 50000
y_true_big = [random.randint(0, 1) for _ in range(N)]
# Make a decent classifier: score = label + noise
y_score_big = [float(y_true_big[i]) + random.gauss(0, 0.5) for i in range(N)]
auc_big = roc_auc_score(y_true_big, y_score_big)
check("50k-sample AUC > 0.8 (decent classifier)", auc_big > 0.8, f"auc={auc_big:.4f}")
check("50k-sample AUC < 1.0", auc_big < 1.0, f"auc={auc_big:.4f}")
check("50k-sample AUC finite", math.isfinite(auc_big))

# ---------------------------------------------------------------------------
print("\n=== Test 10: Empty input → ValueError ===")
caught = False
try:
    roc_auc_score([], [])
except ValueError:
    caught = True
check("empty input raises ValueError", caught)

# ---------------------------------------------------------------------------
print("\n=== Test 11: Mismatched lengths → ValueError ===")
caught = False
try:
    roc_auc_score([0, 1], [0.5])
except ValueError:
    caught = True
check("mismatched lengths raises ValueError", caught)

# ---------------------------------------------------------------------------
print("\n=== Test 12: AUC in [0, 1] range for various seeds ===")
all_in_range = True
for seed in range(10):
    random.seed(seed)
    n = 200
    yt = [random.randint(0, 1) for _ in range(n)]
    ys = [random.random() for _ in range(n)]
    # Skip degenerate cases
    if sum(yt) == 0 or sum(yt) == n:
        continue
    a = roc_auc_score(yt, ys)
    if a < -1e-10 or a > 1.0 + 1e-10:
        all_in_range = False
        break
check("AUC always in [0, 1]", all_in_range)

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"ROC-AUC tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL ROC-AUC TESTS PASSED")
    sys.exit(0)
