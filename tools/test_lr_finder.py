#!/usr/bin/env python3
"""
Tests for LRFinder (learning-rate range test).

Covers construction, LR sweep, loss recording, early stopping on
divergence, suggestion logic, edge cases, and the results property.
"""
import sys, os, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------
print("=== LRFinder: basic construction ===")

finder = pn.LRFinder(start_lr=1e-7, end_lr=10.0, num_steps=100)

check("start_lr stored correctly", finder.start_lr == 1e-7)
check("end_lr stored correctly", finder.end_lr == 10.0)
check("num_steps stored correctly", finder.num_steps == 100)
check("smooth_factor defaults to 0.05", finder.smooth_factor == 0.05)
check("diverge_threshold defaults to 4.0", finder.diverge_threshold == 4.0)
check("initially empty lrs list", len(finder.lrs) == 0)
check("initially empty losses list", len(finder.losses) == 0)

# ---------------------------------------------------------------
print("\n=== LRFinder: LR sweep is exponential ===")

lr_0 = finder.get_lr(0)
lr_last = finder.get_lr(99)

check("get_lr(0) == start_lr", approx(lr_0, 1e-7))
check("get_lr(99) ≈ end_lr", approx(lr_last, 10.0, tol=1e-3))

# Check that consecutive LRs have a constant ratio (log-linear)
ratio_a = finder.get_lr(1) / finder.get_lr(0)
ratio_b = finder.get_lr(50) / finder.get_lr(49)
check("constant ratio between consecutive LRs",
      approx(ratio_a, ratio_b, tol=1e-4))

# LR is monotonically increasing
lrs = [finder.get_lr(i) for i in range(100)]
mono = all(lrs[i] < lrs[i + 1] for i in range(99))
check("LR is strictly increasing", mono)

# ---------------------------------------------------------------
print("\n=== LRFinder: recording losses (decreasing) ===")

finder2 = pn.LRFinder(start_lr=1e-5, end_lr=1.0, num_steps=50)

# Simulate a training run where loss decreases then plateaus
# This mimics a well-behaved loss curve.
losses = []
for i in range(50):
    # Starts at 2.0, drops toward 0.5 with some noise
    noise = 0.05 * math.sin(i * 0.3)
    loss = 2.0 * math.exp(-0.05 * i) + 0.5 + noise
    losses.append(loss)
    diverged = finder2.record(i, loss)
    if diverged:
        break

check("recorded all 50 steps (no early stop for well-behaved loss)",
      len(finder2.lrs) == 50)

check("lrs length matches losses length",
      len(finder2.lrs) == len(finder2.losses))

check("smoothed_losses same length",
      len(finder2.smoothed_losses) == len(finder2.losses))

# ---------------------------------------------------------------
print("\n=== LRFinder: early stopping on divergence ===")

finder3 = pn.LRFinder(start_lr=1e-5, end_lr=1.0, num_steps=100,
                       diverge_threshold=3.0)

# Feed it a loss that starts low and then suddenly blows up
stopped_at = -1
for i in range(100):
    if i < 20:
        loss = 1.0 - 0.02 * i  # decreasing from 1.0 to 0.6
    else:
        loss = 0.6 + 0.5 * (i - 20)  # shoot up rapidly
    if finder3.record(i, loss):
        stopped_at = i
        break

check("early stopping triggered", stopped_at > 0)
check("stopped before recording all 100 steps",
      len(finder3.lrs) < 100)
check("stopped somewhere after the inflection point",
      stopped_at >= 20)

# ---------------------------------------------------------------
print("\n=== LRFinder: suggestion() with good curve ===")

# Build a synthetic loss curve that has a clear sweet spot around
# step 25 out of 100 — that's where loss drops fastest.
finder4 = pn.LRFinder(start_lr=1e-7, end_lr=10.0, num_steps=100)

for i in range(100):
    # Loss drops steeply between steps 20-30, then stabilises
    if i < 20:
        loss = 3.0 - 0.01 * i
    elif i < 30:
        loss = 2.8 - 0.2 * (i - 20)  # steep drop
    elif i < 60:
        loss = 0.8 + 0.001 * (i - 30)  # plateau
    else:
        loss = 0.83 + 0.1 * (i - 60)  # starts to rise
    if finder4.record(i, loss):
        break

suggested = finder4.suggestion()
check("suggestion returns a float", isinstance(suggested, float))
check("suggestion is positive", suggested > 0)

# The suggested LR should be in the region where the steepest drop
# happens (steps 20-35ish), i.e. LR around get_lr(20) to get_lr(35)
lr_low = finder4.get_lr(15)
lr_high = finder4.get_lr(40)
check("suggested LR is in the sweet spot region",
      lr_low <= suggested <= lr_high)

# ---------------------------------------------------------------
print("\n=== LRFinder: suggestion() when loss never decreases ===")

finder5 = pn.LRFinder(start_lr=1e-5, end_lr=1.0, num_steps=20,
                       diverge_threshold=100.0)

# Feed monotonically increasing losses — no good LR
for i in range(20):
    loss = 1.0 + 0.1 * i
    finder5.record(i, loss)

# Should still return *something* — falls back to min-loss index
suggested_bad = finder5.suggestion()
check("suggestion still works when loss never decreases",
      isinstance(suggested_bad, float) and suggested_bad > 0)

# In this case, min loss is at step 0, so suggested should be start_lr
check("fallback suggestion ≈ start_lr",
      approx(suggested_bad, finder5.get_lr(0), tol=1e-4))

# ---------------------------------------------------------------
print("\n=== LRFinder: suggestion() needs at least 3 steps ===")

finder_tiny = pn.LRFinder(start_lr=1e-5, end_lr=1.0, num_steps=10)
finder_tiny.record(0, 1.0)
finder_tiny.record(1, 0.9)

try:
    finder_tiny.suggestion()
    check("suggestion with <3 steps raises RuntimeError", False)
except RuntimeError:
    check("suggestion with <3 steps raises RuntimeError", True)

# Now add one more
finder_tiny.record(2, 0.8)
try:
    result = finder_tiny.suggestion()
    check("suggestion works with exactly 3 steps",
          isinstance(result, float))
except RuntimeError:
    check("suggestion works with exactly 3 steps", False)

# ---------------------------------------------------------------
print("\n=== LRFinder: results property ===")

res = finder4.results
check("results has 'lrs' key", "lrs" in res)
check("results has 'losses' key", "losses" in res)
check("results has 'smoothed_losses' key", "smoothed_losses" in res)
check("results lrs is a list", isinstance(res["lrs"], list))
check("results lengths consistent",
      len(res["lrs"]) == len(res["losses"]) == len(res["smoothed_losses"]))

# ---------------------------------------------------------------
print("\n=== LRFinder: parameter validation ===")

try:
    pn.LRFinder(start_lr=-1.0, end_lr=1.0)
    check("negative start_lr raises ValueError", False)
except ValueError:
    check("negative start_lr raises ValueError", True)

try:
    pn.LRFinder(start_lr=0.0, end_lr=1.0)
    check("zero start_lr raises ValueError", False)
except ValueError:
    check("zero start_lr raises ValueError", True)

try:
    pn.LRFinder(start_lr=1.0, end_lr=0.5)
    check("start_lr > end_lr raises ValueError", False)
except ValueError:
    check("start_lr > end_lr raises ValueError", True)

try:
    pn.LRFinder(start_lr=0.5, end_lr=0.5)
    check("start_lr == end_lr raises ValueError", False)
except ValueError:
    check("start_lr == end_lr raises ValueError", True)

try:
    pn.LRFinder(start_lr=1e-5, end_lr=1.0, num_steps=1)
    check("num_steps=1 raises ValueError", False)
except ValueError:
    check("num_steps=1 raises ValueError", True)

# ---------------------------------------------------------------
print("\n=== LRFinder: custom smooth_factor ===")

# With smooth_factor=0, the smoothed loss should track the raw loss
# (after bias correction converges)
finder_raw = pn.LRFinder(start_lr=1e-5, end_lr=1.0, num_steps=10,
                          smooth_factor=1.0)
for i in range(10):
    finder_raw.record(i, float(i + 1))

# With smooth_factor=1.0 the EMA just tracks the last value exactly
check("smooth_factor=1.0 → last smoothed ≈ last raw loss",
      approx(finder_raw.smoothed_losses[-1], 10.0, tol=0.5))

# ---------------------------------------------------------------
print("\n=== LRFinder: repr ===")

r = repr(finder)
check("repr contains 'LRFinder'", "LRFinder" in r)
check("repr contains start_lr", "1e-07" in r or "1e-7" in r)
check("repr shows recorded count", "recorded=0" in r)

finder_repr2 = pn.LRFinder(start_lr=1e-5, end_lr=1.0, num_steps=5)
finder_repr2.record(0, 1.0)
finder_repr2.record(1, 0.9)
r2 = repr(finder_repr2)
check("repr updates recorded count", "recorded=2" in r2)

# ---------------------------------------------------------------
print(f"\n{'='*50}")
print(f"LRFinder tests: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
