#!/usr/bin/env python3
"""
Tests for OneCycleLR scheduler.

Covers the full 1cycle policy: warmup phase, cosine annealing phase,
edge cases, parameter validation, and integration with the optimizer.
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
print("=== OneCycleLR: basic construction ===")

sched = pn.OneCycleLR(max_lr=0.1, total_steps=100, pct_start=0.3)

check("initial_lr = max_lr / div_factor",
      approx(sched.initial_lr, 0.1 / 25.0))

check("max_lr stored correctly", sched.max_lr == 0.1)
check("total_steps stored correctly", sched.total_steps == 100)
check("pct_start stored correctly", sched.pct_start == 0.3)

check("warmup steps = 30 for pct_start=0.3, total=100",
      sched._warmup_steps == 30)

check("decay steps = 70", sched._decay_steps == 70)

expected_final = (0.1 / 25.0) / 1e4
check("final_lr matches initial_lr / final_div_factor",
      approx(sched.final_lr, expected_final))

# ---------------------------------------------------------------
print("\n=== OneCycleLR: warmup phase ===")

# At step 0 we should be at initial_lr
lr0 = sched.get_lr(0)
check("step 0 → initial_lr", approx(lr0, 0.1 / 25.0))

# At the last warmup step, we should be very close to max_lr
lr_end_warmup = sched.get_lr(29)
check("step 29 → close to max_lr",
      abs(lr_end_warmup - 0.1) < 0.005)

# LR should be monotonically increasing during warmup
warmup_lrs = [sched.get_lr(s) for s in range(30)]
monotonic = all(warmup_lrs[i] <= warmup_lrs[i + 1] for i in range(len(warmup_lrs) - 1))
check("LR monotonically increases during warmup", monotonic)

# Midpoint of warmup should be roughly midpoint of LR range
lr_mid = sched.get_lr(15)
expected_mid = 0.5 * (sched.initial_lr + sched.max_lr)
check("warmup midpoint is approximately linear",
      abs(lr_mid - expected_mid) < 0.005)

# ---------------------------------------------------------------
print("\n=== OneCycleLR: cosine annealing phase ===")

# Right at the start of decay (step 30), LR should equal max_lr
lr_decay_start = sched.get_lr(30)
check("step 30 (decay start) ≈ max_lr",
      abs(lr_decay_start - 0.1) < 0.01)

# At the very last step, LR should be very close to final_lr
# (not exactly equal — the cosine reaches zero at total_steps, not total_steps-1)
lr_end = sched.get_lr(99)
check("step 99 → near final_lr", abs(lr_end - sched.final_lr) < 0.001)

# LR should be monotonically decreasing during cosine annealing
decay_lrs = [sched.get_lr(s) for s in range(30, 100)]
decay_mono = all(decay_lrs[i] >= decay_lrs[i + 1] for i in range(len(decay_lrs) - 1))
check("LR monotonically decreases during annealing", decay_mono)

# ---------------------------------------------------------------
print("\n=== OneCycleLR: step() method ===")

sched2 = pn.OneCycleLR(max_lr=0.01, total_steps=50, pct_start=0.4)
collected = []
for _ in range(50):
    collected.append(sched2.step())

check("step() returns 50 LR values", len(collected) == 50)
check("first step value matches get_lr(0)",
      approx(collected[0], sched2.get_lr(0)))

# After 50 steps, the internal counter should be at 50
check("current_lr equals last step value",
      approx(sched2.current_lr, collected[-1]))

# One more step should give final_lr (past the end)
extra = sched2.step()
check("extra step past total_steps → final_lr",
      approx(extra, sched2.final_lr))

# ---------------------------------------------------------------
print("\n=== OneCycleLR: step() with explicit epoch arg ===")

sched3 = pn.OneCycleLR(max_lr=0.05, total_steps=200)
lr_at_100 = sched3.get_lr(100)
returned = sched3.step(epoch=100)
check("step(epoch=100) returns get_lr(100)",
      approx(returned, lr_at_100))

# After step(epoch=100), the internal counter should be 101
next_lr = sched3.step()
check("next step after epoch=100 gives step 101",
      approx(next_lr, sched3.get_lr(101)))

# ---------------------------------------------------------------
print("\n=== OneCycleLR: edge cases ===")

# total_steps = 1 — no warmup, no decay, just max_lr
sched_one = pn.OneCycleLR(max_lr=0.5, total_steps=1, pct_start=0.0)
lr_only = sched_one.step()
check("total_steps=1, pct_start=0 → returns final_lr or valid LR",
      lr_only >= 0)

# pct_start = 0 — no warmup at all, start at max_lr immediately
sched_no_warmup = pn.OneCycleLR(max_lr=0.1, total_steps=100, pct_start=0.0)
lr_first = sched_no_warmup.get_lr(0)
check("pct_start=0 → step 0 is max_lr", approx(lr_first, 0.1))

# pct_start = 1.0 — all warmup, no decay
sched_all_warmup = pn.OneCycleLR(max_lr=0.1, total_steps=100, pct_start=1.0)
lr_last = sched_all_warmup.get_lr(99)
check("pct_start=1.0 → last step close to max_lr",
      abs(lr_last - 0.1) < 0.005)

# Negative step clamped to 0
lr_neg = sched.get_lr(-5)
check("negative step clamped to step 0", approx(lr_neg, sched.get_lr(0)))

# Way past total_steps
lr_way_past = sched.get_lr(10000)
check("step >> total_steps → final_lr", approx(lr_way_past, sched.final_lr))

# ---------------------------------------------------------------
print("\n=== OneCycleLR: parameter validation ===")

try:
    pn.OneCycleLR(max_lr=0.1, total_steps=0)
    check("total_steps=0 raises ValueError", False)
except ValueError:
    check("total_steps=0 raises ValueError", True)

try:
    pn.OneCycleLR(max_lr=0.1, total_steps=100, pct_start=-0.1)
    check("pct_start=-0.1 raises ValueError", False)
except ValueError:
    check("pct_start=-0.1 raises ValueError", True)

try:
    pn.OneCycleLR(max_lr=0.1, total_steps=100, pct_start=1.5)
    check("pct_start=1.5 raises ValueError", False)
except ValueError:
    check("pct_start=1.5 raises ValueError", True)

# ---------------------------------------------------------------
print("\n=== OneCycleLR: custom div_factor / final_div_factor ===")

sched_custom = pn.OneCycleLR(
    max_lr=0.01, total_steps=200,
    div_factor=10.0, final_div_factor=100.0,
)
check("custom initial_lr = 0.01/10 = 0.001",
      approx(sched_custom.initial_lr, 0.001))
check("custom final_lr = 0.001/100 = 1e-5",
      approx(sched_custom.final_lr, 1e-5))

# ---------------------------------------------------------------
print("\n=== OneCycleLR: full cycle shape sanity ===")

# Run through a full cycle and verify the overall shape:
# start low → peak → end low
big_sched = pn.OneCycleLR(max_lr=1.0, total_steps=1000, pct_start=0.3)
all_lrs = [big_sched.get_lr(s) for s in range(1000)]

peak_idx = all_lrs.index(max(all_lrs))
check("peak LR is near end of warmup phase (step ~299)",
      270 <= peak_idx <= 310)

check("first LR < peak LR", all_lrs[0] < all_lrs[peak_idx])
check("last LR < first LR",
      all_lrs[-1] < all_lrs[0])

# ---------------------------------------------------------------
print("\n=== OneCycleLR: repr ===")
r = repr(sched)
check("repr contains 'OneCycleLR'", "OneCycleLR" in r)
check("repr contains max_lr value", "0.1" in r)

# ---------------------------------------------------------------
print(f"\n{'='*50}")
print(f"OneCycleLR tests: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
