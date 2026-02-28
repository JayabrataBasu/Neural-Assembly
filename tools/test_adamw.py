#!/usr/bin/env python3
"""
tests for AdamW optimizer — construction, defaults, step, repr.

we test through the pyneural wrapper, not raw ctypes, so this also
exercises the binding layer in core.py and optim.py.
"""

import sys
import os
import struct
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyneural import AdamW, Adam
from pyneural.tensor import Tensor
from pyneural.core import _lib

passed = 0
failed = 0

def check(cond, msg):
    global passed, failed
    if cond:
        print(f"  PASS: {msg}")
        passed += 1
    else:
        print(f"  FAIL: {msg}")
        failed += 1


def tval(t, idx):
    """Read a float32 element from a tensor."""
    ptr = _lib.neural_tensor_data(t._ptr)
    raw = ctypes.string_at(ptr + idx * 4, 4)
    return struct.unpack("f", raw)[0]


def set_tval(t, idx, val):
    """Write a float32 element into a tensor."""
    ptr = _lib.neural_tensor_data(t._ptr)
    packed = struct.pack("f", val)
    ctypes.memmove(ptr + idx * 4, packed, 4)


# ── Section 1: Basic construction ────────────────────────────────

print("\n--- AdamW construction ---")

opt = AdamW()
check(opt._ptr is not None and opt._ptr != 0, "default constructor returns valid pointer")
check(opt.learning_rate == 0.001, "default lr = 0.001")
check(opt.beta1 == 0.9, "default beta1 = 0.9")
check(opt.beta2 == 0.999, "default beta2 = 0.999")
check(opt.epsilon == 1e-8, "default epsilon = 1e-8")
check(opt.weight_decay == 0.01, "default weight_decay = 0.01")


# ── Section 2: Custom parameters ────────────────────────────────

print("\n--- AdamW custom params ---")

opt2 = AdamW(learning_rate=0.01, beta1=0.95, beta2=0.99, epsilon=1e-6, weight_decay=0.1)
check(opt2.learning_rate == 0.01, "custom lr = 0.01")
check(opt2.beta1 == 0.95, "custom beta1 = 0.95")
check(opt2.beta2 == 0.99, "custom beta2 = 0.99")
check(opt2.epsilon == 1e-6, "custom epsilon = 1e-6")
check(opt2.weight_decay == 0.1, "custom weight_decay = 0.1")


# ── Section 3: repr ─────────────────────────────────────────────

print("\n--- AdamW repr ---")

r = repr(opt)
check("AdamW" in r, "repr contains 'AdamW'")
check("0.001" in r, "repr shows lr")
check("0.9" in r, "repr shows beta1")
check("0.999" in r, "repr shows beta2")
check("wd=" in r or "weight_decay" in r.lower() or "0.01" in r, "repr shows weight_decay")


# ── Section 4: AdamW is distinct from Adam ───────────────────────

print("\n--- AdamW vs Adam ---")

adam = Adam(learning_rate=0.001)
adamw = AdamW(learning_rate=0.001)
check(type(adam).__name__ == "Adam", "Adam type name is 'Adam'")
check(type(adamw).__name__ == "AdamW", "AdamW type name is 'AdamW'")
check(hasattr(adamw, "weight_decay"), "AdamW has weight_decay attribute")
check(not hasattr(adam, "weight_decay"), "Adam does NOT have weight_decay")


# ── Section 5: step with tiny tensors ────────────────────────────

print("\n--- AdamW step ---")

# set up a 1-element param and gradient
param = Tensor.create([4])
grad = Tensor.create([4])

# fill the param with 1.0 and gradient with 0.1
for i in range(4):
    set_tval(param, i, 1.0)
    set_tval(grad, i, 0.1)

opt_step = AdamW(learning_rate=0.01, weight_decay=0.01)

# single step should not crash
try:
    opt_step.step([param], [grad])
    check(True, "step() completes without error")
except Exception as e:
    check(False, f"step() raised: {e}")

# param should have changed (either from the adam update or weight decay)
p0 = tval(param, 0)
check(abs(p0 - 1.0) > 1e-6, f"param changed after step (now {p0:.6f})")

# run a few more steps — params should keep moving
prev_val = p0
for _ in range(5):
    for i in range(4):
        set_tval(grad, i, 0.1)
    opt_step.step([param], [grad])

p0_after = tval(param, 0)
check(abs(p0_after - prev_val) > 1e-6, f"param keeps changing over 5 steps (now {p0_after:.6f})")


# ── Section 6: step with mismatched lengths should error ─────────

print("\n--- AdamW step validation ---")

try:
    opt_step.step([param], [grad, grad])
    check(False, "mismatched params/grads should raise ValueError")
except ValueError:
    check(True, "mismatched params/grads raises ValueError")
except Exception as e:
    check(False, f"mismatched params/grads raised wrong exception: {type(e).__name__}")


# ── Section 7: multiple param groups ─────────────────────────────

print("\n--- AdamW multi-param step ---")

p1 = Tensor.create([3])
p2 = Tensor.create([5])
g1 = Tensor.create([3])
g2 = Tensor.create([5])

for i in range(3):
    set_tval(p1, i, 2.0)
    set_tval(g1, i, -0.5)
for i in range(5):
    set_tval(p2, i, -1.0)
    set_tval(g2, i, 0.3)

opt_multi = AdamW(learning_rate=0.005, weight_decay=0.05)

try:
    opt_multi.step([p1, p2], [g1, g2])
    check(True, "multi-param step completes")
except Exception as e:
    check(False, f"multi-param step raised: {e}")

# both should have changed
check(abs(tval(p1, 0) - 2.0) > 1e-6, "p1 changed after multi-param step")
check(abs(tval(p2, 0) - (-1.0)) > 1e-6, "p2 changed after multi-param step")


# ── Section 8: zero weight decay should behave like Adam ─────────

print("\n--- AdamW with wd=0 vs Adam ---")

# This is a sanity check — with wd=0, AdamW should give results very
# close to Adam (not exactly identical due to floating-point order, but
# both should at least move in the same direction).

pa = Tensor.create([4])
pb = Tensor.create([4])
ga = Tensor.create([4])
gb = Tensor.create([4])

for i in range(4):
    set_tval(pa, i, 1.0)
    set_tval(pb, i, 1.0)
    set_tval(ga, i, 0.5)
    set_tval(gb, i, 0.5)

opt_adam = Adam(learning_rate=0.01)
opt_adamw0 = AdamW(learning_rate=0.01, weight_decay=0.0)

opt_adam.step([pa], [ga])
opt_adamw0.step([pb], [gb])

# both should have moved in the same direction
va = tval(pa, 0)
vb = tval(pb, 0)
check(abs(va - vb) < 0.01, f"AdamW(wd=0) ≈ Adam: {va:.6f} vs {vb:.6f}")


# ── Summary ──────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"AdamW tests:  {passed} passed, {failed} failed")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
