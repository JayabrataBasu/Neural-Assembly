#!/usr/bin/env python3
"""
Test suite for the extended activation functions and the Tanh bug fix.

All activations are backed by assembly (activations.asm) and accessed
through ctypes.  We test forward-pass correctness by comparing against
known mathematical definitions.  We also verify repr, shape preservation,
and that each activation can be used inside a Sequential.

~55 tests.
"""

import sys, os, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyneural as pn
from pyneural import (
    Tensor, ReLU, Sigmoid, Softmax, Tanh, Sequential,
    GELU, LeakyReLU, ELU, SELU, Swish, Mish, HardSwish, Softplus, HardTanh,
)

passed = 0
failed = 0


def check(cond, name):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


# Helper: get float32 value from a 1-element tensor at flat index
def tval(t, idx=0):
    """Read a float32 value from a tensor using data_ptr."""
    import ctypes
    ptr = t.data_ptr
    if ptr is None or ptr == 0:
        raise ValueError("Null data pointer")
    fp = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
    return fp[idx]


def set_tval(t, idx, val):
    """Write a float32 value into a tensor at flat index."""
    import ctypes
    ptr = t.data_ptr
    fp = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
    fp[idx] = val


# Create a 1-D tensor with given float32 values
def make_tensor(values):
    t = Tensor.create([len(values)])  # default float32
    for i, v in enumerate(values):
        set_tval(t, i, v)
    return t


# ── Reference implementations ───────────────────────────────────

def ref_gelu(x):
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

def ref_leaky_relu(x, alpha=0.01):
    return x if x >= 0 else alpha * x

def ref_elu(x, alpha=1.0):
    return x if x > 0 else alpha * (math.exp(x) - 1.0)

def ref_selu(x):
    lam = 1.0507009873554804934193349852946
    alp = 1.6732632423543772848170429916717
    return lam * x if x > 0 else lam * alp * (math.exp(x) - 1.0)

def ref_swish(x):
    return x / (1.0 + math.exp(-x))

def ref_mish(x):
    sp = math.log(1.0 + math.exp(x))
    return x * math.tanh(sp)

def ref_hardswish(x):
    if x <= -3.0:
        return 0.0
    elif x >= 3.0:
        return x
    else:
        return x * (x + 3.0) / 6.0

def ref_softplus(x):
    return math.log(1.0 + math.exp(x))

def ref_hardtanh(x):
    if x < -1.0:
        return -1.0
    elif x > 1.0:
        return 1.0
    else:
        return x


# ── Section 1: Tanh fix ─────────────────────────────────────────

print("--- Tanh fix ---")

# Before the fix, Tanh was calling sigmoid.  tanh(0)=0, sigmoid(0)=0.5
# So this is the definitive test.
inp_tanh = make_tensor([0.0, 1.0, -1.0, 0.5])
tanh_mod = Tanh()
out_tanh = tanh_mod(inp_tanh)

check(approx(tval(out_tanh, 0), 0.0), "tanh(0) = 0  (was 0.5 with sigmoid bug)")
check(approx(tval(out_tanh, 1), math.tanh(1.0)), "tanh(1) correct")
check(approx(tval(out_tanh, 2), math.tanh(-1.0)), "tanh(-1) correct")
check(approx(tval(out_tanh, 3), math.tanh(0.5)), "tanh(0.5) correct")
check(repr(tanh_mod) == "Tanh()", "Tanh repr")


# ── Section 2: GELU ─────────────────────────────────────────────

print("\n--- GELU ---")

gelu = GELU()
inp_g = make_tensor([0.0, 1.0, -1.0, 2.0, -0.5])
out_g = gelu(inp_g)

for i, x in enumerate([0.0, 1.0, -1.0, 2.0, -0.5]):
    check(approx(tval(out_g, i), ref_gelu(x), tol=1e-3),
          f"gelu({x}) ≈ {ref_gelu(x):.4f}")

check(repr(gelu) == "GELU()", "GELU repr")


# ── Section 3: LeakyReLU ────────────────────────────────────────

print("\n--- LeakyReLU ---")

lrelu = LeakyReLU(alpha=0.1)
inp_lr = make_tensor([2.0, -3.0, 0.0, -0.5, 1.0])
out_lr = lrelu(inp_lr)

for i, x in enumerate([2.0, -3.0, 0.0, -0.5, 1.0]):
    check(approx(tval(out_lr, i), ref_leaky_relu(x, 0.1), tol=1e-3),
          f"leaky_relu({x}, α=0.1) ≈ {ref_leaky_relu(x, 0.1):.4f}")

# Default alpha
lrelu_default = LeakyReLU()
check(lrelu_default.alpha == 0.01, "LeakyReLU default alpha=0.01")
check("0.1" in repr(lrelu), "LeakyReLU repr contains alpha")


# ── Section 4: ELU ──────────────────────────────────────────────

print("\n--- ELU ---")

elu = ELU(alpha=1.0)
inp_e = make_tensor([1.0, -1.0, 0.0, -2.0, 0.5])
out_e = elu(inp_e)

for i, x in enumerate([1.0, -1.0, 0.0, -2.0, 0.5]):
    check(approx(tval(out_e, i), ref_elu(x), tol=1e-3),
          f"elu({x}) ≈ {ref_elu(x):.4f}")

# Custom alpha
elu2 = ELU(alpha=0.5)
inp_e2 = make_tensor([-1.0])
out_e2 = elu2(inp_e2)
check(approx(tval(out_e2, 0), ref_elu(-1.0, 0.5), tol=1e-3),
      "elu(-1, α=0.5) uses custom alpha")
check("ELU" in repr(elu2), "ELU repr")


# ── Section 5: SELU ─────────────────────────────────────────────

print("\n--- SELU ---")

selu = SELU()
inp_s = make_tensor([1.0, -1.0, 0.0, 2.0, -0.5])
out_s = selu(inp_s)

for i, x in enumerate([1.0, -1.0, 0.0, 2.0, -0.5]):
    check(approx(tval(out_s, i), ref_selu(x), tol=1e-3),
          f"selu({x}) ≈ {ref_selu(x):.4f}")

check(repr(selu) == "SELU()", "SELU repr")


# ── Section 6: Swish ────────────────────────────────────────────

print("\n--- Swish ---")

swish = Swish()
inp_sw = make_tensor([0.0, 1.0, -1.0, 3.0, -3.0])
out_sw = swish(inp_sw)

for i, x in enumerate([0.0, 1.0, -1.0, 3.0, -3.0]):
    check(approx(tval(out_sw, i), ref_swish(x), tol=1e-3),
          f"swish({x}) ≈ {ref_swish(x):.4f}")

check(repr(swish) == "Swish()", "Swish repr")


# ── Section 7: Mish ─────────────────────────────────────────────

print("\n--- Mish ---")

mish = Mish()
inp_m = make_tensor([0.0, 1.0, -1.0, 2.0, -2.0])
out_m = mish(inp_m)

for i, x in enumerate([0.0, 1.0, -1.0, 2.0, -2.0]):
    check(approx(tval(out_m, i), ref_mish(x), tol=1e-3),
          f"mish({x}) ≈ {ref_mish(x):.4f}")

check(repr(mish) == "Mish()", "Mish repr")


# ── Section 8: HardSwish ────────────────────────────────────────

print("\n--- HardSwish ---")

hs = HardSwish()
test_vals = [-4.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
inp_hs = make_tensor(test_vals)
out_hs = hs(inp_hs)

for i, x in enumerate(test_vals):
    check(approx(tval(out_hs, i), ref_hardswish(x), tol=1e-3),
          f"hardswish({x}) ≈ {ref_hardswish(x):.4f}")

check(repr(hs) == "HardSwish()", "HardSwish repr")


# ── Section 9: Softplus ─────────────────────────────────────────

print("\n--- Softplus ---")

sp = Softplus()
test_sp = [0.0, 1.0, -1.0, 5.0, -5.0]
inp_sp = make_tensor(test_sp)
out_sp = sp(inp_sp)

for i, x in enumerate(test_sp):
    check(approx(tval(out_sp, i), ref_softplus(x), tol=1e-3),
          f"softplus({x}) ≈ {ref_softplus(x):.4f}")

check(repr(sp) == "Softplus()", "Softplus repr")


# ── Section 10: HardTanh ────────────────────────────────────────

print("\n--- HardTanh ---")

ht = HardTanh()
test_ht = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
inp_ht = make_tensor(test_ht)
out_ht = ht(inp_ht)

for i, x in enumerate(test_ht):
    check(approx(tval(out_ht, i), ref_hardtanh(x), tol=1e-3),
          f"hardtanh({x}) ≈ {ref_hardtanh(x):.4f}")

check(repr(ht) == "HardTanh()", "HardTanh repr")


# ── Section 11: Shape preservation ──────────────────────────────

print("\n--- shape preservation ---")

# All activations should preserve the input shape
inp_2d = Tensor.create([4, 8])  # 2D tensor
for act_cls in [GELU, SELU, Swish, Mish, HardSwish, Softplus, HardTanh]:
    act = act_cls()
    out = act(inp_2d)
    check(out.shape == (4, 8), f"{act_cls.__name__} preserves [4,8] shape")

# Parameterised ones
for act in [LeakyReLU(0.2), ELU(0.5)]:
    out = act(inp_2d)
    check(out.shape == (4, 8), f"{type(act).__name__} preserves [4,8] shape")


# ── Section 12: Sequential integration ──────────────────────────

print("\n--- Sequential integration ---")

# Build a small Sequential with new activations
from pyneural import Linear
seq = Sequential([
    Linear(4, 8),
    GELU(),
    Linear(8, 4),
    Swish(),
])
inp_seq = Tensor.create([1, 4])
out_seq = seq(inp_seq)
check(out_seq.shape == (1, 4), "Sequential(Linear→GELU→Linear→Swish) runs")


# ── Section 13: GELU(0) should be 0 ─────────────────────────────

print("\n--- special values ---")

# GELU(0) = 0 exactly
inp_zero = make_tensor([0.0])
check(approx(tval(GELU()(inp_zero), 0), 0.0, tol=1e-5), "GELU(0) = 0")

# Swish(0) = 0
check(approx(tval(Swish()(inp_zero), 0), 0.0, tol=1e-5), "Swish(0) = 0")

# Mish(0) = 0
check(approx(tval(Mish()(inp_zero), 0), 0.0, tol=1e-5), "Mish(0) = 0")

# Softplus(0) = ln(2)
check(approx(tval(Softplus()(inp_zero), 0), math.log(2.0), tol=1e-3),
      "Softplus(0) = ln(2)")

# HardTanh(0) = 0
check(approx(tval(HardTanh()(inp_zero), 0), 0.0, tol=1e-5), "HardTanh(0) = 0")


# ── Summary ──────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Activation tests:  {passed} passed, {failed} failed")
print(f"{'='*50}")

sys.exit(0 if failed == 0 else 1)
