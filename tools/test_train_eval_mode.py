#!/usr/bin/env python3
"""
Tests for train/eval mode toggling (Batch 2a).

Covers:
  1. Module.train() sets _training = True
  2. Module.eval() sets _training = False
  3. Sequential propagates mode to all children recursively
  4. Nested Sequential containers propagate correctly
  5. Dropout behaviour differs between train and eval
  6. Edge: empty Sequential
  7. Edge: calling train()/eval() multiple times is idempotent
"""

import sys, os, ctypes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn

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


def make_ones(shape):
    t = pn.Tensor.zeros(shape)
    t.fill(1.0)
    return t


def tensor_to_list(t):
    numel = int(pn.core._lib.neural_tensor_numel(t._ptr))
    data_ptr = pn.core._lib.neural_tensor_data(t._ptr)
    dtype = int(pn.core._lib.neural_tensor_dtype(t._ptr))
    if dtype == 0:
        arr = (ctypes.c_float * numel).from_address(data_ptr)
    else:
        arr = (ctypes.c_double * numel).from_address(data_ptr)
    return [float(arr[i]) for i in range(numel)]


# ---------------------------------------------------------------------------
print("=== Test 1: Module.train() / eval() ===")
from pyneural.nn import Module

m = Module()
check("default is training=True", m._training is True)

m.eval()
check("eval() sets training=False", m._training is False)

m.train()
check("train() sets training=True", m._training is True)

m.train(False)
check("train(False) sets training=False", m._training is False)

# ---------------------------------------------------------------------------
print("\n=== Test 2: Sequential propagation (flat) ===")
model = pn.Sequential([
    pn.Linear(4, 8),
    pn.ReLU(),
    pn.Dropout(p=0.5),
    pn.Linear(8, 2),
])

model.eval()
for i, mod in enumerate(model._modules):
    check(f"  child[{i}] eval", not mod._training, f"training={mod._training}")

model.train()
for i, mod in enumerate(model._modules):
    check(f"  child[{i}] train", mod._training, f"training={mod._training}")

# ---------------------------------------------------------------------------
print("\n=== Test 3: Nested Sequential propagation ===")
inner = pn.Sequential([pn.Linear(2, 4), pn.Dropout(p=0.3)])
outer = pn.Sequential([inner, pn.ReLU()])

outer.eval()
check("inner Sequential in eval", not inner._training)
check("inner Linear in eval", not inner._modules[0]._training)
check("inner Dropout in eval", not inner._modules[1]._training)

outer.train()
check("inner Sequential in train", inner._training)
check("inner Linear in train", inner._modules[0]._training)
check("inner Dropout in train", inner._modules[1]._training)

# ---------------------------------------------------------------------------
print("\n=== Test 4: Dropout output changes between modes ===")
drop = pn.Dropout(p=0.5)
x = make_ones([200])

# Train mode: some elements should be zero
drop.train()
y_train = drop(x)
vals_train = tensor_to_list(y_train)
n_zero_train = sum(1 for v in vals_train if abs(v) < 1e-6)

# Eval mode: all should be 1.0
drop.eval()
y_eval = drop(x)
vals_eval = tensor_to_list(y_eval)
n_zero_eval = sum(1 for v in vals_eval if abs(v) < 1e-6)

check("train mode drops some elements", n_zero_train > 20,
      f"only {n_zero_train}/200 dropped")
check("eval mode drops no elements", n_zero_eval == 0,
      f"{n_zero_eval}/200 dropped in eval")

# ---------------------------------------------------------------------------
print("\n=== Test 5: Empty Sequential ===")
empty = pn.Sequential([])
empty.eval()
check("empty Sequential.eval() doesn't crash", not empty._training)
empty.train()
check("empty Sequential.train() doesn't crash", empty._training)

# ---------------------------------------------------------------------------
print("\n=== Test 6: Idempotent calls ===")
m2 = pn.Sequential([pn.Dropout(p=0.2)])
m2.eval()
m2.eval()
m2.eval()
check("triple eval() is fine", not m2._training)
m2.train()
m2.train()
check("double train() is fine", m2._training)

# ---------------------------------------------------------------------------
print("\n=== Test 7: train()/eval() return self (chaining) ===")
m3 = pn.Sequential([pn.Linear(2, 2)])
ret = m3.eval()
check("eval() returns self", ret is m3)
ret2 = m3.train()
check("train() returns self", ret2 is m3)

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Train/eval mode tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL TRAIN/EVAL MODE TESTS PASSED")
    sys.exit(0)
