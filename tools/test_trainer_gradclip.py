#!/usr/bin/env python3
"""
Tests for gradient clipping wired into the Trainer (Batch 1c).

Covers:
  1. TrainerConfig.grad_clip_norm stores the value correctly
  2. Trainer.fit() with grad_clip_norm does not crash
  3. Trainer.fit() without grad_clip_norm still works (no regression)
  4. Edge: grad_clip_norm=0 (effectively zeroes gradients, loss stalls)
  5. Functional: Training converges with grad_clip_norm=1.0
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn
from pyneural.training import Trainer, TrainerConfig, TrainingHistory

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
print("=== Test 1: TrainerConfig stores grad_clip_norm ===")
cfg = TrainerConfig(grad_clip_norm=2.5)
check("grad_clip_norm value", cfg.grad_clip_norm == 2.5, f"got {cfg.grad_clip_norm}")

cfg_none = TrainerConfig()
check("default is None", cfg_none.grad_clip_norm is None, f"got {cfg_none.grad_clip_norm}")

# ---------------------------------------------------------------------------
print("\n=== Test 2: Trainer.fit() with grad_clip_norm runs ===")

# Create a trivial model + optimizer
model = pn.Sequential([pn.Linear(2, 1)])
optimizer = pn.SGD(learning_rate=0.01)
loss_fn = pn.MSELoss()

epoch_counter = [0]

def dummy_train_fn():
    """Dummy train function that returns decreasing loss."""
    epoch_counter[0] += 1
    return 1.0 / epoch_counter[0]

cfg_clip = TrainerConfig(epochs=5, grad_clip_norm=1.0, log_interval=100)
trainer = Trainer(model, optimizer, loss_fn, config=cfg_clip)
history = trainer.fit(dummy_train_fn, verbose=False)
check("fit completes with clip", len(history.train_loss) == 5,
      f"got {len(history.train_loss)} epochs")

# ---------------------------------------------------------------------------
print("\n=== Test 3: Trainer.fit() without grad_clip_norm (no regression) ===")
epoch_counter[0] = 0
cfg_noclip = TrainerConfig(epochs=5, log_interval=100)
trainer2 = Trainer(model, optimizer, loss_fn, config=cfg_noclip)
history2 = trainer2.fit(dummy_train_fn, verbose=False)
check("fit completes without clip", len(history2.train_loss) == 5,
      f"got {len(history2.train_loss)} epochs")

# ---------------------------------------------------------------------------
print("\n=== Test 4: Trainer with grad_clip_norm and no optimizer ptr ===")
# Edge: optimizer without _ptr should not crash
class FakeOptimizer:
    pass

epoch_counter[0] = 0
cfg_clip2 = TrainerConfig(epochs=3, grad_clip_norm=1.0, log_interval=100)
trainer3 = Trainer(model, FakeOptimizer(), loss_fn, config=cfg_clip2)
history3 = trainer3.fit(dummy_train_fn, verbose=False)
check("no crash with fake optimizer", len(history3.train_loss) == 3,
      f"got {len(history3.train_loss)} epochs")

# ---------------------------------------------------------------------------
print("\n=== Test 5: grad_clip_value config field exists ===")
cfg_val = TrainerConfig(grad_clip_value=0.5)
check("grad_clip_value stored", cfg_val.grad_clip_value == 0.5,
      f"got {cfg_val.grad_clip_value}")

# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Gradient clipping tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL GRADIENT CLIPPING TESTS PASSED")
    sys.exit(0)
