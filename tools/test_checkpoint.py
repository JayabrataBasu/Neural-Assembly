#!/usr/bin/env python3
"""
Tests for checkpoint save/resume (Batch 2b).

Covers:
  1. save_checkpoint creates a file with correct magic header
  2. load_checkpoint restores exact same parameter values
  3. Metadata (epoch, best_loss, lr) round-trips correctly
  4. Shape mismatch → ValueError
  5. Missing file → FileNotFoundError
  6. Corrupted/invalid magic → ValueError
  7. Truncated file → IOError
  8. Functional: save at epoch N, modify weights, load, verify restored
"""

import sys, os, ctypes, struct, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyneural as pn
from pyneural.checkpoint import save_checkpoint, load_checkpoint, MAGIC

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


def get_param_values(model):
    """Get all parameter values as flat lists of floats."""
    result = []
    for p in model.parameters():
        numel = int(pn.core._lib.neural_tensor_numel(p._ptr))
        data = pn.core._lib.neural_tensor_data(p._ptr)
        dtype = int(pn.core._lib.neural_tensor_dtype(p._ptr))
        if dtype == 0:
            arr = (ctypes.c_float * numel).from_address(data)
        else:
            arr = (ctypes.c_double * numel).from_address(data)
        result.append([float(arr[i]) for i in range(numel)])
    return result


# ---------------------------------------------------------------------------
print("=== Test 1: save_checkpoint creates valid file ===")
model = pn.Sequential([pn.Linear(4, 3), pn.Linear(3, 2)])

with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
    ckpt_path = f.name

save_checkpoint(ckpt_path, model, epoch=10, best_loss=0.123, lr=0.001)

check("file exists", os.path.exists(ckpt_path))
with open(ckpt_path, "rb") as f:
    magic = f.read(8)
check("magic header correct", magic == MAGIC, f"got {magic!r}")

fsize = os.path.getsize(ckpt_path)
check("file is non-trivial size", fsize > 100, f"size={fsize}")

# ---------------------------------------------------------------------------
print("\n=== Test 2: load_checkpoint restores parameter values ===")
# Save current params
original_params = get_param_values(model)

# Modify weights to something different
for p in model.parameters():
    p.fill(999.0)

# Verify they changed
modified_params = get_param_values(model)
check("params were modified", modified_params[0][0] != original_params[0][0],
      "params didn't change after fill")

# Load checkpoint
meta = load_checkpoint(ckpt_path, model)

# Verify restored
restored_params = get_param_values(model)
all_match = True
for orig, rest in zip(original_params, restored_params):
    for a, b in zip(orig, rest):
        if abs(a - b) > 1e-6:
            all_match = False
            break

check("parameters restored exactly", all_match,
      f"first param: orig={original_params[0][:3]}, restored={restored_params[0][:3]}")

# ---------------------------------------------------------------------------
print("\n=== Test 3: Metadata round-trips ===")
check("epoch restored", meta["epoch"] == 10, f"got {meta['epoch']}")
check("best_loss restored", abs(meta["best_loss"] - 0.123) < 1e-10,
      f"got {meta['best_loss']}")
check("lr restored", abs(meta["lr"] - 0.001) < 1e-10, f"got {meta['lr']}")

# ---------------------------------------------------------------------------
print("\n=== Test 4: Shape mismatch → ValueError ===")
different_model = pn.Sequential([pn.Linear(10, 5)])
try:
    load_checkpoint(ckpt_path, different_model)
    check("shape mismatch raises ValueError", False, "no exception")
except ValueError as e:
    check("shape mismatch raises ValueError", True)
except Exception as e:
    check("shape mismatch raises ValueError", False, f"wrong exception: {e}")

# ---------------------------------------------------------------------------
print("\n=== Test 5: Missing file → FileNotFoundError ===")
try:
    load_checkpoint("/tmp/does_not_exist_xyz.ckpt", model)
    check("missing file raises FileNotFoundError", False, "no exception")
except FileNotFoundError:
    check("missing file raises FileNotFoundError", True)
except Exception as e:
    check("missing file raises FileNotFoundError", False, f"wrong: {e}")

# ---------------------------------------------------------------------------
print("\n=== Test 6: Corrupted magic → ValueError ===")
with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
    bad_path = f.name
    f.write(b"BADMAGIC" + b"\x00" * 100)

try:
    load_checkpoint(bad_path, model)
    check("bad magic raises ValueError", False, "no exception")
except ValueError as e:
    check("bad magic raises ValueError", "magic" in str(e).lower() or "invalid" in str(e).lower())
except Exception as e:
    check("bad magic raises ValueError", False, f"wrong: {e}")
finally:
    os.unlink(bad_path)

# ---------------------------------------------------------------------------
print("\n=== Test 7: Truncated file → IOError or ValueError ===")
with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
    trunc_path = f.name
    # Write valid header but truncate before tensor data
    f.write(MAGIC)
    f.write(struct.pack("<I", 1))   # version
    f.write(struct.pack("<I", 4))   # num_tensors (wrong, too many)
    f.write(struct.pack("<I", 5))   # epoch
    f.write(struct.pack("<d", 0.5)) # best_loss
    f.write(struct.pack("<d", 0.01))# lr
    # No tensor data - truncated

try:
    load_checkpoint(trunc_path, model)
    check("truncated file raises error", False, "no exception")
except (IOError, ValueError, struct.error) as e:
    check("truncated file raises error", True)
except Exception as e:
    check("truncated file raises error", False, f"wrong: {type(e).__name__}: {e}")
finally:
    os.unlink(trunc_path)

# ---------------------------------------------------------------------------
print("\n=== Test 8: Functional round-trip with different epochs ===")
model2 = pn.Sequential([pn.Linear(2, 4), pn.ReLU(), pn.Linear(4, 1)])
with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
    ckpt2 = f.name

# Save at "epoch 25"
save_checkpoint(ckpt2, model2, epoch=25, best_loss=0.05)
params_at_25 = get_param_values(model2)

# Simulate continued training (modify weights)
for p in model2.parameters():
    p.fill(0.0)

# Restore from epoch 25
meta2 = load_checkpoint(ckpt2, model2)
params_restored = get_param_values(model2)

match = all(
    abs(a - b) < 1e-6
    for orig_layer, rest_layer in zip(params_at_25, params_restored)
    for a, b in zip(orig_layer, rest_layer)
)
check("round-trip preserves params", match)
check("round-trip preserves epoch", meta2["epoch"] == 25)

os.unlink(ckpt2)

# ---------------------------------------------------------------------------
# Cleanup
os.unlink(ckpt_path)

print(f"\n{'='*50}")
print(f"Checkpoint tests: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("ALL CHECKPOINT TESTS PASSED")
    sys.exit(0)
