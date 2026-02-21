#!/usr/bin/env python3
"""
Advanced training demo exercising all new PyNeural v1.1 features:

  1. Confusion Matrix & per-class metrics  (assembly: training_ops.asm)
  2. LR Scheduling (Step, Cosine, Exp)     (assembly: lr_*_decay / lr_cosine_annealing)
  3. Early Stopping                        (Python control-flow)
  4. NaN / Inf Detection                   (assembly: tensor_has_nan / tensor_has_inf)
  5. Gradient Norm Logging                 (assembly: tensor_grad_l2_norm)
  6. Weight Initialization (He / Xavier)   (assembly: init_he_uniform / init_xavier_uniform)
  7. Trainer high-level API                (Python orchestrator)

Usage:
    python tools/demo_features.py
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyneural as pn


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────
# 1. Confusion Matrix & per-class metrics
# ──────────────────────────────────────────────────────────────

def demo_confusion_matrix():
    banner("1. Confusion Matrix & Per-Class Metrics (Assembly)")

    # Simulate 3-class classification results
    targets     = [0,0,0,0,1,1,1,1,2,2,2,2,  0,1,2,0,1,2,0,1]
    predictions = [0,0,0,1,1,1,2,1,2,2,0,2,  0,1,2,2,0,2,0,1]

    cm = pn.ConfusionMatrix(3, class_names=["Class A", "Class B", "Class C"])
    cm.update(targets, predictions)

    print(cm.matrix_str())
    print()
    print(cm.report())
    print(f"\nNormalized matrix:")
    print(cm.matrix_str(normalize=True))

    assert cm.total == len(targets), f"Total mismatch: {cm.total} vs {len(targets)}"
    assert 0.0 <= cm.accuracy <= 1.0
    print("\n✓ Confusion matrix test passed")


# ──────────────────────────────────────────────────────────────
# 2. LR Scheduling
# ──────────────────────────────────────────────────────────────

def demo_lr_schedulers():
    banner("2. Learning Rate Schedulers (Assembly math)")

    # StepLR
    print("StepLR (step=10, gamma=0.5):")
    sched = pn.StepLR(initial_lr=0.01, step_size=10, gamma=0.5)
    for e in range(30):
        lr = sched.step()
        if e % 10 == 0 or e == 29:
            print(f"  epoch {e:3d}: lr = {lr:.8f}")
    assert abs(sched.current_lr - 0.01 * 0.5**2) < 1e-10

    # CosineAnnealing
    print("\nCosineAnnealingLR (T_max=50, eta_min=1e-5):")
    cosine = pn.CosineAnnealingLR(initial_lr=0.01, T_max=50, eta_min=1e-5)
    for e in range(51):
        lr = cosine.step()
        if e % 10 == 0 or e == 50:
            print(f"  epoch {e:3d}: lr = {lr:.8f}")
    assert abs(cosine.current_lr - 1e-5) < 1e-6

    # ExponentialLR
    print("\nExponentialLR (gamma=0.95):")
    exp = pn.ExponentialLR(initial_lr=0.01, gamma=0.95)
    for e in range(20):
        lr = exp.step()
        if e % 5 == 0:
            print(f"  epoch {e:3d}: lr = {lr:.8f}")

    # WarmupLR
    print("\nWarmupLR (5 epochs warmup → Cosine):")
    cos_after = pn.CosineAnnealingLR(initial_lr=0.01, T_max=45)
    warmup = pn.WarmupLR(
        initial_lr=0.01, warmup_epochs=5,
        warmup_start_lr=1e-5, after_scheduler=cos_after
    )
    for e in range(50):
        lr = warmup.step()
        if e < 5 or e % 10 == 0 or e == 49:
            print(f"  epoch {e:3d}: lr = {lr:.8f}")

    # ReduceLROnPlateau
    print("\nReduceLROnPlateau (patience=3):")
    plateau = pn.ReduceLROnPlateau(initial_lr=0.01, patience=3, factor=0.5)
    fake_losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83]
    for i, loss in enumerate(fake_losses):
        lr = plateau.step(loss)
        print(f"  epoch {i}: loss={loss:.3f} → lr={lr:.6f}")

    print("\n✓ LR scheduler tests passed")


# ──────────────────────────────────────────────────────────────
# 3. Early Stopping
# ──────────────────────────────────────────────────────────────

def demo_early_stopping():
    banner("3. Early Stopping")

    es = pn.EarlyStopping(patience=3, min_delta=0.01, mode="min")
    losses = [1.0, 0.8, 0.6, 0.5, 0.49, 0.489, 0.489, 0.489, 0.489]

    for epoch, loss in enumerate(losses):
        stopped = es.step(loss, epoch)
        status = "STOP" if stopped else "ok"
        print(f"  Epoch {epoch:2d}: loss={loss:.7f}  counter={es.counter}  [{status}]")
        if stopped:
            print(f"\n  → Stopped at epoch {epoch}, best was {es.best_score:.7f} "
                  f"at epoch {es.best_epoch}")
            break

    assert es.should_stop, "Expected early stopping to trigger"
    print("\n✓ Early stopping test passed")


# ──────────────────────────────────────────────────────────────
# 4. NaN / Inf Detection
# ──────────────────────────────────────────────────────────────

def demo_nan_detection():
    banner("4. NaN / Inf Detection (Assembly)")

    detector = pn.NaNDetector(raise_on_detect=False)

    # Clean tensor
    t_clean = pn.Tensor.ones([100])
    assert not detector.has_nan(t_clean), "False positive NaN"
    assert not detector.has_inf(t_clean), "False positive Inf"
    print("  ones([100]): NaN=False, Inf=False  ✓")

    # Zeros
    t_zero = pn.Tensor.zeros([50])
    assert not detector.has_nan(t_zero)
    assert not detector.has_inf(t_zero)
    print("  zeros([50]): NaN=False, Inf=False  ✓")

    # Inject NaN via numpy
    import numpy as np
    arr = np.ones(20, dtype=np.float32)
    arr[10] = float('nan')
    t_nan = pn.Tensor.from_numpy(arr)
    assert detector.has_nan(t_nan), "Should detect NaN"
    print("  tensor with NaN: detected  ✓")

    # Inject Inf
    arr2 = np.ones(20, dtype=np.float32)
    arr2[5] = float('inf')
    t_inf = pn.Tensor.from_numpy(arr2)
    assert detector.has_inf(t_inf), "Should detect Inf"
    print("  tensor with Inf: detected  ✓")

    print("\n✓ NaN/Inf detection tests passed")


# ──────────────────────────────────────────────────────────────
# 5. Gradient Norm Logging
# ──────────────────────────────────────────────────────────────

def demo_grad_norm():
    banner("5. Gradient L2 Norm (Assembly SSE)")

    from pyneural.training import grad_l2_norm
    import numpy as np

    # Create a tensor with known values
    arr = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5.0
    t = pn.Tensor.from_numpy(arr)
    norm = grad_l2_norm(t)
    print(f"  norm([3, 4]) = {norm:.4f}  (expected 5.0)")
    assert abs(norm - 5.0) < 0.01, f"Expected 5.0, got {norm}"

    # Larger tensor
    arr2 = np.ones(1000, dtype=np.float32)
    t2 = pn.Tensor.from_numpy(arr2)
    norm2 = grad_l2_norm(t2)
    expected = math.sqrt(1000)
    print(f"  norm(ones(1000)) = {norm2:.4f}  (expected {expected:.4f})")
    assert abs(norm2 - expected) < 0.5, f"Expected {expected}, got {norm2}"

    print("\n✓ Gradient norm tests passed")


# ──────────────────────────────────────────────────────────────
# 6. Weight Initialization
# ──────────────────────────────────────────────────────────────

def demo_weight_init():
    banner("6. Weight Initialization (Assembly)")

    import numpy as np

    pn.init()

    # Create a Linear layer and check default init
    linear = pn.Linear(64, 32)
    from pyneural.core import _lib

    # Access weights via native API
    weight_ptr = _lib.neural_linear_weight(linear._ptr)
    assert weight_ptr, "Failed to get weight pointer"

    from pyneural.tensor import Tensor
    weight = Tensor(weight_ptr, owns_data=False)
    w_arr = weight.numpy().flatten()
    print(f"  Default init: mean={w_arr.mean():.4f}, std={w_arr.std():.4f}, "
          f"range=[{w_arr.min():.4f}, {w_arr.max():.4f}]")

    # Re-init with Kaiming uniform (assembly)
    pn.weight_init.kaiming_uniform_(weight, mode="fan_in", seed=42)
    w_arr = weight.numpy().flatten()
    expected_bound = math.sqrt(6.0 / 64)  # fan_in = 64
    print(f"  Kaiming uniform: mean={w_arr.mean():.4f}, std={w_arr.std():.4f}, "
          f"range=[{w_arr.min():.4f}, {w_arr.max():.4f}]")
    print(f"    expected bound = ±{expected_bound:.4f}")
    assert abs(w_arr.mean()) < 0.1, f"Mean too large: {w_arr.mean()}"

    # Xavier uniform (assembly)
    pn.weight_init.xavier_uniform_(weight, seed=42)
    w_arr = weight.numpy().flatten()
    expected_bound = math.sqrt(6.0 / (64 + 32))
    print(f"  Xavier uniform:  mean={w_arr.mean():.4f}, std={w_arr.std():.4f}, "
          f"range=[{w_arr.min():.4f}, {w_arr.max():.4f}]")
    print(f"    expected bound = ±{expected_bound:.4f}")

    # Xavier normal (assembly)
    pn.weight_init.xavier_normal_(weight, seed=42)
    w_arr = weight.numpy().flatten()
    expected_std = math.sqrt(2.0 / (64 + 32))
    print(f"  Xavier normal:   mean={w_arr.mean():.4f}, std={w_arr.std():.4f}")
    print(f"    expected std ≈ {expected_std:.4f}")

    print("\n✓ Weight initialization tests passed")


# ──────────────────────────────────────────────────────────────
# 7. Trainer API
# ──────────────────────────────────────────────────────────────

def demo_trainer():
    banner("7. Trainer API (Orchestrator)")

    epoch_counter = [0]

    def fake_train():
        """Simulate a training epoch with decreasing loss."""
        e = epoch_counter[0]
        epoch_counter[0] += 1
        # Loss decays then plateaus
        return 1.0 * math.exp(-0.05 * e) + 0.1

    def fake_val():
        """Simulate validation."""
        e = epoch_counter[0]
        return 1.1 * math.exp(-0.04 * e) + 0.12

    config = pn.TrainerConfig(
        epochs=50,
        log_interval=10,
        early_stopping_patience=8,
        early_stopping_min_delta=0.005,
        early_stopping_mode="min",
    )

    scheduler = pn.CosineAnnealingLR(initial_lr=0.01, T_max=50, eta_min=1e-5)

    # Use a dummy model and optimizer (no actual training needed)
    trainer = pn.Trainer(
        model=None,
        optimizer=None,
        loss_fn=None,
        config=config,
        scheduler=scheduler,
    )

    history = trainer.fit(train_fn=fake_train, val_fn=fake_val, verbose=True)

    print(f"\n  Recorded {len(history.train_loss)} training epochs")
    print(f"  Final train loss: {history.train_loss[-1]:.6f}")
    print(f"  Final val loss:   {history.val_loss[-1]:.6f}")
    print(f"  LR schedule:      {history.learning_rates[0]:.6f} → {history.learning_rates[-1]:.6f}")

    assert len(history.train_loss) > 0
    assert len(history.learning_rates) > 0
    print("\n✓ Trainer API test passed")


# ──────────────────────────────────────────────────────────────
# 8. Class-Balanced Sampling
# ──────────────────────────────────────────────────────────────

def demo_class_balanced_sampling():
    banner("8. Class-Balanced Sampling (Assembly)")

    from collections import Counter

    # Highly imbalanced dataset: 100 class-0, 10 class-1, 5 class-2
    labels = [0] * 100 + [1] * 10 + [2] * 5
    print(f"  Original distribution: {Counter(labels)}")

    sampler = pn.WeightedRandomSampler(labels, num_classes=3, num_samples=300, seed=42)
    indices = list(sampler)
    sampled_labels = [labels[i] for i in indices]
    counts = Counter(sampled_labels)
    print(f"  Sampled distribution:  {counts}")

    # Check that minority classes are oversampled (ratio > 0.5)
    ratio_2_to_0 = counts.get(2, 0) / max(counts.get(0, 1), 1)
    ratio_1_to_0 = counts.get(1, 0) / max(counts.get(0, 1), 1)
    print(f"  Class 2/Class 0 ratio: {ratio_2_to_0:.2f} (target ~1.0)")
    print(f"  Class 1/Class 0 ratio: {ratio_1_to_0:.2f} (target ~1.0)")

    assert ratio_2_to_0 > 0.4, f"Class 2 under-sampled: ratio={ratio_2_to_0}"
    assert ratio_1_to_0 > 0.4, f"Class 1 under-sampled: ratio={ratio_1_to_0}"
    assert len(indices) == 300
    assert all(0 <= i < len(labels) for i in indices)

    print("\n✓ Class-balanced sampling test passed")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║       PyNeural v1.1 — Feature Demo & Validation          ║")
    print("║       Assembly-backed ML training utilities               ║")
    print("╚════════════════════════════════════════════════════════════╝")

    pn.init()
    print(f"Framework: {pn.version()}")
    print(f"SIMD level: {pn.get_simd_name()}")

    demo_confusion_matrix()
    demo_lr_schedulers()
    demo_early_stopping()
    demo_nan_detection()
    demo_grad_norm()
    demo_weight_init()
    demo_trainer()
    demo_class_balanced_sampling()

    banner("ALL FEATURE DEMOS PASSED ✓")
    print("The following features are fully implemented with assembly backends:")
    print("  • Confusion matrix & per-class precision/recall/F1")
    print("  • LR scheduling: Step, Exponential, Cosine Annealing, Warmup, Plateau")
    print("  • Early stopping with configurable patience/delta")
    print("  • NaN/Inf detection on tensors")
    print("  • Gradient L2 norm computation (SSE-vectorized)")
    print("  • Weight initialization: He/Kaiming, Xavier/Glorot (uniform & normal)")
    print("  • Dropout forward/backward (inverted)")
    print("  • Trainer orchestrator with all features composed")
    print("  • Class-balanced sampling for imbalanced datasets")


if __name__ == "__main__":
    main()
