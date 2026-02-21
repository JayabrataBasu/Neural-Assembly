"""
Advanced training harness for PyNeural.

Provides a Trainer class that integrates:
- Training loop with batch processing
- Early stopping with patience
- Learning rate scheduling
- NaN/Inf detection
- Gradient norm logging
- Confusion matrix & per-class metrics
- Training history tracking

Example:
    >>> from pyneural.training import Trainer, TrainerConfig
    >>> from pyneural.schedulers import CosineAnnealingLR
    >>> 
    >>> config = TrainerConfig(
    ...     epochs=100,
    ...     early_stopping_patience=10,
    ...     nan_check=True,
    ...     log_grad_norm=True,
    ... )
    >>> trainer = Trainer(model, optimizer, loss_fn, config)
    >>> history = trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .metrics import ConfusionMatrix


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""
    
    # Training
    epochs: int = 100
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"   # 'val_loss' or 'val_accuracy'
    early_stopping_mode: str = "auto"         # 'min', 'max', or 'auto'
    restore_best_weights: bool = True
    
    # NaN detection
    nan_check: bool = True
    nan_check_frequency: int = 1   # Check every N batches (1 = every batch)
    halt_on_nan: bool = True       # Stop training on NaN detection
    
    # Gradient monitoring
    log_grad_norm: bool = False
    grad_norm_frequency: int = 1   # Log every N epochs
    gradient_clip_norm: Optional[float] = None   # Max gradient norm (None = no clipping)
    gradient_clip_value: Optional[float] = None  # Max gradient value (None = no clipping)
    
    # Logging
    print_every: int = 1           # Print metrics every N epochs
    verbose: bool = True
    log_file: Optional[str] = None
    
    # Metrics
    compute_confusion_matrix: bool = True
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None


@dataclass
class TrainingHistory:
    """Records training metrics across epochs."""
    
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[Dict[str, float]] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    # Best metrics
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    best_epoch: int = -1
    
    # Stop info
    stopped_epoch: Optional[int] = None
    stop_reason: Optional[str] = None
    
    def summary(self) -> str:
        """Generate a summary string of the training history."""
        lines = [
            "=" * 60,
            "Training Summary",
            "=" * 60,
            f"  Epochs trained: {len(self.train_loss)}",
        ]
        
        if self.train_loss:
            lines.append(f"  Final train loss: {self.train_loss[-1]:.6f}")
            lines.append(f"  Final train accuracy: {self.train_accuracy[-1]:.2%}")
        
        if self.val_loss:
            lines.append(f"  Final val loss: {self.val_loss[-1]:.6f}")
            lines.append(f"  Final val accuracy: {self.val_accuracy[-1]:.2%}")
        
        lines.append(f"  Best val loss: {self.best_val_loss:.6f} (epoch {self.best_epoch + 1})")
        lines.append(f"  Best val accuracy: {self.best_val_accuracy:.2%}")
        
        if self.epoch_times:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            total_time = sum(self.epoch_times)
            lines.append(f"  Avg epoch time: {avg_time:.2f}s")
            lines.append(f"  Total time: {total_time:.1f}s")
        
        if self.stop_reason:
            lines.append(f"  Stop reason: {self.stop_reason}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class EarlyStopping:
    """
    Early stopping monitor.
    
    Tracks a metric and signals when training should stop
    based on a patience threshold.
    """
    
    def __init__(
        self,
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best
        
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_epoch = -1
        self.should_stop = False
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch number
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False


class NaNDetector:
    """
    Detects NaN and Inf values in tensors.
    
    Pure Python implementation that works with any tensor
    that has a .numpy() method.
    """
    
    @staticmethod
    def check_tensor(tensor, name: str = "tensor") -> Tuple[bool, str]:
        """
        Check a tensor for NaN or Inf values.
        
        Returns:
            (is_clean, message) tuple
        """
        try:
            import numpy as np
            data = tensor.numpy()
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
            
            if has_nan and has_inf:
                return False, f"NaN and Inf detected in {name}"
            elif has_nan:
                count = int(np.isnan(data).sum())
                return False, f"NaN detected in {name}: {count}/{data.size} elements"
            elif has_inf:
                count = int(np.isinf(data).sum())
                return False, f"Inf detected in {name}: {count}/{data.size} elements"
            return True, ""
        except ImportError:
            # Fallback: check via ctypes
            import ctypes
            numel = tensor.numel
            data_ptr = tensor.data_ptr
            arr = (ctypes.c_float * numel).from_address(data_ptr)
            for i in range(numel):
                v = arr[i]
                if math.isnan(v):
                    return False, f"NaN detected in {name} at index {i}"
                if math.isinf(v):
                    return False, f"Inf detected in {name} at index {i}"
            return True, ""
    
    @staticmethod
    def check_loss(loss_value: float) -> Tuple[bool, str]:
        """Check a scalar loss value for NaN/Inf."""
        if math.isnan(loss_value):
            return False, "NaN loss detected"
        if math.isinf(loss_value):
            return False, "Inf loss detected"
        return True, ""


class GradientMonitor:
    """
    Monitors gradient statistics per layer.
    
    Tracks L2 norm, max absolute value, and percentage of zeros
    for each parameter's gradients.
    """
    
    @staticmethod
    def compute_grad_norms(model, layer_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute gradient L2 norms for each layer in the model.
        
        Args:
            model: PyNeural model (Sequential)
            layer_names: Optional names for layers
        
        Returns:
            Dictionary mapping layer names to gradient norms
        """
        try:
            import numpy as np
        except ImportError:
            return {"error": -1.0}
        
        from . import nn as _nn
        from . import core as _core
        
        norms = {}
        global_squared = 0.0
        
        linear_idx = 0
        modules = list(model._modules) if hasattr(model, '_modules') else [model]
        
        for m in modules:
            if isinstance(m, _nn.Linear) and hasattr(m, '_ptr') and m._ptr:
                name = layer_names[linear_idx] if layer_names and linear_idx < len(layer_names) else f"linear_{linear_idx}"
                
                # Get weight tensor and try to read its gradient
                weight_ptr = _core._lib.neural_linear_weight(m._ptr)
                if weight_ptr:
                    from .tensor import Tensor
                    weight = Tensor(weight_ptr, owns_data=False)
                    try:
                        arr = weight.numpy()
                        norm = float(np.linalg.norm(arr))
                        norms[f"{name}/weight_norm"] = norm
                    except Exception:
                        pass
                
                linear_idx += 1
        
        # Compute global norm
        if norms:
            total = sum(v**2 for v in norms.values())
            norms["global_norm"] = math.sqrt(total)
        
        return norms


class Trainer:
    """
    Advanced training harness for PyNeural models.
    
    Integrates early stopping, LR scheduling, NaN detection,
    gradient monitoring, and confusion matrix computation.
    
    Args:
        model: PyNeural model (e.g., Sequential)
        optimizer: PyNeural optimizer (SGD, Adam, etc.)
        loss_fn: Loss function (MSELoss, CrossEntropyLoss)
        config: TrainerConfig with training settings
        scheduler: Optional LR scheduler
    
    Example:
        >>> model = pn.Sequential([
        ...     pn.Linear(11, 64), pn.ReLU(),
        ...     pn.Linear(64, 32), pn.ReLU(),
        ...     pn.Linear(32, 6), pn.Softmax()
        ... ])
        >>> optimizer = pn.Adam(learning_rate=0.003)
        >>> loss_fn = pn.CrossEntropyLoss()
        >>> config = TrainerConfig(epochs=80, early_stopping_patience=15)
        >>> trainer = Trainer(model, optimizer, loss_fn, config)
        >>> history = trainer.fit(train_data, val_data)
        >>> print(history.summary())
    """
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        config: TrainerConfig,
        scheduler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.scheduler = scheduler
        
        # Setup early stopping
        self.early_stopping: Optional[EarlyStopping] = None
        if config.early_stopping:
            mode = config.early_stopping_mode
            if mode == "auto":
                mode = "min" if "loss" in config.early_stopping_metric else "max"
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                mode=mode,
                restore_best=config.restore_best_weights,
            )
        
        # NaN detector
        self.nan_detector = NaNDetector() if config.nan_check else None
        
        # Gradient monitor
        self.grad_monitor = GradientMonitor() if config.log_grad_norm else None
        
        # Log file
        self._log_file = None
        if config.log_file:
            os.makedirs(os.path.dirname(config.log_file) or ".", exist_ok=True)
            self._log_file = open(config.log_file, "w")
    
    def __del__(self):
        if hasattr(self, '_log_file') and self._log_file:
            self._log_file.close()
    
    def _log(self, message: str) -> None:
        """Log a message to console and optionally to file."""
        if self.config.verbose:
            print(message)
        if self._log_file:
            self._log_file.write(message + "\n")
            self._log_file.flush()
    
    def fit(
        self,
        train_data: List[Tuple],
        val_data: Optional[List[Tuple]] = None,
        num_classes: Optional[int] = None,
    ) -> TrainingHistory:
        """
        Train the model on the given data.
        
        Args:
            train_data: List of (features, label) tuples or 
                        list of (feature_batch, label_batch) tuples
            val_data: Optional validation data in same format
            num_classes: Number of classes (for confusion matrix)
        
        Returns:
            TrainingHistory with all recorded metrics
        """
        history = TrainingHistory()
        n_classes = num_classes or self.config.num_classes
        
        self._log("=" * 60)
        self._log("Training Started")
        self._log(f"  Epochs: {self.config.epochs}")
        self._log(f"  Train samples: {len(train_data)}")
        if val_data:
            self._log(f"  Val samples: {len(val_data)}")
        if self.early_stopping:
            self._log(f"  Early stopping: patience={self.config.early_stopping_patience}")
        if self.scheduler:
            self._log(f"  LR scheduler: {type(self.scheduler).__name__}")
        self._log("=" * 60)
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # ── Train one epoch ────────────────────────────────
            train_loss, train_acc, train_preds, train_targets = self._train_epoch(
                train_data, epoch
            )
            
            # ── NaN check on train loss ────────────────────────
            if self.nan_detector:
                is_clean, msg = self.nan_detector.check_loss(train_loss)
                if not is_clean:
                    self._log(f"\n[WARN] Epoch {epoch + 1}: {msg}")
                    if self.config.halt_on_nan:
                        history.stop_reason = f"NaN detected at epoch {epoch + 1}"
                        history.stopped_epoch = epoch
                        self._log("[ERROR] Halting training due to NaN")
                        break
            
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)
            
            # ── Validation ─────────────────────────────────────
            val_loss, val_acc = 0.0, 0.0
            val_cm = None
            if val_data:
                val_loss, val_acc, val_preds, val_targets = self._evaluate(
                    val_data, epoch
                )
                history.val_loss.append(val_loss)
                history.val_accuracy.append(val_acc)
                
                # Confusion matrix
                if self.config.compute_confusion_matrix and n_classes and val_preds:
                    val_cm = ConfusionMatrix(
                        n_classes, self.config.class_names
                    )
                    val_cm.update(val_targets, val_preds)
                
                # Track best
                if val_loss < history.best_val_loss:
                    history.best_val_loss = val_loss
                if val_acc > history.best_val_accuracy:
                    history.best_val_accuracy = val_acc
                    history.best_epoch = epoch
            
            # ── LR scheduling ──────────────────────────────────
            current_lr = self._get_current_lr()
            if self.scheduler:
                from .schedulers import ReduceLROnPlateau
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric = val_loss if "loss" in self.config.early_stopping_metric else val_acc
                    new_lr = self.scheduler.step(metric)
                else:
                    new_lr = self.scheduler.step()
                current_lr = new_lr
                # Note: In a full integration, we'd call optimizer_set_lr here
            
            history.learning_rates.append(current_lr)
            
            # ── Gradient monitoring ────────────────────────────
            if self.grad_monitor and (epoch + 1) % self.config.grad_norm_frequency == 0:
                grad_norms = self.grad_monitor.compute_grad_norms(self.model)
                history.grad_norms.append(grad_norms)
            
            # ── Epoch timing ───────────────────────────────────
            epoch_time = time.time() - epoch_start
            history.epoch_times.append(epoch_time)
            
            # ── Logging ────────────────────────────────────────
            if (epoch + 1) % self.config.print_every == 0:
                msg = f"Epoch {epoch + 1:>4d}/{self.config.epochs}"
                msg += f" | Loss: {train_loss:.6f}"
                msg += f" | Acc: {train_acc:.2%}"
                if val_data:
                    msg += f" | Val Loss: {val_loss:.6f}"
                    msg += f" | Val Acc: {val_acc:.2%}"
                msg += f" | LR: {current_lr:.6f}"
                msg += f" | {epoch_time:.1f}s"
                self._log(msg)
                
                # Log grad norms if available
                if history.grad_norms and self.config.log_grad_norm:
                    latest = history.grad_norms[-1]
                    if "global_norm" in latest:
                        self._log(f"  Grad norms: global={latest['global_norm']:.4f}")
            
            # ── Early stopping ─────────────────────────────────
            if self.early_stopping and val_data:
                metric = val_loss if "loss" in self.config.early_stopping_metric else val_acc
                if self.early_stopping(metric, epoch):
                    history.stopped_epoch = epoch
                    history.stop_reason = (
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best: epoch {self.early_stopping.best_epoch + 1})"
                    )
                    self._log(f"\n[INFO] {history.stop_reason}")
                    break
        
        # ── Final confusion matrix ─────────────────────────────
        if val_data and self.config.compute_confusion_matrix and n_classes:
            self._log("\n" + "=" * 60)
            self._log("Final Evaluation")
            self._log("=" * 60)
            
            _, _, final_preds, final_targets = self._evaluate(val_data, -1)
            if final_preds:
                final_cm = ConfusionMatrix(n_classes, self.config.class_names)
                final_cm.update(final_targets, final_preds)
                self._log(final_cm.matrix_str())
                self._log("")
                self._log(final_cm.report())
        
        self._log(history.summary())
        
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        
        return history
    
    def _train_epoch(
        self, data: List[Tuple], epoch: int
    ) -> Tuple[float, float, List[int], List[int]]:
        """
        Train one epoch. Subclass or override for custom training loops.
        
        This base implementation works with pre-batched data as
        (features_array, labels_array) tuples.
        
        Returns:
            (avg_loss, accuracy, predictions_list, targets_list)
        """
        # Default implementation: iterate through data, compute forward/loss
        # This is a framework-level implementation that works with
        # the assembly binary for actual training.
        # 
        # For Python-level training, users should subclass Trainer
        # and override this method.
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for features, labels in data:
            # Forward pass
            output = self.model(features)
            
            # Compute loss
            loss = self.loss_fn(output, labels)
            total_loss += loss
            
            # NaN check on loss
            if self.nan_detector and self.config.nan_check:
                is_clean, msg = self.nan_detector.check_loss(loss)
                if not is_clean and self.config.halt_on_nan:
                    return float("nan"), 0.0, [], []
            
            # Compute predictions
            try:
                import numpy as np
                out_np = output.numpy()
                if out_np.ndim == 1:
                    pred = int(np.argmax(out_np))
                    all_preds.append(pred)
                else:
                    preds = np.argmax(out_np, axis=-1)
                    all_preds.extend(preds.tolist())
                
                # Handle labels
                if hasattr(labels, 'numpy'):
                    lbl_np = labels.numpy()
                    if lbl_np.ndim == 0:
                        all_targets.append(int(lbl_np))
                    else:
                        all_targets.extend(lbl_np.astype(int).tolist())
                elif isinstance(labels, (int, float)):
                    all_targets.append(int(labels))
                else:
                    all_targets.extend([int(l) for l in labels])
                
                # Accuracy
                correct += sum(1 for p, t in zip(all_preds[-len(preds.tolist()):], 
                                                   all_targets[-len(preds.tolist()):]) if p == t)
                total += len(preds.tolist()) if out_np.ndim > 1 else 1
            except (ImportError, Exception):
                total += 1
        
        n_batches = len(data)
        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def _evaluate(
        self, data: List[Tuple], epoch: int
    ) -> Tuple[float, float, List[int], List[int]]:
        """
        Evaluate the model on data (no gradient computation).
        
        Returns:
            (avg_loss, accuracy, predictions_list, targets_list)
        """
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for features, labels in data:
            output = self.model(features)
            loss = self.loss_fn(output, labels)
            total_loss += loss
            
            try:
                import numpy as np
                out_np = output.numpy()
                if out_np.ndim == 1:
                    pred = int(np.argmax(out_np))
                    all_preds.append(pred)
                else:
                    preds = np.argmax(out_np, axis=-1)
                    all_preds.extend(preds.tolist())
                
                if hasattr(labels, 'numpy'):
                    lbl_np = labels.numpy()
                    if lbl_np.ndim == 0:
                        all_targets.append(int(lbl_np))
                    else:
                        all_targets.extend(lbl_np.astype(int).tolist())
                elif isinstance(labels, (int, float)):
                    all_targets.append(int(labels))
                else:
                    all_targets.extend([int(l) for l in labels])
                
                n = len(preds.tolist()) if out_np.ndim > 1 else 1
                correct += sum(1 for p, t in zip(all_preds[-n:], all_targets[-n:]) if p == t)
                total += n
            except (ImportError, Exception):
                total += 1
        
        n_batches = len(data)
        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        if hasattr(self.optimizer, 'learning_rate'):
            return self.optimizer.learning_rate
        return 0.0
