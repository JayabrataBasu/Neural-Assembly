"""
Training utilities for PyNeural.

Provides:
  - NaN/Inf detection (assembly-backed ``tensor_has_nan`` / ``tensor_has_inf``)
  - Gradient L2 norm logging (assembly-backed ``tensor_grad_l2_norm``)
  - Early stopping (pure control-flow)
  - TrainerConfig / TrainingHistory data classes
  - Trainer: high-level training loop orchestrator
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .core import _lib


# ---------------------------------------------------------------------------
# NaN / Inf Detection  (delegates to assembly)
# ---------------------------------------------------------------------------

class NaNDetector:
    """
    Detect NaN or Inf values in tensor data.

    Uses assembly routines ``tensor_has_nan`` and ``tensor_has_inf``
    from training_ops.asm.

    Example:
        >>> detector = NaNDetector()
        >>> detector.check(tensor)          # raises on NaN/Inf
        >>> detector.has_nan(tensor)         # returns bool
    """

    def __init__(self, raise_on_detect: bool = True):
        self.raise_on_detect = raise_on_detect
        self.nan_count = 0
        self.inf_count = 0

    def has_nan(self, tensor) -> bool:
        """Check whether *tensor* contains NaN values (assembly)."""
        data_ptr = _lib.neural_tensor_data(tensor._ptr)
        numel = int(_lib.neural_tensor_numel(tensor._ptr))
        return bool(_lib.neural_tensor_has_nan(data_ptr, numel))

    def has_inf(self, tensor) -> bool:
        """Check whether *tensor* contains Inf values (assembly)."""
        data_ptr = _lib.neural_tensor_data(tensor._ptr)
        numel = int(_lib.neural_tensor_numel(tensor._ptr))
        return bool(_lib.neural_tensor_has_inf(data_ptr, numel))

    def check(self, tensor, name: str = "tensor") -> None:
        """Check for NaN/Inf and optionally raise."""
        if self.has_nan(tensor):
            self.nan_count += 1
            if self.raise_on_detect:
                raise RuntimeError(f"NaN detected in {name}")
        if self.has_inf(tensor):
            self.inf_count += 1
            if self.raise_on_detect:
                raise RuntimeError(f"Inf detected in {name}")

    def reset_counts(self) -> None:
        self.nan_count = 0
        self.inf_count = 0


# ---------------------------------------------------------------------------
# Gradient norm (assembly-backed)
# ---------------------------------------------------------------------------

def grad_l2_norm(tensor) -> float:
    """Compute the L2 norm of a tensor's data (assembly ``tensor_grad_l2_norm``)."""
    data_ptr = _lib.neural_tensor_data(tensor._ptr)
    numel = int(_lib.neural_tensor_numel(tensor._ptr))
    return float(_lib.neural_tensor_grad_l2_norm(data_ptr, numel))


# ---------------------------------------------------------------------------
# Early Stopping (pure control-flow)
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        patience: How many epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' (lower is better) or 'max' (higher is better).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        else:
            return current > self.best_score + self.min_delta

    def step(self, metric: float, epoch: int = 0) -> bool:
        """
        Report a new metric value.

        Returns:
            True if training should stop.
        """
        if self._is_improvement(metric):
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Training history & config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""
    epochs: int = 100
    log_interval: int = 1
    enable_nan_detection: bool = True
    enable_grad_norm_logging: bool = True
    grad_clip_norm: Optional[float] = None
    grad_clip_value: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    early_stopping_mode: str = "min"


@dataclass
class TrainingHistory:
    """Record of training metrics across epochs."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    extra: Dict[str, List[float]] = field(default_factory=dict)

    def last(self, key: str) -> Optional[float]:
        lst = getattr(self, key, self.extra.get(key))
        if lst and len(lst) > 0:
            return lst[-1]
        return None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    High-level training loop orchestrator.

    Composes an optimizer, optional LR scheduler, early stopping,
    NaN detection, and gradient-norm logging into a single ``fit``
    call.

    Example:
        >>> cfg = TrainerConfig(epochs=50, early_stopping_patience=10)
        >>> trainer = Trainer(model, optimizer, loss_fn, config=cfg)
        >>> history = trainer.fit(train_fn, val_fn)
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        config: Optional[TrainerConfig] = None,
        scheduler=None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config or TrainerConfig()
        self.scheduler = scheduler
        self.callbacks = callbacks or []

        self.history = TrainingHistory()
        self.nan_detector = NaNDetector(raise_on_detect=False)

        self._early_stopping: Optional[EarlyStopping] = None
        if self.config.early_stopping_patience is not None:
            self._early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode=self.config.early_stopping_mode,
            )

    def fit(
        self,
        train_fn: Callable[[], float],
        val_fn: Optional[Callable[[], float]] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Run the training loop.

        Args:
            train_fn: Callable that runs one epoch of training and returns loss.
            val_fn: Optional callable that runs validation and returns loss.
            verbose: Whether to print progress.

        Returns:
            TrainingHistory with recorded metrics.
        """
        cfg = self.config

        for epoch in range(cfg.epochs):
            t0 = time.time()

            # --- Train one epoch ---
            train_loss = train_fn()
            self.history.train_loss.append(train_loss)

            # --- Gradient clipping (applied to optimizer's tracked params) ---
            if cfg.grad_clip_norm is not None and self.optimizer is not None:
                opt_ptr = getattr(self.optimizer, "_ptr", None)
                if opt_ptr:
                    _lib.neural_clip_grad_norm(opt_ptr, float(cfg.grad_clip_norm))

            # --- Scheduler step ---
            if self.scheduler is not None:
                lr = self.scheduler.step()
                self.history.learning_rates.append(lr)
                # Apply new LR to optimizer if it has native ptr
                if hasattr(self.optimizer, "_ptr") and self.optimizer._ptr:
                    _lib.neural_optimizer_set_lr(self.optimizer._ptr, lr)

            # --- Validation ---
            val_loss = None
            if val_fn is not None:
                val_loss = val_fn()
                self.history.val_loss.append(val_loss)

            elapsed = time.time() - t0
            self.history.epoch_times.append(elapsed)

            # --- Logging ---
            if verbose and (epoch % cfg.log_interval == 0 or epoch == cfg.epochs - 1):
                msg = f"Epoch {epoch:4d} | train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.6f}"
                if self.history.learning_rates:
                    msg += f" | lr={self.history.learning_rates[-1]:.2e}"
                if self.history.grad_norms:
                    msg += f" | grad_norm={self.history.grad_norms[-1]:.4f}"
                msg += f" | {elapsed:.2f}s"
                print(msg)

            # --- Early stopping ---
            if self._early_stopping is not None:
                metric = val_loss if val_loss is not None else train_loss
                if self._early_stopping.step(metric, epoch):
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch} "
                            f"(best={self._early_stopping.best_score:.6f} "
                            f"at epoch {self._early_stopping.best_epoch})"
                        )
                    break

            # --- Callbacks ---
            for cb in self.callbacks:
                cb(epoch, self.history)

        return self.history
