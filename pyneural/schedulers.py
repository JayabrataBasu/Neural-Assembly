"""
Learning rate schedulers for PyNeural.

Step, exponential, and cosine-annealing schedules delegate their
math to assembly (``lr_step_decay``, ``lr_exponential_decay``,
``lr_cosine_annealing`` in training_ops.asm). WarmupLR and
ReduceLROnPlateau are pure control-flow wrappers.
"""

from __future__ import annotations

from typing import Optional

from .core import _lib


class _LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.last_epoch = -1
        self._last_lr = initial_lr

    def get_lr(self, epoch: int) -> float:
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> float:
        """Advance the scheduler and return the new LR."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        lr = self.get_lr(self.last_epoch)
        self._last_lr = lr
        return lr

    @property
    def current_lr(self) -> float:
        return self._last_lr


class StepLR(_LRScheduler):
    """
    Decay LR by ``gamma`` every ``step_size`` epochs.

    lr = initial_lr * gamma^(epoch // step_size)

    Math is computed in assembly via ``lr_step_decay``.
    """

    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        return _lib.neural_lr_step_decay(
            self.initial_lr, epoch, self.step_size, self.gamma
        )


class ExponentialLR(_LRScheduler):
    """
    Decay LR by ``gamma`` every epoch.

    lr = initial_lr * gamma^epoch

    Math is computed in assembly via ``lr_exponential_decay``.
    """

    def __init__(self, initial_lr: float, gamma: float = 0.99):
        super().__init__(initial_lr)
        self.gamma = gamma

    def get_lr(self, epoch: int) -> float:
        return _lib.neural_lr_exponential_decay(
            self.initial_lr, epoch, self.gamma
        )


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing LR schedule.

    lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max))

    Math is computed in assembly via ``lr_cosine_annealing``.
    """

    def __init__(self, initial_lr: float, T_max: int, eta_min: float = 0.0):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, epoch: int) -> float:
        return _lib.neural_lr_cosine_annealing(
            self.initial_lr, epoch, self.T_max, self.eta_min
        )


class WarmupLR(_LRScheduler):
    """
    Linear warmup followed by another scheduler.

    During warmup LR linearly increases from ``warmup_start_lr``
    to ``initial_lr``. After warmup the ``after_scheduler`` takes over.
    """

    def __init__(
        self,
        initial_lr: float,
        warmup_epochs: int,
        warmup_start_lr: float = 0.0,
        after_scheduler: Optional[_LRScheduler] = None,
    ):
        super().__init__(initial_lr)
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.after_scheduler = after_scheduler

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            if self.warmup_epochs <= 1:
                return self.initial_lr
            alpha = epoch / (self.warmup_epochs - 1)
            return self.warmup_start_lr + alpha * (self.initial_lr - self.warmup_start_lr)
        else:
            if self.after_scheduler is not None:
                return self.after_scheduler.get_lr(epoch - self.warmup_epochs)
            return self.initial_lr


class ReduceLROnPlateau:
    """
    Reduce LR when a metric has stopped improving.

    Pure control-flow; no heavy math needed.
    """

    def __init__(
        self,
        initial_lr: float,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 0.0,
        cooldown: int = 0,
    ):
        self.initial_lr = initial_lr
        self._last_lr = initial_lr
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.cooldown = cooldown

        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.threshold
        else:
            return current > self.best + self.threshold

    def step(self, metric: float) -> float:
        """
        Update scheduler with the latest metric value.

        Returns:
            Current learning rate
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self._last_lr

        if self._is_better(metric):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            new_lr = max(self._last_lr * self.factor, self.min_lr)
            if new_lr < self._last_lr:
                self._last_lr = new_lr
                self.num_bad_epochs = 0
                self.cooldown_counter = self.cooldown

        return self._last_lr

    @property
    def current_lr(self) -> float:
        return self._last_lr
