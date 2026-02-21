"""
Learning rate schedulers for PyNeural.

Provides various LR scheduling strategies that can be composed
with any optimizer. Each scheduler adjusts the learning rate
based on epoch count or training metrics.
"""

from __future__ import annotations

import math
from typing import Optional


class _LRScheduler:
    """
    Base class for learning rate schedulers.
    
    Subclasses must implement `get_lr(epoch)` which returns the
    new learning rate for the given epoch number.
    """
    
    def __init__(self, initial_lr: float):
        self.initial_lr = initial_lr
        self.last_epoch = -1
        self._last_lr = initial_lr
    
    def get_lr(self, epoch: int) -> float:
        """Compute LR for the given epoch. Override in subclasses."""
        raise NotImplementedError
    
    def step(self, epoch: Optional[int] = None) -> float:
        """
        Advance the scheduler and return the new learning rate.
        
        Args:
            epoch: Epoch number (auto-increments if None)
        
        Returns:
            New learning rate
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        lr = self.get_lr(self.last_epoch)
        self._last_lr = lr
        return lr
    
    @property
    def current_lr(self) -> float:
        """Get the current learning rate."""
        return self._last_lr


class StepLR(_LRScheduler):
    """
    Decay LR by a factor every `step_size` epochs.
    
    lr = initial_lr * gamma^(epoch // step_size)
    
    Args:
        initial_lr: Base learning rate
        step_size: Period of LR decay (in epochs)
        gamma: Multiplicative factor of LR decay (default: 0.1)
    
    Example:
        >>> scheduler = StepLR(initial_lr=0.01, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     lr = scheduler.step()
        ...     # lr = 0.01 for epochs 0-29, 0.001 for 30-59, etc.
    """
    
    def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLR(_LRScheduler):
    """
    Decay LR by gamma every epoch.
    
    lr = initial_lr * gamma^epoch
    
    Args:
        initial_lr: Base learning rate
        gamma: Multiplicative factor per epoch (e.g. 0.99)
    """
    
    def __init__(self, initial_lr: float, gamma: float = 0.99):
        super().__init__(initial_lr)
        self.gamma = gamma
    
    def get_lr(self, epoch: int) -> float:
        return self.initial_lr * (self.gamma ** epoch)


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing learning rate schedule.
    
    lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max))
    
    Args:
        initial_lr: Base learning rate
        T_max: Maximum number of epochs (full cosine period)
        eta_min: Minimum learning rate (default: 0)
    
    Example:
        >>> scheduler = CosineAnnealingLR(initial_lr=0.01, T_max=100)
        >>> # LR smoothly decreases from 0.01 to 0 over 100 epochs
    """
    
    def __init__(self, initial_lr: float, T_max: int, eta_min: float = 0.0):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self, epoch: int) -> float:
        if self.T_max == 0:
            return self.eta_min
        return self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * epoch / self.T_max)
        )


class WarmupLR(_LRScheduler):
    """
    Linear warmup followed by another scheduler.
    
    During warmup (epochs 0 to warmup_epochs-1), LR linearly increases
    from warmup_start_lr to the initial_lr. After warmup, the 
    after_scheduler takes over.
    
    Args:
        initial_lr: Target learning rate after warmup
        warmup_epochs: Number of warmup epochs
        warmup_start_lr: Starting LR for warmup (default: 0)
        after_scheduler: Scheduler to use after warmup (optional)
    
    Example:
        >>> cosine = CosineAnnealingLR(initial_lr=0.01, T_max=90)
        >>> scheduler = WarmupLR(initial_lr=0.01, warmup_epochs=10,
        ...                       after_scheduler=cosine)
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
            # Linear warmup
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
    
    Args:
        initial_lr: Base learning rate
        mode: 'min' or 'max' — whether to minimize or maximize the metric
        factor: Factor by which LR is reduced (default: 0.1)
        patience: Number of epochs with no improvement before reducing LR
        threshold: Threshold for measuring improvement (default: 1e-4)
        min_lr: Minimum learning rate (default: 0)
        cooldown: Epochs to wait after a reduction before resuming monitoring
    
    Example:
        >>> scheduler = ReduceLROnPlateau(initial_lr=0.01, patience=10)
        >>> for epoch in range(100):
        ...     val_loss = train_one_epoch()
        ...     lr = scheduler.step(val_loss)
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
        
        Args:
            metric: The monitored metric (e.g., val_loss)
        
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
