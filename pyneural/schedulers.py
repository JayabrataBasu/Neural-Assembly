"""
Learning rate schedulers for PyNeural.

Step, exponential, and cosine-annealing schedules delegate their
math to assembly (``lr_step_decay``, ``lr_exponential_decay``,
``lr_cosine_annealing`` in training_ops.asm). WarmupLR and
ReduceLROnPlateau are pure control-flow wrappers.

OneCycleLR and LRFinder are also included here — they're both
pure Python since the heavy lifting is just a bit of trig and
bookkeeping, not number-crunching worth pushing to assembly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Callable

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


# ---------------------------------------------------------------------------
#  OneCycleLR — 1cycle policy (Smith & Topin, 2018)
# ---------------------------------------------------------------------------

class OneCycleLR(_LRScheduler):
    """
    1cycle learning rate policy (Smith & Topin, 2018).

    Ramps LR from a low value up to ``max_lr`` over the first fraction
    of training, then anneals it back down with cosine decay. Useful
    for super-convergence — you typically get better results in fewer
    epochs than with a flat or step schedule.

    This is a *per-step* scheduler: you should call ``.step()`` after
    every training batch, not once per epoch.

    Args:
        max_lr:           Peak learning rate reached at the end of warmup.
        total_steps:      Total number of training steps (batches) you
                          plan to run.  Usually epochs × batches_per_epoch.
        pct_start:        What fraction of total_steps to spend ramping up.
                          Default 0.3 means 30% warmup, 70% cooldown.
        div_factor:       initial_lr = max_lr / div_factor.
        final_div_factor: final_lr = initial_lr / final_div_factor.
                          So the LR ends up very small at the end.
    """

    def __init__(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ):
        # Work out the three anchor LRs before calling super().__init__,
        # which stores initial_lr for the base class bookkeeping.
        initial_lr = max_lr / div_factor
        super().__init__(initial_lr)

        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if not (0.0 <= pct_start <= 1.0):
            raise ValueError(f"pct_start must be in [0, 1], got {pct_start}")

        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.final_lr = initial_lr / final_div_factor

        # How many steps in each phase
        self._warmup_steps = int(total_steps * pct_start)
        self._decay_steps = total_steps - self._warmup_steps

        # Internal step counter — incremented by step()
        self._current_step = 0

    def get_lr(self, step: int = None) -> float:
        """
        Compute LR for the given step index.

        During warmup we do a simple linear ramp. During cooldown we
        use cosine annealing — it's a gentler landing than linear decay
        and tends to generalise better in practice.
        """
        if step is None:
            step = self._current_step

        if step < 0:
            step = 0
        if step >= self.total_steps:
            return self.final_lr

        if step < self._warmup_steps:
            # Linear ramp from initial_lr to max_lr
            if self._warmup_steps == 0:
                return self.max_lr
            t = step / self._warmup_steps
            return self.initial_lr + t * (self.max_lr - self.initial_lr)
        else:
            # Cosine annealing from max_lr down to final_lr
            if self._decay_steps == 0:
                return self.final_lr
            progress = (step - self._warmup_steps) / self._decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + cosine * (self.max_lr - self.final_lr)

    def step(self, epoch: int = None) -> float:
        """
        Advance one step and return the new learning rate.

        Unlike epoch-based schedulers, OneCycleLR tracks steps internally
        so you typically just call step() with no arguments after each batch.
        If you pass an epoch/step number, it sets the counter to that value.
        """
        if epoch is not None:
            self._current_step = epoch
        lr = self.get_lr(self._current_step)
        self._last_lr = lr
        self._current_step += 1
        return lr

    @property
    def current_lr(self) -> float:
        return self._last_lr

    def __repr__(self) -> str:
        return (
            f"OneCycleLR(max_lr={self.max_lr}, total_steps={self.total_steps}, "
            f"pct_start={self.pct_start})"
        )


# ---------------------------------------------------------------------------
#  LRFinder — learning-rate range test (Smith, 2017)
# ---------------------------------------------------------------------------

class LRFinder:
    """
    Learning-rate range test (Smith, 2017).

    Sweeps the learning rate exponentially from ``start_lr`` to ``end_lr``
    over a number of training steps, recording the loss at each step.
    The idea is to find the LR where training is fastest — the "steepest
    descent" point on the loss-vs-LR curve.

    Usage::

        finder = LRFinder(start_lr=1e-7, end_lr=10.0, num_steps=100)

        for step_idx in range(finder.num_steps):
            lr = finder.get_lr(step_idx)
            # ... set optimizer LR, run one batch, get loss ...
            if finder.record(step_idx, loss):
                break  # loss diverged, stop early

        best_lr = finder.suggestion()

    This is deliberately *not* a scheduler subclass — it's a diagnostic
    tool you run once before training, not something that lives inside
    the training loop permanently.

    Args:
        start_lr:     Lower bound of the LR sweep.
        end_lr:       Upper bound of the LR sweep.
        num_steps:    How many steps to run (one batch each).
        smooth_factor: Exponential smoothing for the loss curve. Helps
                       filter out per-batch noise. 0 = no smoothing.
        diverge_threshold: Stop early if the current (smoothed) loss
                           exceeds diverge_threshold × best_loss.
    """

    def __init__(
        self,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_steps: int = 100,
        smooth_factor: float = 0.05,
        diverge_threshold: float = 4.0,
    ):
        if start_lr <= 0 or end_lr <= 0:
            raise ValueError("start_lr and end_lr must be positive")
        if start_lr >= end_lr:
            raise ValueError(f"start_lr ({start_lr}) must be < end_lr ({end_lr})")
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")

        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_steps = num_steps
        self.smooth_factor = smooth_factor
        self.diverge_threshold = diverge_threshold

        # The multiplier we apply each step to go from start_lr to end_lr
        # in exactly num_steps steps on a log scale.
        self._mult = (end_lr / start_lr) ** (1.0 / (num_steps - 1))

        # Recorded data — populated as you call record()
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self.smoothed_losses: List[float] = []

        self._best_loss = float("inf")
        self._avg_loss = 0.0

    def get_lr(self, step: int) -> float:
        """Return the LR for step *step* (0-indexed)."""
        return self.start_lr * (self._mult ** step)

    def record(self, step: int, loss: float) -> bool:
        """
        Record one (lr, loss) pair and return True if we should stop.

        Applies exponential smoothing to the raw loss so the suggestion()
        method has a cleaner curve to work with.
        """
        lr = self.get_lr(step)
        self.lrs.append(lr)
        self.losses.append(loss)

        # Exponential moving average — start from 0 so the bias
        # correction actually makes sense (same trick Adam uses).
        self._avg_loss = (
            self.smooth_factor * loss
            + (1.0 - self.smooth_factor) * self._avg_loss
        )

        # Bias correction so early values aren't squashed toward 0
        corrected = self._avg_loss / (1.0 - (1.0 - self.smooth_factor) ** (step + 1))
        self.smoothed_losses.append(corrected)

        if corrected < self._best_loss:
            self._best_loss = corrected

        # Did the loss blow up?
        if step > 0 and corrected > self.diverge_threshold * self._best_loss:
            return True  # stop

        return False

    def suggestion(self) -> float:
        """
        Suggest the best learning rate from the recorded sweep.

        Picks the LR with the steepest negative gradient on the
        smoothed loss curve — that's where training is improving fastest.
        Falls back to the LR at minimum loss if the gradient approach
        doesn't find anything sensible.
        """
        if len(self.smoothed_losses) < 3:
            raise RuntimeError(
                "Need at least 3 recorded steps to make a suggestion. "
                "Did you forget to call record()?"
            )

        # Compute the gradient (finite difference) of the smoothed loss
        # in log-LR space.
        grads = []
        for i in range(1, len(self.smoothed_losses)):
            # We use the raw index difference since LRs are equally spaced
            # on a log scale, so the denominator is constant and doesn't
            # change which point has the steepest drop.
            grads.append(self.smoothed_losses[i] - self.smoothed_losses[i - 1])

        # Find the index of the steepest negative gradient
        min_grad_idx = 0
        min_grad = grads[0]
        for i, g in enumerate(grads):
            if g < min_grad:
                min_grad = g
                min_grad_idx = i

        # The steepest drop at grads[i] corresponds to lrs[i+1]
        # (since grads[i] = loss[i+1] - loss[i])
        # But we usually want something slightly to the left of the
        # minimum-gradient point, so use lrs[min_grad_idx] instead.
        if min_grad >= 0:
            # Loss never decreased — just return LR at the lowest loss
            best_idx = 0
            best_val = self.smoothed_losses[0]
            for i, v in enumerate(self.smoothed_losses):
                if v < best_val:
                    best_val = v
                    best_idx = i
            return self.lrs[best_idx]

        return self.lrs[min_grad_idx]

    @property
    def results(self) -> Dict[str, List[float]]:
        """The raw data from the sweep, handy for plotting."""
        return {
            "lrs": list(self.lrs),
            "losses": list(self.losses),
            "smoothed_losses": list(self.smoothed_losses),
        }

    def __repr__(self) -> str:
        return (
            f"LRFinder(start_lr={self.start_lr}, end_lr={self.end_lr}, "
            f"num_steps={self.num_steps}, recorded={len(self.lrs)})"
        )
