"""
Optimizers for PyNeural.

The heavy lifting is done by C implementations in optimizers_c.c which
keep per-parameter moment buffers and handle both float32 and float64.
The Python side is just a thin wrapper for construction and step().
"""

import ctypes
from typing import List

from .core import _lib, NeuralException, _check_error
from .tensor import Tensor


class Optimizer:
    """Base class for optimizers."""

    def __init__(self):
        self._ptr = None

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.opt_free(self._ptr)
            self._ptr = None

    def step(self, params: List[Tensor], grads: List[Tensor]) -> None:
        """
        Perform one optimization step.

        Args:
            params: List of parameter tensors
            grads: List of gradient tensors (same order as params)
        """
        raise NotImplementedError

    def zero_grad(self, grads: List[Tensor]) -> None:
        """
        Zero out all gradients.

        Args:
            grads: List of gradient tensors
        """
        for grad in grads:
            grad.fill(0.0)

    @property
    def lr(self) -> float:
        """Current learning rate (read from C struct)."""
        return _lib.opt_get_lr(self._ptr)

    @lr.setter
    def lr(self, value: float):
        _lib.opt_set_lr(self._ptr, ctypes.c_double(value))

    def _do_step(self, params: List[Tensor], grads: List[Tensor], label: str):
        """Shared step logic — validates args and calls opt_step."""
        if len(params) != len(grads):
            raise ValueError("params and grads must have same length")

        n = len(params)
        param_ptrs = (ctypes.c_void_p * n)(*[p._ptr for p in params])
        grad_ptrs = (ctypes.c_void_p * n)(*[g._ptr for g in grads])
        result = _lib.opt_step(self._ptr, param_ptrs, grad_ptrs, n)
        _check_error(result, label)


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Args:
        learning_rate: Learning rate (default: 0.01).
        momentum: Momentum factor (default: 0.0).
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

        self._ptr = _lib.opt_sgd_create(
            ctypes.c_double(learning_rate),
            ctypes.c_double(momentum),
        )
        if self._ptr is None or self._ptr == 0:
            raise NeuralException(0, "Failed to create SGD optimizer")

    def step(self, params: List[Tensor], grads: List[Tensor]) -> None:
        self._do_step(params, grads, "sgd step")

    def __repr__(self) -> str:
        return f"SGD(lr={self.learning_rate}, momentum={self.momentum})"


class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2015).

    Args:
        learning_rate: Learning rate (default: 0.001).
        beta1: First moment decay rate (default: 0.9).
        beta2: Second moment decay rate (default: 0.999).
        epsilon: Numerical stability constant (default: 1e-8).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self._ptr = _lib.opt_adam_create(
            ctypes.c_double(learning_rate),
            ctypes.c_double(beta1),
            ctypes.c_double(beta2),
            ctypes.c_double(epsilon),
        )
        if self._ptr is None or self._ptr == 0:
            raise NeuralException(0, "Failed to create Adam optimizer")

    def step(self, params: List[Tensor], grads: List[Tensor]) -> None:
        self._do_step(params, grads, "adam step")

    def __repr__(self) -> str:
        return f"Adam(lr={self.learning_rate}, betas=({self.beta1}, {self.beta2}), eps={self.epsilon})"


class AdamW(Optimizer):
    """
    AdamW optimizer — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Weight decay is applied directly to the parameters instead of through
    the gradient, which regularises more effectively at higher learning rates.

    Args:
        learning_rate: Learning rate (default: 0.001).
        beta1: First moment decay rate (default: 0.9).
        beta2: Second moment decay rate (default: 0.999).
        epsilon: Numerical stability constant (default: 1e-8).
        weight_decay: Decoupled weight decay coefficient (default: 0.01).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self._ptr = _lib.opt_adamw_create(
            ctypes.c_double(learning_rate),
            ctypes.c_double(beta1),
            ctypes.c_double(beta2),
            ctypes.c_double(epsilon),
            ctypes.c_double(weight_decay),
        )
        if self._ptr is None or self._ptr == 0:
            raise NeuralException(0, "Failed to create AdamW optimizer")

    def step(self, params: List[Tensor], grads: List[Tensor]) -> None:
        self._do_step(params, grads, "adamw step")

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.learning_rate}, "
            f"betas=({self.beta1}, {self.beta2}), "
            f"eps={self.epsilon}, wd={self.weight_decay})"
        )
