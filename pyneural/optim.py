"""
Optimizers for PyNeural.
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
            _lib.neural_optimizer_free(self._ptr)
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


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Args:
        learning_rate: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0.0)
    
    Example:
        >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
        >>> optimizer.step(params, grads)
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self._ptr = _lib.neural_sgd_create(
            ctypes.c_double(learning_rate),
            ctypes.c_double(momentum)
        )
        if self._ptr is None or self._ptr == 0:
            raise NeuralException(_lib.neural_get_last_error(), "Failed to create SGD optimizer")
    
    def step(self, params: List[Tensor], grads: List[Tensor]) -> None:
        """Perform one optimization step."""
        if len(params) != len(grads):
            raise ValueError("params and grads must have same length")
        
        # Create arrays of pointers
        n = len(params)
        param_ptrs = (ctypes.c_void_p * n)(*[p._ptr for p in params])
        grad_ptrs = (ctypes.c_void_p * n)(*[g._ptr for g in grads])
        
        result = _lib.neural_optimizer_step(
            self._ptr,
            param_ptrs,
            grad_ptrs,
            n
        )
        _check_error(result, "sgd step")
    
    def __repr__(self) -> str:
        return f"SGD(lr={self.learning_rate}, momentum={self.momentum})"


class Adam(Optimizer):
    """
    Adam optimizer.
    
    Args:
        learning_rate: Learning rate (default: 0.001)
        beta1: First moment decay rate (default: 0.9)
        beta2: Second moment decay rate (default: 0.999)
        epsilon: Numerical stability constant (default: 1e-8)
    
    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> optimizer.step(params, grads)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self._ptr = _lib.neural_adam_create(
            ctypes.c_double(learning_rate),
            ctypes.c_double(beta1),
            ctypes.c_double(beta2),
            ctypes.c_double(epsilon)
        )
        if self._ptr is None or self._ptr == 0:
            raise NeuralException(_lib.neural_get_last_error(), "Failed to create Adam optimizer")
    
    def step(self, params: List[Tensor], grads: List[Tensor]) -> None:
        """Perform one optimization step."""
        if len(params) != len(grads):
            raise ValueError("params and grads must have same length")
        
        n = len(params)
        param_ptrs = (ctypes.c_void_p * n)(*[p._ptr for p in params])
        grad_ptrs = (ctypes.c_void_p * n)(*[g._ptr for g in grads])
        
        result = _lib.neural_optimizer_step(
            self._ptr,
            param_ptrs,
            grad_ptrs,
            n
        )
        _check_error(result, "adam step")
    
    def __repr__(self) -> str:
        return f"Adam(lr={self.learning_rate}, betas=({self.beta1}, {self.beta2}), eps={self.epsilon})"
