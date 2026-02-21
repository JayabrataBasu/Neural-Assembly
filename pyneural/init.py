"""
Weight initialization strategies for PyNeural.

Core random-fill operations (He/Kaiming, Xavier/Glorot) are
implemented in x86-64 assembly (training_ops.asm). Python
computes fan_in/fan_out and delegates to the native kernels.
"""

from __future__ import annotations

import ctypes
import math
from typing import Optional, Tuple

from .core import _lib


# ---------------------------------------------------------------------------
# Fan computation (pure Python — just shape arithmetic)
# ---------------------------------------------------------------------------

def _compute_fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Compute fan_in and fan_out for a weight tensor.

    2-D  (out, in):            fan_in = in, fan_out = out
    4-D+ (out, in, *kernel):   fan_in = in * prod(kernel),
                                fan_out = out * prod(kernel)
    """
    ndim = len(shape)
    if ndim < 1:
        raise ValueError("Cannot compute fans for scalar tensor")
    elif ndim == 1:
        fan_in = fan_out = shape[0]
    elif ndim == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        receptive_field = 1
        for s in shape[2:]:
            receptive_field *= s
        fan_in = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field
    return fan_in, fan_out


def _calculate_gain(nonlinearity: str) -> float:
    """Recommended gain factor for common activations."""
    gains = {
        "linear": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": math.sqrt(2.0),
        "leaky_relu": math.sqrt(2.0 / (1 + 0.01 ** 2)),
        "selu": 3.0 / 4.0,
    }
    return gains.get(nonlinearity, 1.0)


# ---------------------------------------------------------------------------
# Assembly-backed initializers
# ---------------------------------------------------------------------------

def kaiming_uniform_(tensor, mode: str = "fan_in", nonlinearity: str = "relu",
                      seed: Optional[int] = None) -> None:
    """
    He/Kaiming uniform initialization (in-place).

    Fills with U(-bound, bound) where bound = gain * sqrt(3 / fan).
    Uses assembly ``init_he_uniform`` for the inner loop.
    """
    gain = _calculate_gain(nonlinearity)
    fan_in, fan_out = _compute_fans(tuple(tensor.shape))
    fan = fan_in if mode == "fan_in" else fan_out

    # He-uniform bound without gain is sqrt(6/fan).
    # With gain:  bound = gain * sqrt(3/fan) = sqrt(gain^2 * 3 / fan)
    # For default relu (gain=sqrt2): bound = sqrt(6/fan), which matches init_he_uniform.
    # For non-default gain we fall back to init_uniform_range.
    data_ptr = _lib.neural_tensor_data(tensor._ptr)
    numel = int(_lib.neural_tensor_numel(tensor._ptr))
    seed_val = seed if seed is not None else 0

    if abs(gain - math.sqrt(2.0)) < 1e-9:
        # Default relu gain → assembly He-uniform directly
        _lib.neural_init_he_uniform(data_ptr, numel, fan, seed_val)
    else:
        bound = gain * math.sqrt(3.0 / fan)
        _lib.neural_init_uniform_range(
            data_ptr, numel,
            ctypes.c_float(-bound), ctypes.c_float(bound),
            seed_val,
        )


def kaiming_normal_(tensor, mode: str = "fan_in", nonlinearity: str = "relu",
                     seed: Optional[int] = None) -> None:
    """
    He/Kaiming normal initialization (in-place).

    Fills with N(0, std) where std = gain / sqrt(fan).
    Uses assembly ``init_he_normal`` or ``init_normal_range``.
    """
    gain = _calculate_gain(nonlinearity)
    fan_in, fan_out = _compute_fans(tuple(tensor.shape))
    fan = fan_in if mode == "fan_in" else fan_out

    data_ptr = _lib.neural_tensor_data(tensor._ptr)
    numel = int(_lib.neural_tensor_numel(tensor._ptr))
    seed_val = seed if seed is not None else 0

    if abs(gain - math.sqrt(2.0)) < 1e-9:
        _lib.neural_init_he_normal(data_ptr, numel, fan, seed_val)
    else:
        std = gain / math.sqrt(fan)
        _lib.neural_init_normal_range(
            data_ptr, numel,
            ctypes.c_float(0.0), ctypes.c_float(std),
            seed_val,
        )


def xavier_uniform_(tensor, gain: float = 1.0, seed: Optional[int] = None) -> None:
    """
    Xavier/Glorot uniform initialization (in-place).

    Fills with U(-bound, bound) where bound = gain * sqrt(6 / (fan_in + fan_out)).
    Uses assembly ``init_xavier_uniform``.
    """
    fan_in, fan_out = _compute_fans(tuple(tensor.shape))
    data_ptr = _lib.neural_tensor_data(tensor._ptr)
    numel = int(_lib.neural_tensor_numel(tensor._ptr))
    seed_val = seed if seed is not None else 0

    if abs(gain - 1.0) < 1e-9:
        _lib.neural_init_xavier_uniform(data_ptr, numel, fan_in, fan_out, seed_val)
    else:
        bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
        _lib.neural_init_uniform_range(
            data_ptr, numel,
            ctypes.c_float(-bound), ctypes.c_float(bound),
            seed_val,
        )


def xavier_normal_(tensor, gain: float = 1.0, seed: Optional[int] = None) -> None:
    """
    Xavier/Glorot normal initialization (in-place).

    Fills with N(0, std) where std = gain * sqrt(2 / (fan_in + fan_out)).
    Uses assembly ``init_xavier_normal``.
    """
    fan_in, fan_out = _compute_fans(tuple(tensor.shape))
    data_ptr = _lib.neural_tensor_data(tensor._ptr)
    numel = int(_lib.neural_tensor_numel(tensor._ptr))
    seed_val = seed if seed is not None else 0

    if abs(gain - 1.0) < 1e-9:
        _lib.neural_init_xavier_normal(data_ptr, numel, fan_in, fan_out, seed_val)
    else:
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        _lib.neural_init_normal_range(
            data_ptr, numel,
            ctypes.c_float(0.0), ctypes.c_float(std),
            seed_val,
        )


# ---------------------------------------------------------------------------
# Simple fill helpers (assembly-free — thin wrappers)
# ---------------------------------------------------------------------------

def zeros_(tensor) -> None:
    """Fill tensor with zeros."""
    tensor.fill(0.0)


def ones_(tensor) -> None:
    """Fill tensor with ones."""
    tensor.fill(1.0)


def constant_(tensor, value: float) -> None:
    """Fill tensor with a constant value."""
    tensor.fill(value)


def uniform_(tensor, low: float = 0.0, high: float = 1.0,
             seed: Optional[int] = None) -> None:
    """Fill tensor with uniform random values via assembly ``init_uniform_range``."""
    data_ptr = _lib.neural_tensor_data(tensor._ptr)
    numel = int(_lib.neural_tensor_numel(tensor._ptr))
    seed_val = seed if seed is not None else 0
    _lib.neural_init_uniform_range(
        data_ptr, numel,
        ctypes.c_float(low), ctypes.c_float(high),
        seed_val,
    )


def normal_(tensor, mean: float = 0.0, std: float = 1.0,
            seed: Optional[int] = None) -> None:
    """Fill tensor with normal random values via assembly ``init_normal_range``."""
    data_ptr = _lib.neural_tensor_data(tensor._ptr)
    numel = int(_lib.neural_tensor_numel(tensor._ptr))
    seed_val = seed if seed is not None else 0
    _lib.neural_init_normal_range(
        data_ptr, numel,
        ctypes.c_float(mean), ctypes.c_float(std),
        seed_val,
    )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def init_weights(module, strategy: str = "kaiming_uniform",
                 nonlinearity: str = "relu", seed: Optional[int] = None) -> None:
    """
    Initialize all weights in a module with the given strategy.

    Walks through all Linear layers in a Sequential model and applies
    the chosen initialization to weights. Biases are zeroed.

    Args:
        module: A PyNeural Module (e.g., Sequential)
        strategy: One of 'kaiming_uniform', 'kaiming_normal',
                  'xavier_uniform', 'xavier_normal', 'zeros', 'normal'
        nonlinearity: Activation function name (for He/Kaiming)
        seed: Optional random seed
    """
    from . import nn as _nn

    init_fns = {
        "kaiming_uniform": lambda t: kaiming_uniform_(t, nonlinearity=nonlinearity, seed=seed),
        "kaiming_normal": lambda t: kaiming_normal_(t, nonlinearity=nonlinearity, seed=seed),
        "xavier_uniform": lambda t: xavier_uniform_(t, seed=seed),
        "xavier_normal": lambda t: xavier_normal_(t, seed=seed),
        "zeros": zeros_,
        "normal": lambda t: normal_(t, seed=seed),
    }

    fn = init_fns.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown init strategy: {strategy}. "
            f"Choose from: {list(init_fns.keys())}"
        )

    modules = [module]
    while modules:
        m = modules.pop(0)
        if isinstance(m, _nn.Linear) and hasattr(m, "_ptr") and m._ptr:
            weight_ptr = _lib.neural_linear_weight(m._ptr)
            if weight_ptr:
                from .tensor import Tensor
                weight = Tensor(weight_ptr, owns_data=False)
                fn(weight)

            bias_ptr = _lib.neural_linear_bias(m._ptr)
            if bias_ptr:
                from .tensor import Tensor
                bias = Tensor(bias_ptr, owns_data=False)
                zeros_(bias)

        if hasattr(m, "_modules"):
            modules.extend(m._modules)
