"""
Weight initialization strategies for PyNeural.

Provides He/Kaiming, Xavier/Glorot, and other standard
initialization methods for neural network parameters.
Works with PyNeural Tensor objects via numpy interop.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple


def _compute_fans(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Compute fan_in and fan_out for a weight tensor.
    
    For 2D tensors (Linear layers): shape = (out_features, in_features)
        fan_in = in_features, fan_out = out_features
    For 4D tensors (Conv2D layers): shape = (out_channels, in_channels, kH, kW)
        fan_in = in_channels * kH * kW
        fan_out = out_channels * kH * kW
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
        # Conv layers: (out_channels, in_channels, *kernel_size)
        receptive_field = 1
        for s in shape[2:]:
            receptive_field *= s
        fan_in = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field
    
    return fan_in, fan_out


def _fill_uniform(data: list, n: int, low: float, high: float, seed: Optional[int] = None):
    """Fill a flat list with uniform random values."""
    if seed is not None:
        random.seed(seed)
    for i in range(n):
        data[i] = random.uniform(low, high)


def _fill_normal(data: list, n: int, mean: float, std: float, seed: Optional[int] = None):
    """Fill a flat list with normal random values (Box-Muller)."""
    if seed is not None:
        random.seed(seed)
    for i in range(n):
        data[i] = random.gauss(mean, std)


def kaiming_uniform_(tensor, mode: str = "fan_in", nonlinearity: str = "relu",
                      seed: Optional[int] = None) -> None:
    """
    He/Kaiming uniform initialization (in-place).
    
    Fills the tensor with values from U(-bound, bound) where:
        bound = sqrt(6 / fan) * gain
    
    Recommended for ReLU and LeakyReLU activations.
    
    Args:
        tensor: PyNeural Tensor to initialize
        mode: 'fan_in' (default) or 'fan_out'
        nonlinearity: 'relu' (gain=sqrt(2)), 'leaky_relu', 'linear', 'sigmoid', 'tanh'
        seed: Optional random seed for reproducibility
    """
    gain = _calculate_gain(nonlinearity)
    fan_in, fan_out = _compute_fans(tuple(tensor.shape))
    fan = fan_in if mode == "fan_in" else fan_out
    
    bound = gain * math.sqrt(3.0 / fan)
    
    try:
        import numpy as np
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        arr = tensor.numpy()
        arr[:] = rng.uniform(-bound, bound, arr.shape).astype(np.float32)
    except ImportError:
        # Fallback: pure Python
        numel = tensor.numel
        data = [0.0] * numel
        _fill_uniform(data, numel, -bound, bound, seed)
        _set_tensor_data(tensor, data)


def kaiming_normal_(tensor, mode: str = "fan_in", nonlinearity: str = "relu",
                     seed: Optional[int] = None) -> None:
    """
    He/Kaiming normal initialization (in-place).
    
    Fills the tensor with values from N(0, std) where:
        std = gain / sqrt(fan)
    
    Args:
        tensor: PyNeural Tensor to initialize
        mode: 'fan_in' (default) or 'fan_out'
        nonlinearity: Activation function name
        seed: Optional random seed
    """
    gain = _calculate_gain(nonlinearity)
    fan_in, fan_out = _compute_fans(tuple(tensor.shape))
    fan = fan_in if mode == "fan_in" else fan_out
    
    std = gain / math.sqrt(fan)
    
    try:
        import numpy as np
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        arr = tensor.numpy()
        arr[:] = rng.normal(0, std, arr.shape).astype(np.float32)
    except ImportError:
        numel = tensor.numel
        data = [0.0] * numel
        _fill_normal(data, numel, 0.0, std, seed)
        _set_tensor_data(tensor, data)


def xavier_uniform_(tensor, gain: float = 1.0, seed: Optional[int] = None) -> None:
    """
    Xavier/Glorot uniform initialization (in-place).
    
    Fills the tensor with values from U(-bound, bound) where:
        bound = gain * sqrt(6 / (fan_in + fan_out))
    
    Recommended for sigmoid and tanh activations.
    
    Args:
        tensor: PyNeural Tensor to initialize
        gain: Scaling factor (default: 1.0)
        seed: Optional random seed
    """
    fan_in, fan_out = _compute_fans(tuple(tensor.shape))
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    
    try:
        import numpy as np
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        arr = tensor.numpy()
        arr[:] = rng.uniform(-bound, bound, arr.shape).astype(np.float32)
    except ImportError:
        numel = tensor.numel
        data = [0.0] * numel
        _fill_uniform(data, numel, -bound, bound, seed)
        _set_tensor_data(tensor, data)


def xavier_normal_(tensor, gain: float = 1.0, seed: Optional[int] = None) -> None:
    """
    Xavier/Glorot normal initialization (in-place).
    
    Fills the tensor with values from N(0, std) where:
        std = gain * sqrt(2 / (fan_in + fan_out))
    
    Args:
        tensor: PyNeural Tensor to initialize
        gain: Scaling factor (default: 1.0)
        seed: Optional random seed
    """
    fan_in, fan_out = _compute_fans(tuple(tensor.shape))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    try:
        import numpy as np
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        arr = tensor.numpy()
        arr[:] = rng.normal(0, std, arr.shape).astype(np.float32)
    except ImportError:
        numel = tensor.numel
        data = [0.0] * numel
        _fill_normal(data, numel, 0.0, std, seed)
        _set_tensor_data(tensor, data)


def zeros_(tensor) -> None:
    """Fill tensor with zeros (in-place)."""
    tensor.fill(0.0)


def ones_(tensor) -> None:
    """Fill tensor with ones (in-place)."""
    tensor.fill(1.0)


def constant_(tensor, value: float) -> None:
    """Fill tensor with a constant value (in-place)."""
    tensor.fill(value)


def uniform_(tensor, low: float = 0.0, high: float = 1.0,
             seed: Optional[int] = None) -> None:
    """Fill tensor with uniform random values (in-place)."""
    try:
        import numpy as np
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        arr = tensor.numpy()
        arr[:] = rng.uniform(low, high, arr.shape).astype(np.float32)
    except ImportError:
        numel = tensor.numel
        data = [0.0] * numel
        _fill_uniform(data, numel, low, high, seed)
        _set_tensor_data(tensor, data)


def normal_(tensor, mean: float = 0.0, std: float = 1.0,
            seed: Optional[int] = None) -> None:
    """Fill tensor with normal random values (in-place)."""
    try:
        import numpy as np
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        arr = tensor.numpy()
        arr[:] = rng.normal(mean, std, arr.shape).astype(np.float32)
    except ImportError:
        numel = tensor.numel
        data = [0.0] * numel
        _fill_normal(data, numel, mean, std, seed)
        _set_tensor_data(tensor, data)


def _calculate_gain(nonlinearity: str) -> float:
    """Calculate the recommended gain for an activation function."""
    gains = {
        "linear": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": math.sqrt(2.0),
        "leaky_relu": math.sqrt(2.0 / (1 + 0.01**2)),
        "selu": 3.0 / 4.0,
    }
    return gains.get(nonlinearity, 1.0)


def _set_tensor_data(tensor, flat_data: list) -> None:
    """Set tensor data from a flat Python list (fallback when numpy unavailable)."""
    import ctypes
    data_ptr = tensor.data_ptr
    arr = (ctypes.c_float * len(flat_data))(*flat_data)
    ctypes.memmove(data_ptr, arr, len(flat_data) * 4)


def init_weights(module, strategy: str = "kaiming_uniform",
                 nonlinearity: str = "relu", seed: Optional[int] = None) -> None:
    """
    Initialize all weights in a module with the given strategy.
    
    This is a convenience function that applies the chosen initialization
    to all Linear layers in a Sequential model.
    
    Args:
        module: A PyNeural Module (e.g., Sequential)
        strategy: One of 'kaiming_uniform', 'kaiming_normal', 
                  'xavier_uniform', 'xavier_normal', 'zeros', 'normal'
        nonlinearity: Activation function name (for He/Kaiming)
        seed: Optional random seed
    """
    from . import nn as _nn
    from . import core as _core
    
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
        raise ValueError(f"Unknown init strategy: {strategy}. Choose from: {list(init_fns.keys())}")
    
    # Walk through all Linear layers
    modules = [module]
    while modules:
        m = modules.pop(0)
        if isinstance(m, _nn.Linear) and hasattr(m, '_ptr') and m._ptr:
            # Get weight tensor via native API
            weight_ptr = _core._lib.neural_linear_weight(m._ptr)
            if weight_ptr:
                from .tensor import Tensor
                weight = Tensor(weight_ptr, owns_data=False)
                fn(weight)
            
            # Bias stays zeros (standard practice)
            bias_ptr = _core._lib.neural_linear_bias(m._ptr)
            if bias_ptr:
                from .tensor import Tensor
                bias = Tensor(bias_ptr, owns_data=False)
                zeros_(bias)
        
        if hasattr(m, '_modules'):
            modules.extend(m._modules)
