"""
PyNeural - Python Bindings for Neural Assembly Framework

A high-performance deep learning framework implemented in x86-64 assembly
with Python bindings via ctypes.

Example Usage:
    >>> import pyneural as pn
    >>> 
    >>> # Initialize the framework
    >>> pn.init()
    >>> 
    >>> # Create tensors
    >>> a = pn.Tensor.zeros([2, 3])
    >>> b = pn.Tensor.ones([2, 3])
    >>> 
    >>> # Perform operations
    >>> c = a + b
    >>> print(c.numpy())
    >>> 
    >>> # Create neural network layers
    >>> linear = pn.Linear(10, 5)
    >>> output = linear(input_tensor)
    >>> 
    >>> # Cleanup
    >>> pn.shutdown()
"""

from .tensor import Tensor
from .nn import Linear, ReLU, Sigmoid, Softmax, Sequential
from .optim import SGD, Adam
from .autograd import (
    no_grad,
    enable_grad,
    is_grad_enabled,
    set_grad_enabled,
    inference_mode,
    register_hook,
    remove_hook,
    clear_hooks,
)
from .core import (
    init,
    shutdown,
    version,
    get_last_error,
    get_error_message,
    clear_error,
    get_simd_level,
    get_simd_name,
)
from .dataset import (
    Dataset,
    DataLoader,
    TensorDataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)
from .config import Config

__version__ = "1.0.0"
__all__ = [
    # Core
    "init",
    "shutdown", 
    "version",
    "get_last_error",
    "get_error_message",
    "clear_error",
    "get_simd_level",
    "get_simd_name",
    # Tensor
    "Tensor",
    # Neural Network
    "Linear",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Sequential",
    # Optimizers
    "SGD",
    "Adam",
    # Autograd
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "inference_mode",
    "register_hook",
    "remove_hook",
    "clear_hooks",
    # Data
    "Dataset",
    "DataLoader",
    "TensorDataset",
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "BatchSampler",
    "Config",
]
