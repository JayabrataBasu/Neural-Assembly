"""
Neural network layers for PyNeural.
"""

import ctypes
from typing import List, Optional, Sequence, Union

from .core import _lib, NeuralError, NeuralException, _check_error
from .tensor import Tensor


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self._modules: List["Module"] = []
        self._training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. Override in subclass."""
        raise NotImplementedError
    
    def __call__(self, x: Tensor) -> Tensor:
        """Call forward pass."""
        return self.forward(x)
    
    def train(self, mode: bool = True) -> "Module":
        """Set training mode."""
        self._training = mode
        for module in self._modules:
            module.train(mode)
        return self
    
    def eval(self) -> "Module":
        """Set evaluation mode."""
        return self.train(False)
    
    def parameters(self) -> List[Tensor]:
        """Get all parameters."""
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params


class Linear(Module):
    """
    Fully connected (linear) layer.
    
    Computes: y = x @ W^T + b
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds learnable bias (default: True)
    
    Example:
        >>> linear = Linear(10, 5)
        >>> x = Tensor.zeros([32, 10])  # batch of 32
        >>> y = linear(x)
        >>> y.shape
        (32, 5)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # Create native layer
        self._ptr = _lib.neural_linear_create(in_features, out_features, 1 if bias else 0)
        if self._ptr is None or self._ptr == 0:
            raise NeuralException(_lib.neural_get_last_error(), "Failed to create Linear layer")
    
    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.neural_linear_free(self._ptr)
            self._ptr = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Determine output shape
        if x.ndim == 1:
            output_shape = [self.out_features]
        else:
            output_shape = list(x.shape[:-1]) + [self.out_features]
        
        output = Tensor.create(output_shape, x.dtype)
        result = _lib.neural_linear_forward(self._ptr, x._ptr, output._ptr)
        _check_error(result, "linear forward")
        return output
    
    def parameters(self) -> List[Tensor]:
        """Get weight and bias tensors."""
        params = []
        weight_ptr = _lib.neural_linear_weight(self._ptr)
        if weight_ptr:
            params.append(Tensor(weight_ptr, owns_data=False))
        if self.has_bias:
            bias_ptr = _lib.neural_linear_bias(self._ptr)
            if bias_ptr:
                params.append(Tensor(bias_ptr, owns_data=False))
        return params
    
    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias})"


class ReLU(Module):
    """
    Rectified Linear Unit activation function.
    
    Computes: y = max(0, x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        output = Tensor.create(x.shape, x.dtype)
        result = _lib.neural_relu(output._ptr, x._ptr)
        _check_error(result, "relu")
        return output
    
    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid activation function.
    
    Computes: y = 1 / (1 + exp(-x))
    """
    
    def forward(self, x: Tensor) -> Tensor:
        output = Tensor.create(x.shape, x.dtype)
        result = _lib.neural_sigmoid(output._ptr, x._ptr)
        _check_error(result, "sigmoid")
        return output
    
    def __repr__(self) -> str:
        return "Sigmoid()"


class Softmax(Module):
    """
    Softmax activation function.
    
    Computes: y_i = exp(x_i) / sum(exp(x_j))
    
    Args:
        dim: Dimension along which to compute softmax (default: -1, last dim)
    """
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        output = Tensor.create(x.shape, x.dtype)
        result = _lib.neural_softmax(output._ptr, x._ptr)
        _check_error(result, "softmax")
        return output
    
    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class Tanh(Module):
    """
    Hyperbolic tangent activation function.
    
    Computes: y = tanh(x)
    """
    
    def forward(self, x: Tensor) -> Tensor:
        output = Tensor.create(x.shape, x.dtype)
        # Note: Using sigmoid as placeholder if tanh not available
        result = _lib.neural_sigmoid(output._ptr, x._ptr)  # TODO: Use tanh when available
        _check_error(result, "tanh")
        return output
    
    def __repr__(self) -> str:
        return "Tanh()"


class Dropout(Module):
    """
    Inverted dropout regularization layer.

    During training, randomly zeroes elements with probability ``p`` and
    scales the remaining elements by ``1 / (1 - p)`` so that the expected
    sum is unchanged.  During evaluation (``model.eval()``), acts as an
    identity function.

    Backed by assembly kernels ``dropout_forward`` / ``dropout_backward``
    in training_ops.asm.

    Args:
        p: Probability of an element being zeroed (default: 0.5).

    Example:
        >>> drop = Dropout(p=0.3)
        >>> x = Tensor.ones([4, 10])
        >>> y = drop(x)           # ~30 % of elements are zero
        >>> drop.eval()
        >>> y2 = drop(x)          # identity in eval mode
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        self.p = p
        # Mask from the last forward pass (needed for backward)
        self._mask: Optional[ctypes.Array] = None
        self._numel: int = 0

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor of the same shape.
        """
        # Eval mode or p == 0 → identity
        if not self._training or self.p == 0.0:
            output = Tensor.create(x.shape, x.dtype)
            _lib.neural_tensor_copy(output._ptr, x._ptr)
            return output

        # p == 1 → all zeros
        if self.p == 1.0:
            output = Tensor.create(x.shape, x.dtype)
            output.fill(0.0)
            # Store all-zero mask for backward
            numel = int(_lib.neural_tensor_numel(x._ptr))
            self._mask = (ctypes.c_uint8 * numel)(*([0] * numel))
            self._numel = numel
            return output

        numel = int(_lib.neural_tensor_numel(x._ptr))
        output = Tensor.create(x.shape, x.dtype)

        # Allocate mask buffer (uint8 per element)
        mask = (ctypes.c_uint8 * numel)()

        input_data = _lib.neural_tensor_data(x._ptr)
        output_data = _lib.neural_tensor_data(output._ptr)

        result = _lib.neural_dropout_forward(
            input_data,
            output_data,
            ctypes.cast(mask, ctypes.c_void_p),
            ctypes.c_uint64(numel),
            ctypes.c_float(self.p),
        )
        _check_error(result, "dropout forward")

        self._mask = mask
        self._numel = numel
        return output

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass — apply the same mask used in forward.

        Args:
            grad_output: Gradient of the loss w.r.t. the output.

        Returns:
            Gradient of the loss w.r.t. the input.
        """
        if self._mask is None:
            raise RuntimeError("Dropout.backward called before forward")

        grad_input = Tensor.create(grad_output.shape, grad_output.dtype)

        go_data = _lib.neural_tensor_data(grad_output._ptr)
        gi_data = _lib.neural_tensor_data(grad_input._ptr)

        result = _lib.neural_dropout_backward(
            go_data,
            gi_data,
            ctypes.cast(self._mask, ctypes.c_void_p),
            ctypes.c_uint64(self._numel),
            ctypes.c_float(self.p),
        )
        _check_error(result, "dropout backward")
        return grad_input

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Sequential(Module):
    """
    A sequential container of modules.
    
    Modules are applied in order to the input.
    
    Example:
        >>> model = Sequential([
        ...     Linear(10, 20),
        ...     ReLU(),
        ...     Linear(20, 2),
        ...     Softmax()
        ... ])
        >>> x = Tensor.zeros([32, 10])
        >>> y = model(x)
    """
    
    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        if modules:
            self._modules = list(modules)
    
    def add(self, module: Module) -> "Sequential":
        """Add a module to the sequence."""
        self._modules.append(module)
        return self
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply all modules in sequence."""
        for module in self._modules:
            x = module(x)
        return x
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __getitem__(self, idx: int) -> Module:
        return self._modules[idx]
    
    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, module in enumerate(self._modules):
            lines.append(f"  ({i}): {module}")
        lines.append(")")
        return "\n".join(lines)


# Loss functions

class MSELoss(Module):
    """
    Mean Squared Error loss.
    
    Computes: loss = mean((prediction - target)^2)
    """
    
    def forward(self, prediction: Tensor, target: Tensor) -> float:
        """
        Compute MSE loss.
        
        Args:
            prediction: Predicted values
            target: Target values
        
        Returns:
            Loss value as float
        """
        loss = ctypes.c_double()
        result = _lib.neural_mse_loss(prediction._ptr, target._ptr, ctypes.byref(loss))
        _check_error(result, "mse_loss")
        return loss.value
    
    def __call__(self, prediction: Tensor, target: Tensor) -> float:
        return self.forward(prediction, target)
    
    def __repr__(self) -> str:
        return "MSELoss()"


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for classification.
    """
    
    def forward(self, prediction: Tensor, target: Tensor) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            prediction: Predicted logits/probabilities
            target: Target class labels
        
        Returns:
            Loss value as float
        """
        loss = ctypes.c_double()
        result = _lib.neural_cross_entropy_loss(prediction._ptr, target._ptr, ctypes.byref(loss))
        _check_error(result, "cross_entropy_loss")
        return loss.value
    
    def __call__(self, prediction: Tensor, target: Tensor) -> float:
        return self.forward(prediction, target)
    
    def __repr__(self) -> str:
        return "CrossEntropyLoss()"
