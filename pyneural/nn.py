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
        result = _lib.act_tanh(output._ptr, x._ptr)
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


# Normalization layers

class BatchNorm1d(Module):
    """
    Batch Normalization over a 2D input (B × C).

    During training, normalises using batch statistics and updates
    running mean/var via exponential moving average.  During evaluation,
    uses the accumulated running statistics.

    Backed by C implementation in batchnorm.c using float64 arrays.

    Args:
        num_features: Number of features (C dimension).
        momentum: EMA coefficient for running stats (default: 0.1).
        eps: Small constant for numerical stability (default: 1e-5).

    Example:
        >>> bn = BatchNorm1d(64)
        >>> x = ...  # (batch, 64) float64 array
        >>> y = bn.forward_f64(x_ptr, out_ptr, batch_size)
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self._ptr = _lib.batchnorm1d_create(num_features, momentum, eps)
        if self._ptr is None or self._ptr == 0:
            raise NeuralException(0, "Failed to create BatchNorm1d")

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.batchnorm1d_free(self._ptr)
            self._ptr = None

    def forward_f64(self, input_ptr, output_ptr, batch_size: int) -> int:
        """
        Forward pass on raw float64 pointers.

        Args:
            input_ptr: ctypes pointer to input data (B × C doubles).
            output_ptr: ctypes pointer to output buffer (B × C doubles).
            batch_size: Number of samples in the batch.

        Returns:
            0 on success.
        """
        return _lib.batchnorm1d_forward(
            self._ptr, input_ptr, output_ptr,
            batch_size, 1 if self._training else 0
        )

    def backward_f64(self, grad_out_ptr, grad_in_ptr,
                     grad_gamma_ptr, grad_beta_ptr, batch_size: int) -> int:
        """Backward pass on raw float64 pointers."""
        return _lib.batchnorm1d_backward(
            self._ptr, grad_out_ptr, grad_in_ptr,
            grad_gamma_ptr, grad_beta_ptr, batch_size
        )

    @property
    def gamma_ptr(self):
        return _lib.batchnorm1d_gamma(self._ptr)

    @property
    def beta_ptr(self):
        return _lib.batchnorm1d_beta(self._ptr)

    @property
    def running_mean_ptr(self):
        return _lib.batchnorm1d_running_mean(self._ptr)

    @property
    def running_var_ptr(self):
        return _lib.batchnorm1d_running_var(self._ptr)

    def __repr__(self) -> str:
        return f"BatchNorm1d({self.num_features}, momentum={self.momentum}, eps={self.eps})"


class LayerNorm(Module):
    """
    Layer Normalization over the last dimension of a 2D input (B × C).

    Normalises across features for each sample independently.
    Always uses the sample's own statistics (no running stats).

    Backed by C implementation in batchnorm.c using float64 arrays.

    Args:
        num_features: Number of features (C dimension).
        eps: Small constant for numerical stability (default: 1e-5).
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self._ptr = _lib.layernorm_create(num_features, eps)
        if self._ptr is None or self._ptr == 0:
            raise NeuralException(0, "Failed to create LayerNorm")

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.layernorm_free(self._ptr)
            self._ptr = None

    def forward_f64(self, input_ptr, output_ptr, batch_size: int) -> int:
        """Forward pass on raw float64 pointers."""
        return _lib.layernorm_forward(
            self._ptr, input_ptr, output_ptr, batch_size
        )

    def backward_f64(self, grad_out_ptr, grad_in_ptr,
                     grad_gamma_ptr, grad_beta_ptr, batch_size: int) -> int:
        """Backward pass on raw float64 pointers."""
        return _lib.layernorm_backward(
            self._ptr, grad_out_ptr, grad_in_ptr,
            grad_gamma_ptr, grad_beta_ptr, batch_size
        )

    @property
    def gamma_ptr(self):
        return _lib.layernorm_gamma(self._ptr)

    @property
    def beta_ptr(self):
        return _lib.layernorm_beta(self._ptr)

    def __repr__(self) -> str:
        return f"LayerNorm({self.num_features}, eps={self.eps})"


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


class LabelSmoothingCrossEntropy(Module):
    """
    Cross-entropy loss with label smoothing.

    Smoothed target:  (1 - α) · one_hot(target) + α / K

    Works with raw double* arrays via forward_f64 / backward_f64,
    or with Python lists via forward / backward.

    Args:
        smoothing: Label smoothing factor α ∈ [0, 1].  0 = standard CE.
        num_classes: Number of output classes (required).
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.smoothing = smoothing
        self.num_classes = num_classes
        self._cached_logits = None
        self._cached_targets = None
        self._cached_batch_size = None

    def forward_f64(self, logits_ptr, targets_ptr, batch_size):
        """
        Forward pass using raw double*/int64_t* pointers.

        Args:
            logits_ptr:  c_void_p to double[batch_size * num_classes]
            targets_ptr: c_void_p to int64_t[batch_size]
            batch_size:  number of samples

        Returns:
            float: mean loss
        """
        loss = ctypes.c_double()
        result = _lib.label_smoothing_ce_forward(
            logits_ptr, targets_ptr,
            batch_size, self.num_classes,
            self.smoothing, ctypes.byref(loss)
        )
        _check_error(result, "label_smoothing_ce_forward")
        self._cached_logits = logits_ptr
        self._cached_targets = targets_ptr
        self._cached_batch_size = batch_size
        return loss.value

    def backward_f64(self, grad_out_ptr, logits_ptr=None, targets_ptr=None, batch_size=None):
        """
        Backward pass using raw pointers.  Falls back to cached values from forward.

        Args:
            grad_out_ptr: c_void_p to double[batch_size * num_classes] (output)
            logits_ptr:   optional override (default: use cached from forward)
            targets_ptr:  optional override
            batch_size:   optional override
        """
        lp = logits_ptr if logits_ptr is not None else self._cached_logits
        tp = targets_ptr if targets_ptr is not None else self._cached_targets
        bs = batch_size if batch_size is not None else self._cached_batch_size
        if lp is None or tp is None or bs is None:
            raise RuntimeError("backward called before forward; provide explicit pointers")
        result = _lib.label_smoothing_ce_backward(
            lp, tp, bs, self.num_classes, self.smoothing, grad_out_ptr
        )
        _check_error(result, "label_smoothing_ce_backward")

    def forward(self, logits_list, targets_list):
        """
        Convenience: accept Python lists.

        Args:
            logits_list: flat list of floats, length batch_size * num_classes
            targets_list: list of int class indices, length batch_size
        Returns:
            float: mean loss
        """
        bs = len(targets_list)
        nc = self.num_classes
        logits_arr = (ctypes.c_double * (bs * nc))(*logits_list)
        targets_arr = (ctypes.c_int64 * bs)(*targets_list)
        return self.forward_f64(
            ctypes.cast(logits_arr, ctypes.c_void_p),
            ctypes.cast(targets_arr, ctypes.c_void_p),
            bs
        )

    def __call__(self, logits, targets):
        return self.forward(logits, targets)

    def __repr__(self) -> str:
        return f"LabelSmoothingCrossEntropy(num_classes={self.num_classes}, smoothing={self.smoothing})"


class Embedding(Module):
    """
    Embedding lookup table (C-backed).

    Stores a weight matrix W[num_embeddings × embedding_dim] in float64.
    Forward maps integer indices to their corresponding embedding vectors.

    Args:
        num_embeddings: Size of the dictionary (vocabulary size).
        embedding_dim: Size of each embedding vector.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        if num_embeddings < 1:
            raise ValueError(f"num_embeddings must be >= 1, got {num_embeddings}")
        if embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {embedding_dim}")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._ptr = _lib.embedding_create(num_embeddings, embedding_dim)
        if not self._ptr:
            raise MemoryError("Failed to create Embedding")
        self._cached_indices = None
        self._cached_seq_len = None

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.embedding_free(self._ptr)
            self._ptr = None

    @property
    def weight_ptr(self):
        """Raw pointer to the weight matrix (double*)."""
        return _lib.embedding_weight(self._ptr)

    def forward_f64(self, indices_ptr, seq_len, output_ptr):
        """
        Forward pass using raw pointers.

        Args:
            indices_ptr: c_void_p to int64_t[seq_len]
            seq_len: number of indices
            output_ptr: c_void_p to double[seq_len * embedding_dim]
        Returns:
            int: 0 on success
        """
        result = _lib.embedding_forward(self._ptr, indices_ptr, seq_len, output_ptr)
        _check_error(result, "embedding_forward")
        self._cached_indices = indices_ptr
        self._cached_seq_len = seq_len
        return result

    def backward_f64(self, grad_output_ptr, grad_weight_ptr,
                     indices_ptr=None, seq_len=None):
        """
        Backward pass using raw pointers.

        Args:
            grad_output_ptr: c_void_p to double[seq_len * embedding_dim]
            grad_weight_ptr: c_void_p to double[num_embeddings * embedding_dim]
                             (accumulated, must be pre-zeroed if fresh)
            indices_ptr: optional override
            seq_len: optional override
        """
        ip = indices_ptr if indices_ptr is not None else self._cached_indices
        sl = seq_len if seq_len is not None else self._cached_seq_len
        if ip is None or sl is None:
            raise RuntimeError("backward called before forward; provide explicit pointers")
        result = _lib.embedding_backward(self._ptr, ip, sl, grad_output_ptr, grad_weight_ptr)
        _check_error(result, "embedding_backward")

    def forward(self, indices_list):
        """
        Convenience: accept Python list of int indices.

        Args:
            indices_list: List of integer indices.

        Returns:
            List of float (flat, length = len(indices) * embedding_dim).
        """
        seq_len = len(indices_list)
        idx_arr = (ctypes.c_int64 * seq_len)(*indices_list)
        out_arr = (ctypes.c_double * (seq_len * self.embedding_dim))()
        self.forward_f64(
            ctypes.cast(idx_arr, ctypes.c_void_p),
            seq_len,
            ctypes.cast(out_arr, ctypes.c_void_p),
        )
        return [float(out_arr[i]) for i in range(seq_len * self.embedding_dim)]

    def __call__(self, indices):
        return self.forward(indices)

    def __repr__(self) -> str:
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
