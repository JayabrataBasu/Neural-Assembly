"""
pyneural.activations — Extended activation function modules.

Each class wraps an assembly kernel from activations.asm via the C API.
Python does nothing here except allocate the output tensor and call
through ctypes — all actual math runs in x86-64 assembly.

The four "classic" activations (ReLU, Sigmoid, Softmax, Tanh) live in
nn.py for historical reasons.  Everything else is here.
"""

from __future__ import annotations

from .core import _lib, _check_error
from .tensor import Tensor
from .nn import Module


class GELU(Module):
    """
    Gaussian Error Linear Unit.

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Used heavily in transformers (BERT, GPT).
    """

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(_lib.neural_gelu(out._ptr, x._ptr), "gelu")
        return out

    def __repr__(self) -> str:
        return "GELU()"


class LeakyReLU(Module):
    """
    Leaky ReLU: max(alpha * x, x).

    Lets a small gradient through for negative inputs instead of
    clamping to zero.

    Args:
        alpha: Slope for negative region (default 0.01).
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(
            _lib.neural_leaky_relu(out._ptr, x._ptr, self.alpha),
            "leaky_relu",
        )
        return out

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class ELU(Module):
    """
    Exponential Linear Unit.

    ELU(x) = x if x > 0, else alpha * (exp(x) - 1)

    Pushes mean activations closer to zero, which can speed up
    convergence compared to plain ReLU.

    Args:
        alpha: Scale for the negative factor (default 1.0).
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(
            _lib.neural_elu(out._ptr, x._ptr, self.alpha),
            "elu",
        )
        return out

    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha})"


class SELU(Module):
    """
    Scaled Exponential Linear Unit.

    SELU(x) = lambda * (x if x > 0, else alpha * (exp(x) - 1))

    With the self-normalising constants lambda ~= 1.0507, alpha ~= 1.6733.
    Designed so that activations converge to zero mean and unit variance
    when weights are initialised with LeCun normal.  No batch norm needed.
    """

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(_lib.neural_selu(out._ptr, x._ptr), "selu")
        return out

    def __repr__(self) -> str:
        return "SELU()"


class Swish(Module):
    """
    Swish / SiLU (Sigmoid Linear Unit).

    Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Smooth, non-monotonic.  Often outperforms ReLU in deeper networks.
    """

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(_lib.neural_swish(out._ptr, x._ptr), "swish")
        return out

    def __repr__(self) -> str:
        return "Swish()"


class Mish(Module):
    """
    Mish activation.

    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Similar to Swish but slightly smoother near zero.
    """

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(_lib.neural_mish(out._ptr, x._ptr), "mish")
        return out

    def __repr__(self) -> str:
        return "Mish()"


class HardSwish(Module):
    """
    Hard Swish — a piecewise-linear approximation of Swish.

    HardSwish(x) = 0 if x <= -3, x if x >= 3, else x * (x + 3) / 6

    Faster to compute than real Swish, used in MobileNetV3.
    """

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(_lib.neural_hardswish(out._ptr, x._ptr), "hardswish")
        return out

    def __repr__(self) -> str:
        return "HardSwish()"


class Softplus(Module):
    """
    Softplus — smooth approximation of ReLU.

    Softplus(x) = ln(1 + exp(x))

    Always positive, differentiable everywhere.
    """

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(_lib.neural_softplus(out._ptr, x._ptr), "softplus")
        return out

    def __repr__(self) -> str:
        return "Softplus()"


class HardTanh(Module):
    """
    Hard Tanh — piecewise-linear approximation of tanh.

    HardTanh(x) = -1 if x < -1, 1 if x > 1, else x

    Cheap and useful as a clamping activation.
    """

    def forward(self, x: Tensor) -> Tensor:
        out = Tensor.create(x.shape, x.dtype)
        _check_error(_lib.neural_hardtanh(out._ptr, x._ptr), "hardtanh")
        return out

    def __repr__(self) -> str:
        return "HardTanh()"
