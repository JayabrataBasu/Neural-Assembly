"""
Architecture DSL parser for PyNeural.

Parses architecture definition strings into sequences of nn.Module layers.

Supports two syntaxes:

1. **Simple (legacy):** Comma-separated integers for dense networks.
   ```
   architecture = 784,256,128,10
   ```
   Builds: Linear(784,256) -> ReLU -> Linear(256,128) -> ReLU -> Linear(128,10)

2. **Extended:** Named layer specifications with parameters.
   ```
   architecture = Conv2D(1,32,3,padding=1),ReLU,MaxPool2D(2),
                  Conv2D(32,64,3),ReLU,MaxPool2D(2),
                  Flatten,
                  Linear(1024,128),ReLU,Dropout(0.5),
                  Linear(128,10),Softmax
   ```

Example:
    >>> from pyneural.arch import parse_architecture
    >>> model = parse_architecture("784,256,128,10", activation="relu")
    >>> model = parse_architecture("Linear(784,256),ReLU,Dropout(0.3),Linear(256,10)")
"""

from __future__ import annotations

import re
from typing import List, Optional

from . import nn
from .conv import Conv2D, MaxPool2D
from .pooling import AvgPool2D
from .activations import (
    GELU, LeakyReLU, ELU, SELU, Swish, Mish, HardSwish, Softplus, HardTanh,
)


# ---------------------------------------------------------------------------
# Layer registry
# ---------------------------------------------------------------------------

_LAYER_MAP = {
    # Dense
    "linear": nn.Linear,
    "dense": nn.Linear,
    # Activations
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "tanh": nn.Tanh,
    "gelu": GELU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "selu": SELU,
    "swish": Swish,
    "mish": Mish,
    "hardswish": HardSwish,
    "softplus": Softplus,
    "hardtanh": HardTanh,
    # Regularization
    "dropout": nn.Dropout,
    "batchnorm1d": nn.BatchNorm1d,
    "batchnorm": nn.BatchNorm1d,
    "layernorm": nn.LayerNorm,
    # Structural
    "flatten": nn.Flatten,
    # Convolution / Pooling
    "conv2d": Conv2D,
    "maxpool2d": MaxPool2D,
    "maxpool": MaxPool2D,
    "avgpool2d": AvgPool2D,
    "avgpool": AvgPool2D,
}

_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "gelu": GELU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "selu": SELU,
    "swish": Swish,
    "mish": Mish,
}


# ---------------------------------------------------------------------------
# Token parser
# ---------------------------------------------------------------------------

def _tokenize(arch_str: str) -> List[str]:
    """
    Split an architecture string into layer tokens, respecting parentheses.

    "Conv2D(1,32,3),ReLU,Linear(32,10)" -> ["Conv2D(1,32,3)", "ReLU", "Linear(32,10)"]
    """
    tokens = []
    current = ""
    depth = 0
    for ch in arch_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            token = current.strip()
            if token:
                tokens.append(token)
            current = ""
        else:
            current += ch
    token = current.strip()
    if token:
        tokens.append(token)
    return tokens


def _parse_layer_token(token: str) -> nn.Module:
    """
    Parse a single layer token string into a Module instance.

    Examples:
        "ReLU"            -> ReLU()
        "Linear(128,64)"  -> Linear(128, 64)
        "Dropout(0.5)"    -> Dropout(p=0.5)
        "Conv2D(1,32,3,padding=1)" -> Conv2D(1, 32, 3, padding=1)
    """
    # Match "Name(args)" or just "Name"
    m = re.match(r'^(\w+)(?:\((.*)\))?$', token.strip())
    if not m:
        raise ValueError(f"Invalid layer specification: '{token}'")

    name = m.group(1)
    args_str = m.group(2)

    layer_cls = _LAYER_MAP.get(name.lower())
    if layer_cls is None:
        raise ValueError(
            f"Unknown layer type: '{name}'. "
            f"Available: {', '.join(sorted(_LAYER_MAP.keys()))}"
        )

    if args_str is None or args_str.strip() == "":
        return layer_cls()

    # Parse positional and keyword arguments
    pos_args = []
    kw_args = {}
    for arg in _split_args(args_str):
        arg = arg.strip()
        if "=" in arg:
            k, v = arg.split("=", 1)
            kw_args[k.strip()] = _parse_value(v.strip())
        else:
            pos_args.append(_parse_value(arg))

    return layer_cls(*pos_args, **kw_args)


def _split_args(args_str: str) -> List[str]:
    """Split argument string by commas, respecting nested parens."""
    args = []
    current = ""
    depth = 0
    for ch in args_str:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    return args


def _parse_value(s: str):
    """Parse a string value to the appropriate Python type."""
    # Boolean
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    # Integer
    try:
        return int(s)
    except ValueError:
        pass
    # Float
    try:
        return float(s)
    except ValueError:
        pass
    # String
    return s


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def _is_simple_architecture(arch_str: str) -> bool:
    """Check if architecture string is the simple integer-only format."""
    tokens = _tokenize(arch_str)
    return all(t.isdigit() for t in tokens)


def parse_architecture(
    arch_str: str,
    activation: str = "relu",
    output_activation: Optional[str] = None,
    dropout: Optional[float] = None,
) -> nn.Sequential:
    """
    Parse an architecture definition string into a Sequential model.

    Supports two formats:

    **Simple format** (comma-separated integers):
        "784,256,128,10" -> Linear layers with activation between each.

    **Extended format** (named layers with parameters):
        "Conv2D(1,32,3),ReLU,Flatten,Linear(32,10)"

    Args:
        arch_str: Architecture definition string.
        activation: Activation function for simple format (default: 'relu').
        output_activation: Optional activation for the final layer in simple format.
        dropout: Optional dropout rate between layers in simple format.

    Returns:
        A Sequential model.
    """
    arch_str = arch_str.strip()
    if not arch_str:
        raise ValueError("Empty architecture string")

    if _is_simple_architecture(arch_str):
        return _parse_simple(arch_str, activation, output_activation, dropout)
    else:
        return _parse_extended(arch_str)


def _parse_simple(
    arch_str: str,
    activation: str,
    output_activation: Optional[str],
    dropout: Optional[float],
) -> nn.Sequential:
    """Parse the simple integer-only architecture format."""
    sizes = [int(t) for t in _tokenize(arch_str)]
    if len(sizes) < 2:
        raise ValueError("Simple architecture needs at least 2 layer sizes")

    act_cls = _ACTIVATION_MAP.get(activation.lower())
    if act_cls is None:
        raise ValueError(f"Unknown activation: '{activation}'")

    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        # Add activation (skip for last layer unless output_activation specified)
        if i < len(sizes) - 2:
            layers.append(act_cls())
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        elif output_activation is not None:
            out_cls = _ACTIVATION_MAP.get(output_activation.lower())
            if out_cls:
                layers.append(out_cls())

    return nn.Sequential(layers)


def _parse_extended(arch_str: str) -> nn.Sequential:
    """Parse the extended named-layer architecture format."""
    tokens = _tokenize(arch_str)
    layers = [_parse_layer_token(t) for t in tokens]
    return nn.Sequential(layers)


# ---------------------------------------------------------------------------
# ResidualBlock DSL support
# ---------------------------------------------------------------------------

def parse_residual_block(block_str: str) -> nn.ResidualBlock:
    """
    Parse a residual block specification.

    Format: "ResidualBlock(Linear(128,128),ReLU,Linear(128,128))"

    Args:
        block_str: Block definition string.

    Returns:
        A ResidualBlock module.
    """
    m = re.match(r'^ResidualBlock\((.*)\)$', block_str.strip())
    if not m:
        raise ValueError(f"Invalid ResidualBlock specification: '{block_str}'")

    inner = m.group(1)
    tokens = _tokenize(inner)
    layers = [_parse_layer_token(t) for t in tokens]
    return nn.ResidualBlock(layers)
