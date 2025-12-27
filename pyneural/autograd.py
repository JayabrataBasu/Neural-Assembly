"""
Autograd utilities for PyNeural.
"""

from contextlib import contextmanager
from typing import Generator

# Global gradient tracking state
_grad_enabled = True


@contextmanager
def no_grad() -> Generator[None, None, None]:
    """
    Context manager to disable gradient computation.
    
    Useful for inference to reduce memory usage.
    
    Example:
        >>> with no_grad():
        ...     output = model(input)
    """
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


@contextmanager
def enable_grad() -> Generator[None, None, None]:
    """
    Context manager to enable gradient computation.
    
    Can be used to override no_grad() context.
    
    Example:
        >>> with no_grad():
        ...     with enable_grad():
        ...         # Gradients are tracked here
        ...         pass
    """
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled."""
    return _grad_enabled


def set_grad_enabled(enabled: bool) -> None:
    """Set gradient computation enabled state."""
    global _grad_enabled
    _grad_enabled = enabled
