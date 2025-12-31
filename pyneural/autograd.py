"""
Autograd utilities for PyNeural.

This module provides automatic differentiation utilities including:
- no_grad(): Context manager to disable gradient computation
- enable_grad(): Context manager to enable gradient computation
- set_grad_enabled(): Function to set gradient state
- is_grad_enabled(): Function to check gradient state
- GradientHook: Class for registering gradient hooks
"""

from contextlib import contextmanager
from typing import Callable, Generator, List, Optional, Any

# Global gradient tracking state
_grad_enabled = True

# Gradient hooks registry
_gradient_hooks: List[Callable] = []


@contextmanager
def no_grad() -> Generator[None, None, None]:
    """
    Context manager to disable gradient computation.
    
    Useful for inference to reduce memory usage and computation time.
    Operations performed in this context will not track gradients.
    
    Example:
        >>> with no_grad():
        ...     output = model(input)  # No gradients computed
        ...     loss = criterion(output, target)  # Also no gradients
    
    Note:
        This is commonly used during:
        - Model evaluation/inference
        - Computing metrics that shouldn't affect training
        - Loading and preprocessing data
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
    
    Can be used to override no_grad() context when needed.
    
    Example:
        >>> with no_grad():
        ...     x = compute_something()  # No gradients
        ...     with enable_grad():
        ...         y = x * 2  # Gradients ARE tracked here
        ...     z = y + 1  # No gradients again
    """
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled() -> bool:
    """
    Check if gradient computation is currently enabled.
    
    Returns:
        True if gradients are being tracked, False otherwise.
        
    Example:
        >>> print(is_grad_enabled())  # True (default)
        >>> with no_grad():
        ...     print(is_grad_enabled())  # False
        >>> print(is_grad_enabled())  # True again
    """
    return _grad_enabled


def set_grad_enabled(enabled: bool) -> bool:
    """
    Set gradient computation enabled state.
    
    Args:
        enabled: Whether to enable gradient computation
        
    Returns:
        The previous state (for restoration if needed)
    
    Example:
        >>> prev_state = set_grad_enabled(False)
        >>> # Do inference
        >>> set_grad_enabled(prev_state)  # Restore
    """
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = enabled
    return prev


class inference_mode:
    """
    Context manager for inference mode (similar to no_grad but more efficient).
    
    This is an alias for no_grad() in PyNeural, but exists for API compatibility
    with PyTorch-style code.
    
    Example:
        >>> with inference_mode():
        ...     predictions = model(data)
    """
    
    def __init__(self):
        self._prev_state = None
    
    def __enter__(self):
        global _grad_enabled
        self._prev_state = _grad_enabled
        _grad_enabled = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _grad_enabled
        _grad_enabled = self._prev_state
        return False


def register_hook(hook: Callable[[Any], Optional[Any]]) -> int:
    """
    Register a global gradient hook.
    
    The hook will be called on all gradient computations with the gradient
    tensor as argument. The hook can optionally return a modified gradient.
    
    Args:
        hook: Callable that takes a gradient and optionally returns a modified gradient
        
    Returns:
        Hook handle (index) for removal
        
    Example:
        >>> def print_grad(grad):
        ...     print(f"Gradient: {grad.shape}")
        ...     return grad  # Return unchanged
        >>> handle = register_hook(print_grad)
    """
    _gradient_hooks.append(hook)
    return len(_gradient_hooks) - 1


def remove_hook(handle: int) -> bool:
    """
    Remove a gradient hook by its handle.
    
    Args:
        handle: Hook handle returned by register_hook()
        
    Returns:
        True if hook was removed, False if handle was invalid or already removed
    """
    if 0 <= handle < len(_gradient_hooks) and _gradient_hooks[handle] is not None:
        _gradient_hooks[handle] = None  # Mark as removed
        return True
    return False


def clear_hooks() -> None:
    """Remove all gradient hooks."""
    global _gradient_hooks
    _gradient_hooks = []


def _apply_hooks(grad: Any) -> Any:
    """Apply all registered hooks to a gradient (internal function)."""
    result = grad
    for hook in _gradient_hooks:
        if hook is not None:
            hook_result = hook(result)
            if hook_result is not None:
                result = hook_result
    return result
