#!/usr/bin/env python3
"""
Test suite for autograd functionality (Issue #18).

Tests:
- no_grad context manager
- enable_grad context manager  
- set_grad_enabled function
- is_grad_enabled function
- inference_mode context manager
- Tensor.detach() method
- Tensor.detach_() in-place method
- Gradient hooks
- requires_grad property
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyneural as pn


def test_no_grad():
    """Test no_grad context manager."""
    print("Testing no_grad()...", end=" ")
    
    # Should be enabled by default
    assert pn.is_grad_enabled(), "Gradients should be enabled by default"
    
    with pn.no_grad():
        assert not pn.is_grad_enabled(), "Gradients should be disabled in no_grad()"
        
        # Nested no_grad should work
        with pn.no_grad():
            assert not pn.is_grad_enabled(), "Nested no_grad should also be disabled"
        
        # Still disabled after inner context
        assert not pn.is_grad_enabled(), "Should still be disabled after inner no_grad"
    
    # Should be restored after context
    assert pn.is_grad_enabled(), "Gradients should be restored after no_grad()"
    
    print("PASSED")


def test_enable_grad():
    """Test enable_grad context manager."""
    print("Testing enable_grad()...", end=" ")
    
    # Test enable_grad inside no_grad
    with pn.no_grad():
        assert not pn.is_grad_enabled()
        
        with pn.enable_grad():
            assert pn.is_grad_enabled(), "enable_grad should override no_grad"
        
        # Should go back to disabled
        assert not pn.is_grad_enabled(), "Should be disabled after enable_grad exits"
    
    assert pn.is_grad_enabled(), "Should be enabled at the end"
    
    print("PASSED")


def test_set_grad_enabled():
    """Test set_grad_enabled function."""
    print("Testing set_grad_enabled()...", end=" ")
    
    # Save current state
    original = pn.is_grad_enabled()
    
    # Disable
    prev = pn.set_grad_enabled(False)
    assert prev == original, "Should return previous state"
    assert not pn.is_grad_enabled(), "Should be disabled"
    
    # Enable
    prev = pn.set_grad_enabled(True)
    assert not prev, "Previous should have been False"
    assert pn.is_grad_enabled(), "Should be enabled"
    
    # Restore
    pn.set_grad_enabled(original)
    
    print("PASSED")


def test_inference_mode():
    """Test inference_mode context manager."""
    print("Testing inference_mode()...", end=" ")
    
    assert pn.is_grad_enabled()
    
    with pn.inference_mode():
        assert not pn.is_grad_enabled(), "inference_mode should disable gradients"
    
    assert pn.is_grad_enabled(), "Should be restored after inference_mode"
    
    print("PASSED")


def test_tensor_detach():
    """Test Tensor.detach() method."""
    print("Testing Tensor.detach()...", end=" ")
    
    pn.init()
    
    # Create tensor with requires_grad
    x = pn.Tensor.from_list([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad = True
    
    assert x.requires_grad, "Original should have requires_grad=True"
    
    # Detach
    y = x.detach()
    
    assert not y.requires_grad, "Detached tensor should have requires_grad=False"
    
    # Check that they share data (same pointer or same values)
    x_data = x.tolist()
    y_data = y.tolist()
    
    for i in range(2):
        for j in range(2):
            assert abs(x_data[i][j] - y_data[i][j]) < 1e-6, "Data should match"
    
    print("PASSED")


def test_tensor_detach_inplace():
    """Test Tensor.detach_() in-place method."""
    print("Testing Tensor.detach_()...", end=" ")
    
    x = pn.Tensor.from_list([1.0, 2.0, 3.0])
    x.requires_grad = True
    
    assert x.requires_grad
    
    # In-place detach
    result = x.detach_()
    
    assert result is x, "detach_() should return self"
    assert not x.requires_grad, "requires_grad should be False after detach_()"
    
    print("PASSED")


def test_gradient_hooks():
    """Test gradient hook registration."""
    print("Testing gradient hooks...", end=" ")
    
    hook_called = [False]  # Use list to allow mutation in closure
    
    def my_hook(grad):
        hook_called[0] = True
        return grad
    
    # Register hook
    handle = pn.register_hook(my_hook)
    assert handle >= 0, "Should return valid handle"
    
    # Remove hook
    assert pn.remove_hook(handle), "Should successfully remove hook"
    
    # Removing again should fail
    assert not pn.remove_hook(handle), "Second removal should fail"
    
    # Clear hooks
    pn.register_hook(lambda g: g)
    pn.register_hook(lambda g: g)
    pn.clear_hooks()  # Should not raise
    
    print("PASSED")


def test_requires_grad_property():
    """Test requires_grad property on tensors."""
    print("Testing requires_grad property...", end=" ")
    
    x = pn.Tensor.zeros([3, 3])
    
    # Default should be False
    assert not x.requires_grad, "Default should be False"
    
    # Set to True
    x.requires_grad = True
    assert x.requires_grad, "Should be True after setting"
    
    # Set back to False
    x.requires_grad = False
    assert not x.requires_grad, "Should be False after unsetting"
    
    print("PASSED")


def test_grad_fn_property():
    """Test grad_fn property on tensors."""
    print("Testing grad_fn property...", end=" ")
    
    x = pn.Tensor.ones([2, 2])
    
    # Should be None by default
    assert x.grad_fn is None, "grad_fn should be None for leaf tensors"
    
    # Detach should clear grad_fn
    x.requires_grad = True
    y = x.detach()
    assert y.grad_fn is None, "Detached tensor should have no grad_fn"
    
    print("PASSED")


def test_is_leaf_property():
    """Test is_leaf property on tensors."""
    print("Testing is_leaf property...", end=" ")
    
    x = pn.Tensor.ones([2, 2])
    x.requires_grad = True
    
    # User-created tensor with requires_grad should be leaf
    assert x.is_leaf, "User-created tensor with requires_grad should be leaf"
    
    # Detached tensor without requires_grad is not leaf
    y = x.detach()
    assert not y.is_leaf, "Tensor without requires_grad is not leaf"
    
    print("PASSED")


def test_exception_safety():
    """Test that gradient state is restored even if exception occurs."""
    print("Testing exception safety...", end=" ")
    
    assert pn.is_grad_enabled()
    
    try:
        with pn.no_grad():
            assert not pn.is_grad_enabled()
            raise ValueError("Test exception")
    except ValueError:
        pass  # Expected
    
    # State should be restored even after exception
    assert pn.is_grad_enabled(), "State should be restored after exception"
    
    print("PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Autograd Functionality Tests (Issue #18)")
    print("=" * 60)
    
    pn.init()
    
    tests = [
        test_no_grad,
        test_enable_grad,
        test_set_grad_enabled,
        test_inference_mode,
        test_tensor_detach,
        test_tensor_detach_inplace,
        test_gradient_hooks,
        test_requires_grad_property,
        test_grad_fn_property,
        test_is_leaf_property,
        test_exception_safety,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    pn.shutdown()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
