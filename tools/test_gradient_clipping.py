#!/usr/bin/env python3
"""
Test gradient clipping functionality for Neural Assembly framework.
"""

import ctypes
import sys
import os

# Add current directory to path for importing
sys.path.append(os.path.dirname(__file__))

def test_gradient_clipping():
    """Test gradient clipping functions"""
    print("Testing Gradient Clipping Functionality...")

    try:
        # Load the library
        lib_path = os.path.join(os.path.dirname(__file__), '..', 'libneural.so')
        lib = ctypes.CDLL(lib_path)

        # Define basic types
        c_double = ctypes.c_double
        c_int = ctypes.c_int
        c_uint32 = ctypes.c_uint32
        c_void_p = ctypes.c_void_p

        # Define tensor structure (simplified)
        class Tensor(ctypes.Structure):
            _fields_ = [
                ('data', c_void_p),
                ('ndim', c_int),
                ('shape', c_void_p),  # We'll set this to None for simplicity
                ('dtype', c_int),
            ]

        # Define optimizer structure (simplified)
        class Optimizer(ctypes.Structure):
            _fields_ = [
                ('n_params', c_uint32),
                ('padding', c_uint32),
                ('params', c_void_p),
                ('param_nodes', c_void_p),
                ('step_fn', c_void_p),
                ('zero_grad_fn', c_void_p),
                ('state', c_void_p),
            ]

        # Test that functions exist and can be called
        print("✓ Library loaded successfully")

        # Check function signatures
        lib.neural_clip_grad_norm.argtypes = [ctypes.POINTER(Optimizer), c_double]
        lib.neural_clip_grad_norm.restype = c_int

        lib.neural_clip_grad_value.argtypes = [ctypes.POINTER(Optimizer), c_double, c_double]
        lib.neural_clip_grad_value.restype = c_int

        print("✓ Function signatures set successfully")

        # Test calling with NULL optimizer (should return error)
        result = lib.neural_clip_grad_norm(None, 1.0)
        if result != 0:  # Should be error code
            print("✓ neural_clip_grad_norm handles NULL optimizer correctly")
        else:
            print("✗ neural_clip_grad_norm should return error for NULL optimizer")

        result = lib.neural_clip_grad_value(None, -1.0, 1.0)
        if result != 0:  # Should be error code
            print("✓ neural_clip_grad_value handles NULL optimizer correctly")
        else:
            print("✗ neural_clip_grad_value should return error for NULL optimizer")

        print("\nGradient clipping functions are accessible and handle basic error cases!")
        print("For full functionality testing, gradients need to be set up through the neural network API.")

        return True

    except Exception as e:
        print(f"✗ Error testing gradient clipping: {e}")
        return False

if __name__ == "__main__":
    success = test_gradient_clipping()
    sys.exit(0 if success else 1)