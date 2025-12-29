#!/usr/bin/env python3
"""
Debug test for gradient clipping segfault.
This isolates the exact point of failure.
"""

import ctypes
import sys
import os
import signal

def signal_handler(signum, frame):
    print(f"Signal {signum} received at frame {frame}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_clipping_segfault():
    """Test to isolate the segfault in gradient clipping"""
    print("Testing Gradient Clipping Segfault Isolation...")

    signal.signal(signal.SIGSEGV, signal_handler)

    try:
        # Load the library
        lib_path = os.path.join(os.path.dirname(__file__), '..', 'libneural.so')
        lib = ctypes.CDLL(lib_path)

        # Define types
        c_double = ctypes.c_double
        c_int = ctypes.c_int

        # Define optimizer structure (simplified)
        class Optimizer(ctypes.Structure):
            _fields_ = [
                ('n_params', ctypes.c_uint32),
                ('padding', ctypes.c_uint32),
                ('params', ctypes.c_void_p),
                ('param_nodes', ctypes.c_void_p),
                ('step_fn', ctypes.c_void_p),
                ('zero_grad_fn', ctypes.c_void_p),
                ('state', ctypes.c_void_p),
            ]

        # Set function signatures
        lib.neural_sgd_create.argtypes = [c_double, c_double]
        lib.neural_sgd_create.restype = ctypes.POINTER(Optimizer)

        lib.neural_optimizer_free.argtypes = [ctypes.POINTER(Optimizer)]
        lib.neural_optimizer_free.restype = None

        lib.neural_clip_grad_norm.argtypes = [ctypes.POINTER(Optimizer), c_double]
        lib.neural_clip_grad_norm.restype = c_int

        lib.neural_clip_grad_value.argtypes = [ctypes.POINTER(Optimizer), c_double, c_double]
        lib.neural_clip_grad_value.restype = c_int

        print("✓ Library loaded and types defined")

        # Test 1: Create optimizer
        print("\n--- Creating optimizer ---")
        optimizer = lib.neural_sgd_create(0.01, 0.0)
        if not optimizer:
            print("✗ Failed to create optimizer")
            return False

        print("✓ Optimizer created")
        print(f"  n_params: {optimizer.contents.n_params}")
        print(f"  params: {optimizer.contents.params}")
        print(f"  param_nodes: {optimizer.contents.param_nodes}")

        # Test 2: Call clipping function
        print("\n--- Calling neural_clip_grad_norm ---")
        print("About to call neural_clip_grad_norm...")
        result = lib.neural_clip_grad_norm(optimizer, 1.0)
        print(f"✓ neural_clip_grad_norm returned: {result}")

        # Test 3: Call value clipping
        print("\n--- Calling neural_clip_grad_value ---")
        print("About to call neural_clip_grad_value...")
        result = lib.neural_clip_grad_value(optimizer, -1.0, 1.0)
        print(f"✓ neural_clip_grad_value returned: {result}")

        lib.neural_optimizer_free(optimizer)
        print("\n✅ No segfault occurred!")
        return True

    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clipping_segfault()
    sys.exit(0 if success else 1)