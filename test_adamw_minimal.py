#!/usr/bin/env python3
"""
Minimal test to verify AdamW optimizer creation and basic functionality
"""

import ctypes
import os

# Load the neural assembly library
lib_path = os.path.join(os.path.dirname(__file__), 'libneural.so')
lib = ctypes.CDLL(lib_path)

# C optimizer API (optimizers_c.c)
lib.opt_adamw_create.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.opt_adamw_create.restype = ctypes.c_void_p
lib.opt_get_lr.argtypes = [ctypes.c_void_p]
lib.opt_get_lr.restype = ctypes.c_double
lib.opt_free.argtypes = [ctypes.c_void_p]
lib.opt_free.restype = None

def test_adamw_basic():
    """Test basic AdamW optimizer functionality"""
    print("Testing AdamW optimizer...")
    
    # Test parameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01
    
    # Create AdamW optimizer via C backend
    optimizer = lib.opt_adamw_create(lr, beta1, beta2, eps, weight_decay)
    
    if optimizer:
        actual_lr = lib.opt_get_lr(optimizer)
        print("✓ PASS: AdamW optimizer created successfully")
        print(f"  Learning rate: {actual_lr}")
        print(f"  Beta1: {beta1}")
        print(f"  Beta2: {beta2}")
        print(f"  Epsilon: {eps}")
        print(f"  Weight decay: {weight_decay}")
        lib.opt_free(optimizer)
        return True
    else:
        print("✗ FAIL: AdamW optimizer creation failed")
        return False

if __name__ == "__main__":
    success = test_adamw_basic()
    exit(0 if success else 1)
