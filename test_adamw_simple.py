#!/usr/bin/env python3
"""
Simple test to verify AdamW optimizer creation
"""

import ctypes
import os

# Load the neural assembly library
lib_path = os.path.join(os.path.dirname(__file__), 'libneural.so')
lib = ctypes.CDLL(lib_path)

# C optimizer API (optimizers_c.c)
lib.opt_adamw_create.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.opt_adamw_create.restype = ctypes.c_void_p
lib.opt_free.argtypes = [ctypes.c_void_p]
lib.opt_free.restype = None

def test_adamw_creation():
    """Test that AdamW optimizer can be created"""
    print("Testing AdamW optimizer creation...")
    
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01
    
    # Create AdamW optimizer via the C backend
    optimizer = lib.opt_adamw_create(lr, beta1, beta2, eps, weight_decay)
    
    if optimizer:
        print("✓ PASS: AdamW optimizer created successfully")
        lib.opt_free(optimizer)
        return True
    else:
        print("✗ FAIL: AdamW optimizer creation failed")
        return False

if __name__ == "__main__":
    success = test_adamw_creation()
    exit(0 if success else 1)
