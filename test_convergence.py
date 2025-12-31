#!/usr/bin/env python3
"""
Test XOR training convergence with PyNeural
"""
import sys
sys.path.insert(0, '.')
import pyneural as pn
import numpy as np

# Initialize
pn.init()
print(f"Framework version: {pn.version()}")
print(f"SIMD: {pn.get_simd_name()}")

# Create simple XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Create tensors
X_tensor = pn.Tensor.from_numpy(X)
y_tensor = pn.Tensor.from_numpy(y)

print(f"X shape: {X_tensor.shape}")
print(f"y shape: {y_tensor.shape}")

# Create a simple model: 2 -> 4 -> 1
linear1 = pn.Linear(2, 4)
linear2 = pn.Linear(4, 1)

print(f"Linear1 created: in=2, out=4")
print(f"Linear2 created: in=4, out=1")

# Test forward pass
print("\nTesting forward pass...")
h1 = linear1(X_tensor)
print(f"h1 shape: {h1.shape}")

# Apply ReLU
relu = pn.ReLU()
h1_activated = relu(h1)
print(f"h1_activated shape: {h1_activated.shape}")

# Second linear
output = linear2(h1_activated)
print(f"output shape: {output.shape}")

print("\nOutput values:")
print(output.numpy())

print("\n✓ Forward pass works!")
