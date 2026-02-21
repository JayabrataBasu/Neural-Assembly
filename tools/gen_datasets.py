#!/usr/bin/env python3
"""
Generate synthetic datasets for Neural Assembly framework examples.
Creates CSV files for XOR, sine wave, spiral, and other test problems.
"""

import numpy as np
import os

# Ensure output directory exists
os.makedirs('csv', exist_ok=True)

def generate_xor_data():
    """Generate XOR dataset."""
    print("Generating XOR data")

    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    np.savetxt('csv/xor_train.csv', X, delimiter=',', fmt='%.1f')
    np.savetxt('csv/xor_labels.csv', y, delimiter=',', fmt='%.1f')

    print("  Created: csv/xor_train.csv, csv/xor_labels.csv")

def generate_sine_data(n_train=1000, n_val=200):
    """Generate sine wave regression data"""
    print(f"Generating sine wave data: {n_train} training, {n_val} validation samples")
    
    # Training data: x in [-pi, pi]
    np.random.seed(42)
    x_train = np.random.uniform(-np.pi, np.pi, n_train)
    y_train = np.sin(x_train)
    
    # Validation data
    x_val = np.random.uniform(-np.pi, np.pi, n_val)
    y_val = np.sin(x_val)
    
    # Save to CSV
    np.savetxt('csv/sine_train.csv', x_train.reshape(-1, 1), delimiter=',', fmt='%.6f')
    np.savetxt('csv/sine_labels.csv', y_train.reshape(-1, 1), delimiter=',', fmt='%.6f')
    np.savetxt('csv/sine_val.csv', x_val.reshape(-1, 1), delimiter=',', fmt='%.6f')
    np.savetxt('csv/sine_val_labels.csv', y_val.reshape(-1, 1), delimiter=',', fmt='%.6f')
    
    print("  Created: csv/sine_train.csv, csv/sine_labels.csv")
    print("  Created: csv/sine_val.csv, csv/sine_val_labels.csv")

def generate_spiral_data(n_samples=500, noise=0.2):
    """Generate two-class spiral dataset"""
    print(f"Generating spiral data: {n_samples} samples per class")
    
    np.random.seed(1337)
    
    # Class 0: spiral 1
    theta0 = np.linspace(0, 4*np.pi, n_samples) + np.random.randn(n_samples) * noise
    r0 = theta0 / (4*np.pi)
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)
    
    # Class 1: spiral 2 (rotated by pi)
    theta1 = np.linspace(0, 4*np.pi, n_samples) + np.pi + np.random.randn(n_samples) * noise
    r1 = theta1 / (4*np.pi)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)
    
    # Combine
    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    Y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    
    # Shuffle
    idx = np.random.permutation(len(Y))
    X, Y = X[idx], Y[idx]
    
    # Split train/val
    split = int(0.8 * len(Y))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    # Save
    np.savetxt('csv/spiral_train.csv', X_train, delimiter=',', fmt='%.6f')
    np.savetxt('csv/spiral_labels.csv', Y_train.reshape(-1, 1), delimiter=',', fmt='%.0f')
    np.savetxt('csv/spiral_val.csv', X_val, delimiter=',', fmt='%.6f')
    np.savetxt('csv/spiral_val_labels.csv', Y_val.reshape(-1, 1), delimiter=',', fmt='%.0f')
    
    print("  Created: csv/spiral_train.csv, csv/spiral_labels.csv")
    print("  Created: csv/spiral_val.csv, csv/spiral_val_labels.csv")

def generate_deep_network_data(n_train=2000, n_val=500, n_features=10, n_classes=10):
    """Generate multi-class classification data for deep network testing"""
    print(f"Generating deep network data: {n_train} training, {n_val} validation")
    print(f"  Features: {n_features}, Classes: {n_classes}")
    
    np.random.seed(99999)
    
    # Generate class centers
    centers = np.random.randn(n_classes, n_features) * 2
    
    def generate_samples(n):
        X = []
        Y = []
        samples_per_class = n // n_classes
        for c in range(n_classes):
            samples = centers[c] + np.random.randn(samples_per_class, n_features) * 0.5
            X.append(samples)
            Y.extend([c] * samples_per_class)
        X = np.vstack(X)
        Y = np.array(Y)
        # Shuffle
        idx = np.random.permutation(len(Y))
        return X[idx], Y[idx]
    
    X_train, Y_train = generate_samples(n_train)
    X_val, Y_val = generate_samples(n_val)
    
    # Save
    np.savetxt('csv/deep_train.csv', X_train, delimiter=',', fmt='%.6f')
    np.savetxt('csv/deep_labels.csv', Y_train.reshape(-1, 1), delimiter=',', fmt='%.0f')
    np.savetxt('csv/deep_val.csv', X_val, delimiter=',', fmt='%.6f')
    np.savetxt('csv/deep_val_labels.csv', Y_val.reshape(-1, 1), delimiter=',', fmt='%.0f')
    
    print("  Created: csv/deep_train.csv, csv/deep_labels.csv")
    print("  Created: csv/deep_val.csv, csv/deep_val_labels.csv")

def generate_mnist_subset(n_train=1000, n_test=200):
    """Generate a small synthetic MNIST-like dataset for testing"""
    print(f"Generating synthetic MNIST-like data: {n_train} training, {n_test} test")
    print("  (For real MNIST, download from http://yann.lecun.com/exdb/mnist/)")
    
    np.random.seed(12345)
    
    # Generate simple digit-like patterns (random, not real digits)
    def make_synthetic_digit(digit, n_samples):
        """Create synthetic 28x28 images that encode digit patterns"""
        images = []
        for _ in range(n_samples):
            img = np.zeros((28, 28))
            # Create a simple pattern based on digit
            center_x, center_y = 14, 14
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    # Different patterns for different digits
                    if digit == 0:
                        # Circle
                        if 8 < dist < 12:
                            img[i, j] = 1.0
                    elif digit == 1:
                        # Vertical line
                        if 12 < j < 16 and 5 < i < 23:
                            img[i, j] = 1.0
                    elif digit == 2:
                        # Top half circle + diagonal + bottom line
                        if (i < 14 and 8 < dist < 12 and i < 14) or (13 < i < 22 and abs(i - j) < 3):
                            img[i, j] = 1.0
                    else:
                        # Random pattern for other digits
                        if np.random.rand() < 0.1 * (1 + np.cos(2 * np.pi * digit * dist / 28)):
                            img[i, j] = 1.0
            
            # Add noise
            img += np.random.randn(28, 28) * 0.1
            img = np.clip(img, 0, 1)
            images.append(img.flatten())
        return np.array(images)
    
    # Generate data
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    
    train_per_class = n_train // 10
    test_per_class = n_test // 10
    
    for digit in range(10):
        X_train.append(make_synthetic_digit(digit, train_per_class))
        Y_train.extend([digit] * train_per_class)
        X_test.append(make_synthetic_digit(digit, test_per_class))
        Y_test.extend([digit] * test_per_class)
    
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    # Shuffle
    idx_train = np.random.permutation(len(Y_train))
    idx_test = np.random.permutation(len(Y_test))
    X_train, Y_train = X_train[idx_train], Y_train[idx_train]
    X_test, Y_test = X_test[idx_test], Y_test[idx_test]
    
    # Save
    np.savetxt('csv/mnist_train.csv', X_train, delimiter=',', fmt='%.6f')
    np.savetxt('csv/mnist_train_labels.csv', Y_train.reshape(-1, 1), delimiter=',', fmt='%.0f')
    np.savetxt('csv/mnist_test.csv', X_test, delimiter=',', fmt='%.6f')
    np.savetxt('csv/mnist_test_labels.csv', Y_test.reshape(-1, 1), delimiter=',', fmt='%.0f')
    
    print("  Created: csv/mnist_train.csv, csv/mnist_train_labels.csv")
    print("  Created: csv/mnist_test.csv, csv/mnist_test_labels.csv")
    print("  Note: This is synthetic data. For real MNIST, use proper dataset loaders.")

if __name__ == '__main__':
    print("=" * 60)
    print("Neural Assembly Framework - Dataset Generator")
    print("=" * 60)
    print()
    
    generate_xor_data()
    print()
    
    generate_sine_data()
    print()
    
    generate_spiral_data()
    print()
    
    generate_deep_network_data()
    print()
    
    generate_mnist_subset()
    print()
    
    print("=" * 60)
    print("All datasets generated successfully!")
    print("=" * 60)
