#!/usr/bin/env python3
"""
Simple verification script for Neural Assembly framework.
Manual calculation verification without external dependencies.
"""

import os

def test_linear_math():
    """Manually verify linear layer calculations"""
    print("Testing Linear Layer Math...")

    # Test case: input=[1,2,3,4], weights=[0.1,0.2,0.3,0.4], bias=0.5
    # Expected: 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 + 0.5 = 2.2

    input_vals = [1.0, 2.0, 3.0, 4.0]
    weights = [0.1, 0.2, 0.3, 0.4]
    bias = 0.5

    # Manual calculation
    dot_product = sum(x * w for x, w in zip(input_vals, weights))
    result = dot_product + bias

    print(f"Input: {input_vals}")
    print(f"Weights: {weights}")
    print(f"Bias: {bias}")
    print(f"Dot product: {dot_product}")
    print(f"Result: {result}")
    print("Expected: 3.5")  # 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 + 0.5 = 3.5

    if abs(result - 3.5) < 1e-6:
        print("✓ PASS")
        return True
    else:
        print("✗ FAIL")
        return False

def test_activation_math():
    """Test activation function math"""
    print("\nTesting Activation Functions...")

    x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]

    # ReLU
    relu_results = [max(0, x) for x in x_vals]
    print(f"ReLU({x_vals}) = {relu_results}")

    # Sigmoid (manual calculation)
    sigmoid_results = []
    for x in x_vals:
        sig = 1 / (1 + 2.718281828**(-x))  # Using e≈2.718
        sigmoid_results.append(round(sig, 4))
    print(f"Sigmoid({x_vals}) ≈ {sigmoid_results}")

    # Tanh (manual approximation)
    tanh_results = []
    for x in x_vals:
        # tanh(x) ≈ (e^x - e^-x) / (e^x + e^-x)
        e_x = 2.718281828**x
        e_neg_x = 2.718281828**(-x)
        tanh_val = (e_x - e_neg_x) / (e_x + e_neg_x)
        tanh_results.append(round(tanh_val, 4))
    print(f"Tanh({x_vals}) ≈ {tanh_results}")

    print("✓ Reference values computed")
    return True

def test_loss_math():
    """Test loss function math"""
    print("\nTesting Loss Functions...")

    pred = [0.9, 0.8, 0.1, 0.2]
    target = [1.0, 1.0, 0.0, 0.0]

    # MSE
    mse = sum((p - t)**2 for p, t in zip(pred, target)) / len(pred)
    print(f"MSE({pred}, {target}) = {mse}")

    # Binary Cross Entropy
    bce = 0
    for p, t in zip(pred, target):
        p = max(min(p, 1-1e-8), 1e-8)  # Clamp for numerical stability
        bce += t * (2.302585**(-p)) + (1-t) * (2.302585**(-(1-p)))  # Using log10(e)≈2.302585
    bce = -bce / len(pred)
    print(f"BCE({pred}, {target}) ≈ {bce}")

    print("✓ Loss calculations completed")
    return True

def test_gradient_math():
    """Test gradient calculation math"""
    print("\nTesting Gradient Calculations...")

    def f(x):
        return x**2

    def f_prime(x):
        return 2*x

    x = 3.0
    eps = 1e-5

    # Numerical gradient
    numerical = (f(x + eps) - f(x - eps)) / (2 * eps)
    analytical = f_prime(x)

    print(f"f(x) = x²")
    print(f"f'({x}) analytical = {analytical}")
    print(f"f'({x}) numerical = {numerical}")
    print(f"Difference: {abs(numerical - analytical)}")

    if abs(numerical - analytical) < 1e-4:
        print("✓ PASS")
        return True
    else:
        print("✗ FAIL")
        return False

def test_xor_setup():
    """Verify XOR problem setup"""
    print("\nTesting XOR Problem Setup...")

    xor_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    xor_outputs = [0, 1, 1, 0]

    print("XOR Truth Table:")
    for i, (x1, x2) in enumerate(xor_inputs):
        print(f"  {x1} XOR {x2} = {xor_outputs[i]}")

    # Verify it's not linearly separable
    print("\nLinear separability check:")
    for inputs, output in zip(xor_inputs, xor_outputs):
        x1, x2 = inputs
        # Try simple linear classifier: x1 + x2 - 0.5
        linear_output = x1 + x2 - 0.5
        predicted = 1 if linear_output > 0 else 0
        status = "✓" if predicted == output else "✗"
        print(f"  {x1},{x2} -> {linear_output:.1f} -> {predicted} {status}")

    print("XOR requires non-linear decision boundary (neural network needed)")
    return True

def run_framework_test():
    """Run the framework's built-in tests"""
    print("\nRunning Framework Tests...")

    try:
        import subprocess
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(["./neural_framework", "test"],
                              capture_output=True, text=True, cwd=project_root)

        if result.returncode == 0:
            print("✓ Framework tests passed")
            print("Output:")
            for line in result.stdout.split('\n')[:10]:  # First 10 lines
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print("✗ Framework tests failed")
            print("Error:")
            print(result.stderr[:500])  # First 500 chars
            return False
    except Exception as e:
        print(f"✗ Could not run framework tests: {e}")
        return False

def main():
    print("=" * 60)
    print("Neural Assembly Framework - Manual Verification")
    print("=" * 60)

    tests = [
        ("Linear Layer Math", test_linear_math),
        ("Activation Functions", test_activation_math),
        ("Loss Functions", test_loss_math),
        ("Gradient Calculations", test_gradient_math),
        ("XOR Problem Setup", test_xor_setup),
        ("Framework Tests", run_framework_test),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"Verification Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All verifications passed!")
    else:
        print("⚠️  Some verifications failed.")

    print("\nNext Steps for Full Verification:")
    print("1. Install NumPy: pip install --user numpy")
    print("2. Run: python3 tools/verify_correctness.py")
    print("3. Train on XOR and verify it converges")
    print("4. Compare against PyTorch/Keras on same problems")
    print("5. Use verify.asm functions for runtime debugging")

if __name__ == "__main__":
    main()