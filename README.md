# Neural Assembly Framework

A minimal deep learning framework implemented entirely in x86-64 assembly language.

## Overview

This project implements a complete neural network training framework in pure x86-64 assembly, targeting Linux with NASM assembler. It demonstrates that even low-level systems programming can be used for machine learning workloads.

## Features

- **Memory Management** (`mem.asm`)
  - Arena allocator for efficient memory management
  - Aligned memory allocation for SIMD operations
  - Memory pools for tensor storage

- **Tensor Operations** (`tensor.asm`)
  - N-dimensional tensor support (up to 4 dimensions)
  - Tensor creation, reshaping, and views
  - Automatic stride calculation

- **SIMD-Optimized Math Kernels** (`math_kernels.asm`)
  - SSE/AVX vectorized operations
  - Blocked matrix multiplication (SGEMM)
  - SAXPY, dot product, element-wise operations

- **Automatic Differentiation** (`autograd.asm`)
  - Tape-based computational graph
  - Reverse-mode autodiff (backpropagation)
  - Gradient accumulation

- **Neural Network Layers** (`nn_layers.asm`)
  - Linear (fully connected)
  - Conv2D (2D convolution)
  - BatchNorm (batch normalization)
  - Dropout
  - MaxPool2D

- **Activation Functions** (`activations.asm`)
  - ReLU
  - Sigmoid
  - Tanh
  - Softmax

- **Loss Functions** (`losses.asm`)
  - Mean Squared Error (MSE)
  - Cross-Entropy
  - Binary Cross-Entropy

- **Optimizers** (`optimizers.asm`)
  - SGD with momentum
  - Adam

- **Data Loading** (`dataset.asm`)
  - CSV file parsing
  - Mini-batch loading
  - Dataset shuffling (Fisher-Yates)

- **Model I/O** (`model_io.asm`)
  - Binary model serialization
  - Checkpoint save/restore

- **Configuration** (`config_parser.asm`)
  - INI-style configuration parsing
  - Training hyperparameters
  - Model architecture specification

## Requirements

- Linux x86-64
- NASM (Netwide Assembler) 2.14+
- GNU Make
- (Optional) GDB for debugging

## Building

```bash
# Build the framework
make

# Build with debug symbols
make debug

# Clean build artifacts
make clean
```

## Usage

### Training a Model

```bash
./neural_framework train config.ini [output_model.bin]
```

### Running Inference

```bash
./neural_framework infer config.ini model.bin
```

### Running Gradient Checks

```bash
./neural_framework test
```

## Configuration File Format

The framework uses INI-style configuration files:

```ini
[model]
input_size = 784
hidden_size = 128
output_size = 10
activation = relu

[training]
epochs = 100
batch_size = 32
learning_rate = 0.001

[optimizer]
type = adam
beta1 = 0.9
beta2 = 0.999

[data]
train_file = data/train.csv
test_file = data/test.csv
```

See `example_config.ini` for a complete example.

## Architecture

### Memory Layout

The framework uses a custom arena allocator with mmap-based memory allocation:

```
┌─────────────────────────────────────┐
│           Arena Header              │
├─────────────────────────────────────┤
│        Tensor Storage Pool          │
├─────────────────────────────────────┤
│      Computational Graph Nodes      │
├─────────────────────────────────────┤
│         Gradient Storage            │
└─────────────────────────────────────┘
```

### Tensor Structure

```
Offset  Size   Field
0       8      data pointer (32-byte aligned)
8       4      ndim (number of dimensions)
12      16     dims[4] (dimension sizes)
28      16     strides[4]
44      4      requires_grad (boolean)
48      8      grad pointer
56      8      autograd node pointer
```

### Calling Convention

The framework follows the System V AMD64 ABI:
- Arguments: RDI, RSI, RDX, RCX, R8, R9
- Float arguments: XMM0-XMM7
- Return value: RAX (integer), XMM0 (float)
- Preserved registers: RBX, RBP, R12-R15

## SIMD Optimization

Matrix operations use SSE/AVX instructions for vectorization:

- **SGEMM**: 4x4 blocked multiplication with loop unrolling
- **Element-wise ops**: 4-wide SIMD processing
- **Reductions**: Horizontal adds with SSE3 HADDPS

## Example: XOR Network

A minimal example training a network to learn XOR:

```
Input: [0,0] -> Output: 0
Input: [0,1] -> Output: 1
Input: [1,0] -> Output: 1
Input: [1,1] -> Output: 0
```

Configuration:
```ini
[model]
input_size = 2
hidden_size = 4
output_size = 1
activation = sigmoid

[training]
epochs = 1000
batch_size = 4
learning_rate = 0.1

[optimizer]
type = sgd
momentum = 0.9
```

## File Structure

```
Neural Assembly/
├── main.asm            # Entry point and CLI
├── mem.asm             # Memory management
├── utils.asm           # Utility functions
├── tensor.asm          # Tensor operations
├── math_kernels.asm    # SIMD math operations
├── autograd.asm        # Automatic differentiation
├── activations.asm     # Activation functions
├── nn_layers.asm       # Neural network layers
├── losses.asm          # Loss functions
├── optimizers.asm      # Optimization algorithms
├── dataset.asm         # Data loading
├── model_io.asm        # Model serialization
├── config_parser.asm   # Configuration parsing
├── Makefile            # Build system
├── example_config.ini  # Example configuration
└── README.md           # This file
```

## Limitations

- Single-precision (float32) only
- Maximum 4 tensor dimensions
- No GPU support (CPU SIMD only)
- Limited error handling
- No multi-threading (single-core)

## Performance Considerations

- Use aligned memory (32-byte) for SIMD operations
- Batch size should be multiple of 4 for optimal SIMD utilization
- Blocked matrix multiplication reduces cache misses

## Debugging

Build with debug symbols:
```bash
make debug
```

Run with GDB:
```bash
gdb ./neural_framework
(gdb) break main
(gdb) run train config.ini
```

Useful GDB commands for assembly:
```gdb
(gdb) info registers          # View all registers
(gdb) p/f $xmm0              # Print XMM register as float
(gdb) x/4f $rdi              # Examine 4 floats at address
(gdb) disas                   # Disassemble current function
```

## Contributing

This is an educational project demonstrating low-level ML implementation. Contributions for:
- Bug fixes
- Performance optimizations
- Additional layer types
- Better documentation

## License

MIT License - See LICENSE file for details.

## References

- Intel® 64 and IA-32 Architectures Software Developer's Manual
- NASM Documentation
- System V AMD64 ABI Specification
- "Automatic Differentiation in Machine Learning: a Survey" - Baydin et al.
