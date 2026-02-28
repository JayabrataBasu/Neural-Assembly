# Neural Assembly Framework — v2.0.0

A deep learning framework with its core written in x86-64 assembly and C, plus thin Python bindings.

## Overview

This project implements a complete neural network training framework in x86-64 assembly and C, targeting Linux with NASM assembler and GCC. The compute-heavy pieces (SIMD matmul, tensor ops, convolution, pooling, normalization, fuzzy logic) live in assembly and C; Python just marshals arguments and manages lifetimes. It demonstrates that even low-level systems programming can power real ML workloads.

## Features

### Core (Assembly)

- **Memory Management** (`mem.asm`)
  - Arena allocator for efficient memory management
  - Aligned memory allocation for SIMD operations
  - Memory pools for tensor storage

- **Tensor Operations** (`tensor.asm`)
  - N-dimensional tensor support (up to 4 dimensions)
  - Tensor creation, reshaping, and views
  - Automatic stride calculation

- **SIMD-Optimized Math Kernels** (`math_kernels.asm`, `simd.asm`)
  - Runtime CPU feature detection (SSE, AVX, AVX-512)
  - AVX-512 vectorized operations (processes 16 floats/iteration)
  - AVX2/AVX fallback (8 floats/iteration)
  - SSE fallback (4 floats/iteration)
  - Blocked matrix multiplication (SGEMM)
  - FMA (Fused Multiply-Add) support

- **Automatic Differentiation** (`autograd.asm`)
  - Tape-based computational graph
  - Reverse-mode autodiff (backpropagation)
  - Gradient accumulation

- **Neural Network Layers** (`nn_layers.asm`)
  - Linear (fully connected)
  - Dropout (inverted, with mask)

- **Activation Functions** (`activations.asm`)
  - ReLU, Sigmoid, Tanh, Softmax

- **Loss Functions** (`losses.asm`)
  - Mean Squared Error (MSE)
  - Cross-Entropy
  - Binary Cross-Entropy

- **Optimizers** (`optimizers.asm`)
  - SGD with momentum
  - Adam / AdamW
  - Runtime learning rate get/set
  - Optimizer state save/load

- **Training Ops** (`training_ops.asm`)
  - Confusion matrix update & per-class precision/recall/F1
  - LR schedules: step decay, exponential decay, cosine annealing
  - NaN/Inf detection on float32 tensors (parity-flag trick)
  - Gradient L2 norm (SSE-vectorized sum-of-squares + sqrtss)
  - Dropout forward/backward (inverted, with mask)
  - Weight initialization: He/Kaiming, Xavier/Glorot uniform & normal

- **Data Loading** (`dataset.asm`) — CSV parsing, mini-batch loading, Fisher-Yates shuffling
- **Model I/O** (`model_io.asm`) — Binary serialization, checkpoint save/restore, optimizer state persistence
- **Configuration** (`config_parser.asm`) — INI-style parsing for hyperparameters and architecture
- **Error Handling** (`error.asm`) — 19 error codes, tensor validation, NaN detection, null-pointer checks
- **Testing** (`tests.asm`) — Unit tests for all components, numerical gradient checking
- **Multi-threading** (`threads.asm`) — pthreads pool, work queue, parallel for, CPU core detection

### C Modules

- **Conv2D & MaxPool2D** (`conv2d.c`) — *NEW in v2.0*
  - Im2col + GEMM convolution (forward + backward)
  - Kaiming uniform weight initialization
  - MaxPool2D with argmax mask for backward routing
  - NCHW layout, float64, full gradient computation (input, weight, bias)

- **BatchNorm1d & LayerNorm** (`batchnorm.c`)
  - BatchNorm1d: running-mean/var EMA, train vs eval modes, full backward
  - LayerNorm: per-sample normalization, no running stats

- **Label-Smoothed CE & ROC-AUC** (`metrics_losses.c`)
  - Numerically stable log-softmax + smoothed targets
  - ROC-AUC via trapezoidal integration on sorted FPR/TPR curve

- **Data Transforms** (`transforms.c`)
  - Per-feature z-score normalization (forward + inverse)
  - Min-max scaling to [0, 1]
  - Statistics computation (mean, std, min, max) in one pass

- **Embedding Layer** (`embedding.c`)
  - Row-lookup forward, gradient-accumulation backward
  - Xavier-uniform init, index-range validation

- **Fuzzy Logic Inference Engine** (`fuzzy.c`)
  - Membership functions: triangular, trapezoidal, gaussian
  - Operators: AND (min), OR (max), NOT (complement)
  - Defuzzification: centroid, bisector, mean-of-maximum
  - Mamdani rule engine (up to 256 rules, 16 antecedents)

- **TensorBoard Logging** (`tb_logger.c`)
  - TFRecord/TFEvent file writer (CRC32C, protobuf encoding)
  - Scalar, multi-scalar, and histogram-stats logging

- **Model Pruning** (`pruning.c`)
  - Unstructured magnitude pruning (threshold & top-k)
  - Structured row/column pruning (L2 norm)
  - Sparsity analysis, mask reapply, threshold search

- **INT8 Quantization** (`quantize.c`)
  - Symmetric & affine (asymmetric) quantization
  - MinMax and percentile calibration
  - Quantized int8 matrix multiply (int32 accumulation)

### Python Layer (`pyneural/`)

Thin wrappers that marshal arguments to C/assembly — all heavy math stays below.

- **Layers**: Linear, Conv2D, MaxPool2D, BatchNorm1d, LayerNorm, Embedding, Dropout
- **Activations**: ReLU, Sigmoid, Tanh, Softmax
- **Losses**: MSELoss, CrossEntropyLoss, LabelSmoothingCrossEntropy
- **Optimizers**: SGD, Adam, AdamW (with gradient clipping)
- **Schedulers**: OneCycleLR (warmup + cosine annealing), LRFinder (exponential sweep)
- **Checkpoint**: save/load full model state (epoch, loss, LR, all parameters)
- **Metrics**: ROC-AUC score, confusion matrix, precision/recall/F1
- **Transforms**: Normalize (z-score), MinMaxScale, Compose pipelines
- **Fuzzy Logic**: FuzzySystem class + standalone membership / defuzzification helpers
- **Utilities**: TensorBoard SummaryWriter, Pruner, Quantizer, NaNDetector, EarlyStopping

## Requirements

- Linux x86-64
- NASM (Netwide Assembler) 2.14+
- GNU Make
- (Optional) GDB for debugging

## Building

```bash
# Build the framework
make

# Build shared library
make lib

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

## Validation Suite (Recommended)

Run the staged validation suite to check build, dataset generation, assembly tests,
Python integration tests, training viability across multiple configs, and API
compatibility checks.

```bash
# Optional but recommended: isolated Python env for numpy-based checks
python3 -m venv .neuasm
./.neuasm/bin/pip install numpy

# Smoke validation (fast)
make validate-smoke

# Regression validation (comprehensive)
make validate
```

The validation runner is implemented in `tools/run_validation_suite.py` and supports:

```bash
python3 tools/run_validation_suite.py --tier smoke
python3 tools/run_validation_suite.py --tier regression
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
lr_step_size = 10       # Decay LR every 10 epochs (0 to disable)
lr_gamma = 0.5          # Multiply LR by 0.5 at each step

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

```c
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

```asm
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

Matrix operations use runtime-detected SIMD instructions for vectorization:

- **AVX-512**: 16-wide SIMD processing (512-bit zmm registers)
- **AVX2/AVX**: 8-wide SIMD processing (256-bit ymm registers)
- **SSE**: 4-wide SIMD processing (128-bit xmm registers)
- **SGEMM**: 4x4 blocked multiplication with loop unrolling
- **FMA**: Fused multiply-add for improved accuracy and speed
- **Reductions**: Horizontal adds with SSE3 HADDPS

CPU features are detected at runtime:

```bash
./neural_framework test  # Shows detected SIMD level
```

## Example Configurations

Example configurations are provided in the `configs/` directory:

| Config | Description | Architecture |
| -------- | ------------- | -------------- |
| `xor_config.ini` | XOR learning | 2→8→1 |
| `mnist_config.ini` | MNIST digits | 784→256→128→10 |
| `sine_config.ini` | Sine regression | 1→32→32→1 |
| `binary_class_config.ini` | Spiral classification | 2→16→16→1 |
| `deep_network_config.ini` | Deep network | 10→64→64→64→64→10 |
| `wine_quality_config.ini` | UCI Wine Quality (real-world) | 11→64→32→6 |

Generate synthetic datasets:

```bash
python3 tools/gen_datasets.py
```

## Real-World Benchmark: UCI Wine Quality

The framework includes a real-world benchmark using the [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality) dataset — 1,599 red wine samples with 11 physicochemical features predicting quality scores across 6 classes.

### Setup & Training

```bash
# Download and prepare the dataset (requires internet)
python3 tools/prepare_wine_quality.py

# Train (11→64→32→6, Adam, cross-entropy, 80 epochs)
./neural_framework train configs/wine_quality_config.ini /tmp/na_wine.bin
```

### Results

| Metric | Value |
| -------- | ------- |
| Parameters | 3,046 |
| Final Train Loss | 0.94 |
| Train Accuracy | ~60% |
| Val Accuracy | ~65–68% |

This dataset provides a meaningful, non-trivial benchmark: the model learns real patterns (loss decreasing from 1.51 to 0.94) but doesn't achieve trivial 100% accuracy due to genuine class overlap in the data. Comparable to typical ML baselines for this dataset (50–65% accuracy with simple models).

### Dataset Details

- **Source**: UCI Machine Learning Repository (red wine subset)
- **Features**: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol
- **Classes**: 6 quality levels (scores 3–8 remapped to 0–5)
- **Preprocessing**: min-max normalization, stratified 80/20 split
- **Class distribution**: imbalanced (42.6% class 5, 39.9% class 6, 12.4% class 7, etc.)

## Example: XOR Network

A minimal example training a network to learn XOR:

```asm
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

```asm
Neural Assembly/
├── main.asm            # Entry point and CLI
├── mem.asm             # Memory management (arena allocator)
├── utils.asm           # Utility functions
├── error.asm           # Error handling system
├── simd.asm            # SIMD detection and optimized kernels
├── tensor.asm          # Tensor operations
├── math_kernels.asm    # SIMD math operations
├── autograd.asm        # Automatic differentiation
├── activations.asm     # Activation functions
├── nn_layers.asm       # Neural network layers (Linear, Dropout)
├── losses.asm          # Loss functions
├── optimizers.asm      # Optimization algorithms
├── dataset.asm         # Data loading
├── model_io.asm        # Model serialization
├── config_parser.asm   # Configuration parsing
├── threads.asm         # Multi-threading support
├── training_ops.asm    # LR schedules, metrics, weight init
├── tests.asm           # Unit tests and gradient checking
├── compat.asm          # C library compatibility
├── verify.asm          # Symbol verification
│
├── conv2d.c            # Conv2D + MaxPool2D (im2col/GEMM)
├── batchnorm.c         # BatchNorm1d + LayerNorm
├── metrics_losses.c    # Label-smoothed CE + ROC-AUC
├── transforms.c        # Data normalization / scaling
├── embedding.c         # Embedding layer (lookup + backward)
├── fuzzy.c             # Fuzzy logic inference engine
├── tb_logger.c         # TensorBoard TFEvent logging
├── pruning.c           # Model pruning (magnitude, structured)
├── quantize.c          # INT8 quantization
│
├── neural_api.h        # Public C header for all exports
├── Makefile            # Build system (asm + C → libneural.so)
│
├── pyneural/           # Python wrappers (thin ctypes layer)
│   ├── __init__.py     # Public API + version
│   ├── core.py         # ctypes bindings to libneural.so
│   ├── tensor.py       # Tensor class
│   ├── nn.py           # Linear, activations, losses, Embedding, norms
│   ├── conv.py         # Conv2D, MaxPool2D
│   ├── optim.py        # SGD, Adam, AdamW
│   ├── autograd.py     # Autograd wrapper
│   ├── dataset.py      # DataLoader
│   ├── config.py       # Config parser
│   ├── schedulers.py   # OneCycleLR, LRFinder
│   ├── fuzzy.py        # FuzzySystem + helpers
│   ├── transforms.py   # Normalize, MinMaxScale, Compose
│   ├── checkpoint.py   # Checkpoint save/load
│   ├── metrics.py      # ROC-AUC
│   ├── tb_logger.py    # SummaryWriter
│   ├── pruning.py      # Pruner
│   └── quantize.py     # Quantizer
│
├── tools/              # Test suites, data generators, utilities
│   ├── run_validation_suite.py
│   ├── test_conv2d.py        # 83 tests
│   ├── test_fuzzy.py         # 65 tests
│   ├── test_lr_finder.py     # 39 tests
│   ├── test_transforms.py    # 37 tests
│   ├── test_onecycle.py      # 35 tests
│   ├── test_train_eval_mode.py # 26 tests
│   ├── test_batchnorm.py     # 22 tests
│   ├── test_embedding.py     # 19 tests
│   ├── test_label_smoothing.py # 17 tests
│   ├── test_layernorm.py     # 16 tests
│   ├── test_roc_auc.py       # 16 tests
│   ├── test_checkpoint.py    # 14 tests
│   ├── test_dropout.py       # 13 tests
│   ├── test_autograd.py      #  — autograd integration
│   ├── test_dataloader.py    #  — dataloader integration
│   ├── gen_synth.py          # Synthetic data generator
│   └── gen_datasets.py       # Dataset generator for examples
│
├── configs/            # Example configurations
│   ├── xor_config.ini
│   ├── mnist_config.ini
│   ├── sine_config.ini
│   └── ...
├── csv/                # Training data
└── README.md           # This file
```

## Limitations

- Assembly core uses single-precision (float32); C modules use double (float64)
- Maximum 4 tensor dimensions
- No GPU support (CPU SIMD only)

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

## Feature Roadmap — All Complete as of v2.0.0

Every feature below has a dedicated test suite and passes the full 23-check validation.

### Core Infrastructure (v1.0)

- ✅ **Confusion Matrix & Per-Class Metrics** — Precision, recall, F1 per class (assembly)
- ✅ **Early Stopping** — Patience-based validation loss monitoring
- ✅ **LR Scheduling** — Step decay, exponential decay, cosine annealing, warmup, ReduceLROnPlateau
- ✅ **NaN/Inf Detection** — Assembly `tensor_has_nan`/`tensor_has_inf` with parity-flag trick
- ✅ **Gradient Norm Logging** — SSE-vectorized L2 norm
- ✅ **Weight Initialization** — He/Kaiming & Xavier/Glorot, uniform & normal
- ✅ **Dropout Regularization** — Inverted dropout forward/backward (assembly)
- ✅ **Class-Balanced Sampling** — `WeightedRandomSampler` backed by assembly

### Extended Modules (v1.1–v1.2)

- ✅ **TensorBoard Logging** — C TFEvent writer + Python `SummaryWriter`
- ✅ **Model Pruning** — Magnitude, top-k, structured row/column (C + Python)
- ✅ **INT8 Quantization** — Symmetric/affine, calibration, quantized matmul (C + Python)
- ✅ **Dropout Wrapper** — Python `Dropout` class with train/eval mode (13 tests)
- ✅ **Gradient Clipping** — Assembly-backed `clip_grad_norm` wired into Trainer (6 tests)
- ✅ **Train/Eval Mode** — Module base class propagating to children (26 tests)
- ✅ **Checkpoint Save/Resume** — Binary format v1 with metadata (14 tests)

### Normalization & Losses (v1.2)

- ✅ **BatchNorm1d** — Running-mean EMA, train/eval modes, full backward (C, 22 tests)
- ✅ **LayerNorm** — Per-sample normalization, no running stats (C, 16 tests)
- ✅ **Label Smoothing** — Numerically stable log-softmax + smoothed targets (C, 17 tests)
- ✅ **ROC-AUC** — Trapezoidal integration on sorted FPR/TPR curve (C, 16 tests)

### Data & Embeddings (v1.2)

- ✅ **Data Transforms** — Z-score, min-max, Compose pipeline (C, 37 tests)
- ✅ **Embedding Layer** — Row lookup + gradient accumulation backward (C, 19 tests)

### Schedulers & Diagnostics (v1.2)

- ✅ **OneCycleLR** — Linear warmup + cosine annealing, per-step (35 tests)
- ✅ **LRFinder** — Exponential sweep, EMA smoothing, auto-suggestion (39 tests)

### Fuzzy Logic (v1.2)

- ✅ **Fuzzy Inference Engine** — Mamdani rule engine, 3 MF types, 3 defuzz methods (C, 65 tests)

### Convolution & Pooling (v2.0)

- ✅ **Conv2D** — Im2col + GEMM forward/backward, Kaiming init, bias support (C, 83 tests combined)
- ✅ **MaxPool2D** — Argmax-mask forward + scatter backward (C)

**Total test count: 408 across 14 dedicated test suites, plus validation and integration tests.**

## License

MIT License - See LICENSE file for details.

## References

- Intel® 64 and IA-32 Architectures Software Developer's Manual
- NASM Documentation
- System V AMD64 ABI Specification
- "Automatic Differentiation in Machine Learning: a Survey" - Baydin et al.
