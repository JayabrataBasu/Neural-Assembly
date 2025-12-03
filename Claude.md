# Level 10 Spec: Minimal Deep Learning Framework in x86-64 Assembly (Linux, NASM)

Target:

* Environment: Linux x86-64, System V ABI, NASM syntax, ELF64.
* Language: Pure assembly except for system libraries (libc/libm) where explicitly allowed.
* Goal: Train and run small feedforward and convolutional networks with autograd.

You are implementing the following subsystems. Do not deviate.

---

1. Global Conventions

---

1.1 ABI and Calling

* Use System V x86-64 calling convention:

  * Arguments: RDI, RSI, RDX, RCX, R8, R9, then stack.
  * Return: RAX (integer/pointer), XMM0 (float/double).
  * Callee-saved: RBX, RBP, R12–R15.
* All functions must:

  * Preserve callee-saved registers.
  * Align stack to 16 bytes at call sites.

1.2 File and Module Structure
Create separate NASM files for:

* `main.asm`                – entry, CLI handler.
* `mem.asm`                 – memory allocator/wrappers.
* `tensor.asm`              – tensor types and operations.
* `math_kernels.asm`        – core BLAS-like kernels.
* `autograd.asm`            – graph and backprop engine.
* `nn_layers.asm`           – layer definitions (Linear, Conv2d, etc.).
* `activations.asm`         – ReLU, Sigmoid, Tanh, Softmax.
* `losses.asm`              – MSE, Cross-Entropy.
* `optimizers.asm`          – SGD, Adam.
* `dataset.asm`             – dataset loading, batching.
* `model_io.asm`            – model save/load.
* `config_parser.asm`       – config/DSL parser.
* `utils.asm`               – logging, error handling, assertions, string utils.
* `tests.asm`               – numeric gradient checks, sanity tests.

Provide a Makefile:

* Builds `train` and `infer` binaries.
* Links against `libc` and `libm` where needed (`exp`, `log`, `pow`).

---

2. Memory and Core Runtime (mem.asm, utils.asm)

---

2.1 Memory Management
Implement:

* `mem_alloc(size: uint64) -> void*`

  * Wrapper over `mmap` or `malloc` (via libc). Choose one and be consistent.
* `mem_free(ptr: void*)`

  * Wrapper over `munmap` or `free`.
* `mem_alloc_aligned(size, alignment) -> void*`

  * For tensors requiring alignment (e.g., 32-byte for AVX).

2.2 Utility Routines
Implement:

* `panic(const char* msg)` – prints message to stderr, exits with non-zero.
* `log_info(const char* msg)` – prints message to stdout.
* `assert(cond, const char* msg)` – if cond == 0, call `panic`.
* `str_cmp`, `str_len`, `str_copy`, `str_to_int`, `str_to_float` – minimal string utilities.

All logging and error handling must go through these functions.

---

3. Tensor Subsystem (tensor.asm)

---

3.1 Dtypes
Support at least:

* `DT_FLOAT32 = 0`
* `DT_FLOAT64 = 1`

3.2 Tensor Descriptor
Define a struct layout in memory (C-style, contiguous):

```c
struct Tensor {
    void* data;        // pointer to raw data
    uint64_t ndim;     // number of dimensions
    uint64_t* shape;   // pointer to array[ndim] of uint64
    uint64_t* stride;  // pointer to array[ndim] of uint64, in element units
    uint32_t dtype;    // one of DT_*
    uint32_t flags;    // bitfield: OWN_DATA (1), OWN_META (2)
};
```

Implement functions:

* `tensor_create(ndim, shape*, dtype) -> Tensor*`

  * Allocate descriptor, shape, stride, and data.
  * Compute default row-major strides.
* `tensor_free(Tensor*)`

  * Free data, shape, stride according to flags.
* `tensor_zeros(ndim, shape*, dtype) -> Tensor*`
* `tensor_copy(src: Tensor*) -> Tensor*`
* `tensor_fill(Tensor*, value_as_double)`
* `tensor_numel(Tensor*) -> uint64_t`
* `tensor_reshape(Tensor*, ndim_new, shape_new*) -> Tensor*`

  * Must check compatibility; panic on mismatch.
* `tensor_view(Tensor*, ndim_new, shape*, stride*) -> Tensor*`

  * Shares data, does not free original.

Limitations:

* No broadcasting in this version. All elementwise ops require equal shapes or explicitly documented behavior.

---

4. Math Kernels (math_kernels.asm)

---

4.1 Elementwise Operations
Implement for float32 and float64:

* `ew_add(out, a, b)`
* `ew_sub(out, a, b)`
* `ew_mul(out, a, b)`
* `ew_div(out, a, b)`
* `ew_scalar_add(out, a, scalar)`
* `ew_scalar_mul(out, a, scalar)`
* `ew_max(out, a, b)` (for ReLU/min-max ops).

Requirements:

* All tensors must be same shape, same dtype.
* Use vectorization:

  * For float32: AVX or SSE, processing multiple elements per iteration.
  * For float64: SSE2 or AVX.

4.2 Reductions
Implement:

* `reduce_sum(tensor, axis, out_tensor)` – sum over specified axis.
* `reduce_mean(tensor, axis, out_tensor)` – reuse sum + scaling.

4.3 Matrix Multiply
Implement:

* `matmul(out, A, B)`

  * A: (M x K), B: (K x N), out: (M x N).
  * Support float32 and float64.
  * Memory layout: row-major.
  * Use blocking and SIMD (AVX/SSE) for reasonable performance.

Constraints:

* Inputs must be 2D tensors.
* No implicit transpose; define explicit `tensor_transpose_2d`.

---

5. Autograd Engine (autograd.asm)

---

5.1 Graph Node Structure
Define:

```c
struct Node {
    Tensor* value;         // forward value
    Tensor* grad;          // gradient w.r.t. this node
    void (*backward_fn)(struct Node* self); // fn to compute grads for parents
    uint32_t n_parents;
    struct Node** parents; // array of parent Node*
    uint32_t visited;      // for topological sorting
};
```

5.2 Node Creation
Implement:

* `node_from_tensor(Tensor*, requires_grad) -> Node*`
* For each operation (e.g., add, mul, matmul, relu, linear, conv2d), define:

  * `op_forward(...) -> Node*`:

    * Allocates output tensor.
    * Allocates Node for output.
    * Sets `parents`, `backward_fn`.

5.3 Backward Execution
Implement:

* `zero_grad(Node* root)` – recursively zero all grads in reachable nodes.
* `backward(Node* loss_node)`

  * Initialize `loss_node->grad` to scalar 1 tensor.
  * Perform reverse topological sort over graph reachable from loss_node.
  * For each node in reverse order, call `backward_fn`.

5.4 Backward Functions
For each supported op, implement a backward function:

* `add_backward`
* `sub_backward`
* `mul_backward`
* `matmul_backward`
* `relu_backward`
* `sigmoid_backward`
* `tanh_backward`
* `softmax_backward`
* `linear_backward`
* `conv2d_backward`
* `mse_loss_backward`
* `cross_entropy_loss_backward`

Each backward must:

* Read `self->grad` (dL/dOutput).
* Compute gradients for each parent’s `grad` (accumulate if non-null).

Grad tensor allocation:

* Ensure `grad` for each leaf/parameter is allocated once, then only updated.

---

6. Layers and Activations (nn_layers.asm, activations.asm)

---

6.1 Parameterized Modules
Define a generic module struct:

```c
struct Module {
    uint32_t n_params;
    Tensor** params;   // array of parameter tensors
    Node** param_nodes; // Node* wrappers for params
    void (*forward)(struct Module*, Node* input, Node** output);
};
```

Implement modules:

* `Linear(in_features, out_features)`

  * Parameters: weight (out_features x in_features), bias (out_features).
  * Forward: y = xW^T + b (with broadcast on batch).
* `Conv2d(in_channels, out_channels, kernel_h, kernel_w, stride, padding)`

  * Parameters: weight, bias.
  * Implement naive loop-based conv in forward; matching backward.

6.2 Activations
Implement autograd-capable:

* `relu(Node* x) -> Node*`
* `sigmoid(Node* x) -> Node*` (use `exp` from libm).
* `tanh(Node* x) -> Node*`
* `softmax(Node* x, axis) -> Node*`

Implement forward as Node-producing ops, not plain tensor functions.

---

7. Loss Functions (losses.asm)

---

Implement:

* `mse_loss(pred: Node*, target: Node*) -> Node*`

  * Output: scalar Node.
* `cross_entropy_loss(logits: Node*, target_indices: Tensor*) -> Node*`

  * Softmax internally or expect logits and apply log-softmax.

Widely used reduction: mean over batch.

---

8. Optimizers (optimizers.asm)

---

8.1 Optimizer Interface
Define:

```c
struct Optimizer {
    uint32_t n_params;
    Tensor** params;     // tensors to update
    Tensor** grads;      // corresponding grad tensors
    void (*step)(struct Optimizer*);
    void (*zero_grad)(struct Optimizer*);
    void* state;         // optimizer-specific state
};
```

8.2 Implement SGD

* Hyperparameters:

  * `lr` (learning rate, float64)
  * `momentum` (optional, float64; 0.0 to disable)
* State:

  * For each param: velocity tensor (same shape/dtype).
* `step`:

  * For each param:

    * v = momentum * v + grad
    * param -= lr * v

8.3 Implement Adam

* Hyperparameters:

  * `lr`, `beta1`, `beta2`, `eps`.
* State:

  * m (first moment) tensor per param.
  * v (second moment) tensor per param.
  * t (time step, uint64_t).
* `step`:

  * Standard Adam update, per element.

---

9. Dataset Loading and Batching (dataset.asm)

---

9.1 Dataset Representation

```c
struct Dataset {
    uint64_t n_samples;
    Tensor* data;    // features
    Tensor* labels;  // labels (indices or one-hot)
};
```

9.2 Loading
Implement:

* `dataset_load_csv(const char* data_path, const char* label_path, ...) -> Dataset*`

  * Simple CSV reader using `fopen`, `fgets`, `strtok` from libc.
* Optionally support a small binary format for faster loading.

9.3 Batching
Implement:

* `dataset_get_batch(Dataset*, batch_index, batch_size, Tensor** out_x, Tensor** out_y)`

  * Returns views or copies representing the batch.

No shuffling required in v1 but define:

* `dataset_shuffle_indices(Dataset*, uint64_t* index_array)` if time permits.

---

10. Model I/O (model_io.asm)

---

10.1 Serialization Format
Define a simple binary format:

Header:

* Magic bytes: `DLASM1\0`
* uint32: number of tensors.
  For each tensor:
* uint32: name_length
* char[name_length]: tensor_name (e.g., "linear1.weight")
* uint32: dtype
* uint32: ndim
* uint64[ndim]: shape
* raw data bytes: contiguous in row-major, in tensor dtype.

10.2 Implement

* `model_save(const char* path, Module** modules, uint32_t n_modules)`

  * Iterate all params; collect tensors and names.
* `model_load(const char* path, Module** modules, uint32_t n_modules)`

  * Load and assign into existing tensors (shape and dtype must match; assert).

---

11. Config / DSL Parser (config_parser.asm)

---

11.1 Config Format
Define a simple line-based config:

Example:

```
[model]
input_dim=784
layer=Linear,784,64
activation=ReLU
layer=Linear,64,10
activation=Softmax

[train]
optimizer=Adam
lr=0.001
epochs=10
batch_size=64
loss=CrossEntropy
```

11.2 Parser Responsibilities

* Read file line by line.
* Skip comments (`# ...`) and empty lines.
* For `[model]` block:

  * Build ordered list of layers and activations.
* For `[train]` block:

  * Populate training hyperparameters.

11.3 Output
Parser must:

* Construct an array of `Module*` representing the sequential model.
* Construct a config struct:

```c
struct TrainConfig {
    char optimizer_name[16];
    double lr;
    uint64_t epochs;
    uint64_t batch_size;
    char loss_name[16];
};
```

---

12. Training Loop (main.asm)

---

12.1 Train Binary (`train`)
Responsibilities:

* Parse command line:

  * `train <config_path> <data_path> <label_path> <output_model_path>`
* Steps:

  1. Load config.
  2. Construct model modules from config.
  3. Wrap parameters in autograd `Node`s.
  4. Load dataset.
  5. Initialize optimizer based on config.
  6. For each epoch:

     * For each batch:

       * Get batch tensors `x`, `y`.
       * Forward pass:

         * Wrap `x` as `Node* input`.
         * Sequentially call module forward functions to get `Node* logits`.
       * Compute loss `Node* loss` (per config).
       * Zero all grads.
       * `backward(loss)`.
       * `optimizer.step()`.
     * Optionally log epoch loss.
  7. Save model parameters to output path.

12.2 Infer Binary (`infer`)
Responsibilities:

* Command line:

  * `infer <model_path> <data_path>`
* Steps:

  1. Construct same model structure as training used (hard-coded or via config).
  2. Load weights from `model_path`.
  3. Load data samples.
  4. For each sample or batch:

     * Forward pass only.
     * Print predicted class indices or probabilities.

---

13. Testing and Validation (tests.asm)

---

13.1 Numeric Gradient Checking
Implement:

* `grad_check(Node* output, Tensor* param, double eps)`

  * For each element in `param` (or small subset):

    * Perturb param[i] by ±eps.
    * Recompute loss.
    * Approx grad = (L+ - L-) / (2*eps).
    * Compare with backprop grad at element i.

Use this for:

* Linear layer.
* Simple MLP on XOR.

13.2 Unit Tests
Implement basic tests:

* Tensor creation, reshape, matmul.
* ReLU, sigmoid outputs for known inputs.
* Autograd on simple scalar expressions (e.g., f(x) = x^2).

---

14. Performance and Constraints

---

14.1 Constraints

* Network sizes: small (e.g., up to a few thousand parameters).
* Batch sizes: moderate (e.g., 32–256).
* No multi-threading required in initial version.

14.2 Performance Expectations

* Use SIMD in inner loops.
* Avoid heap fragmentation:

  * Reuse buffers where possible (especially optimizer states, gradients).

---

15. Build and Deployment

---

15.1 Build

* Provide Makefile targets:

  * `make train`
  * `make infer`
  * `make tests`
* Use `nasm -f elf64` and `ld` or `gcc` for linking with `-lm`.

15.2 Deliverables

* Source `.asm` files as specified.
* `Makefile`.
* Example config files.
* Example dataset (small CSV) and instructions.
* Minimal README describing:

  * Build steps.
  * CLI usage.
  * Config syntax.

This is the complete Level 10 spec. Do not introduce new abstractions beyond those listed. If a choice is underspecified, choose the simplest implementation that satisfies the described behavior.
