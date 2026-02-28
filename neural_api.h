#ifndef NEURAL_API_H
#define NEURAL_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================ */
/* Error Codes */
/* ============================================================================ */

typedef enum {
    NEURAL_OK                   = 0,
    NEURAL_ERR_NULL_POINTER     = 1,
    NEURAL_ERR_OUT_OF_MEMORY    = 2,
    NEURAL_ERR_INVALID_ARGUMENT = 3,
    NEURAL_ERR_SHAPE_MISMATCH   = 4,
    NEURAL_ERR_DTYPE_MISMATCH   = 5,
    NEURAL_ERR_FILE_NOT_FOUND   = 6,
    NEURAL_ERR_FILE_READ        = 7,
    NEURAL_ERR_FILE_WRITE       = 8,
    NEURAL_ERR_PARSE_ERROR      = 9,
    NEURAL_ERR_INVALID_CONFIG   = 10,
    NEURAL_ERR_TENSOR_TOO_LARGE = 11,
    NEURAL_ERR_INVALID_DTYPE    = 12,
    NEURAL_ERR_DIM_MISMATCH     = 13,
    NEURAL_ERR_NOT_IMPLEMENTED  = 14,
    NEURAL_ERR_INTERNAL         = 15,
    NEURAL_ERR_GRAD_CHECK       = 16,
    NEURAL_ERR_NAN_DETECTED     = 17,
    NEURAL_ERR_INF_DETECTED     = 18
} NeuralError;

/* ============================================================================ */
/* Data Types */
/* ============================================================================ */

typedef enum {
    NEURAL_DTYPE_FLOAT32 = 0,
    NEURAL_DTYPE_FLOAT64 = 1
} NeuralDtype;

/* ============================================================================ */
/* Opaque Types (forward declarations) */
/* ============================================================================ */

typedef struct NeuralTensor NeuralTensor;
typedef struct NeuralNode NeuralNode;
typedef struct NeuralLinear NeuralLinear;
typedef struct NeuralSequential NeuralSequential;
typedef struct NeuralOptimizer NeuralOptimizer;
typedef struct NeuralDataset NeuralDataset;
typedef struct NeuralConfig NeuralConfig;

/* ============================================================================ */
/* Framework Initialization */
/* ============================================================================ */

/**
 * @brief Initialize the neural framework.
 * @return NEURAL_OK on success, error code on failure
 *
 * Must be called before using any other functions.
 */
int neural_init(void);

/**
 * @brief Shutdown the neural framework and free all resources.
 */
void neural_shutdown(void);

/**
 * @brief Get the version string of the framework.
 * @return Pointer to version string (do not free)
 */
const char* neural_version(void);

/* ============================================================================ */
/* Error Handling */
/* ============================================================================ */

/**
 * @brief Clear the last error.
 */
void neural_clear_error(void);

/**
 * @brief Get the last error code.
 * @return Last error code (NEURAL_OK if no error)
 */
int neural_get_last_error(void);

/**
 * @brief Get the error message for an error code.
 * @param error_code Error code to get message for
 * @return Pointer to error message string (do not free)
 */
const char* neural_get_error_message(int error_code);

/* ============================================================================ */
/* SIMD Information */
/* ============================================================================ */

/**
 * @brief Get the available SIMD level.
 * @return SIMD level (0=scalar, 1=SSE2, 2=AVX, 3=AVX2, 4=AVX-512)
 */
int neural_get_simd_level(void);

/**
 * @brief Get SIMD level name as string.
 * @return Pointer to SIMD name string (do not free)
 */
const char* neural_get_simd_name(void);

/* ============================================================================ */
/* Tensor Operations */
/* ============================================================================ */

/**
 * @brief Create a new tensor with given shape.
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type (NEURAL_DTYPE_FLOAT32 or NEURAL_DTYPE_FLOAT64)
 * @return Pointer to new tensor, or NULL on error
 */
NeuralTensor* neural_tensor_create(const uint64_t* shape, uint64_t ndim, int dtype);

/**
 * @brief Create a tensor filled with zeros.
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @return Pointer to new tensor, or NULL on error
 */
NeuralTensor* neural_tensor_zeros(const uint64_t* shape, uint64_t ndim, int dtype);

/**
 * @brief Create a tensor filled with ones.
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @return Pointer to new tensor, or NULL on error
 */
NeuralTensor* neural_tensor_ones(const uint64_t* shape, uint64_t ndim, int dtype);

/**
 * @brief Create a tensor with random values (uniform [0, 1)).
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @return Pointer to new tensor, or NULL on error
 */
NeuralTensor* neural_tensor_random(const uint64_t* shape, uint64_t ndim, int dtype);

/**
 * @brief Create a tensor from existing data (copies the data).
 * @param data Pointer to source data
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type
 * @return Pointer to new tensor, or NULL on error
 */
NeuralTensor* neural_tensor_from_data(const void* data, const uint64_t* shape, uint64_t ndim, int dtype);

/**
 * @brief Create a tensor that wraps an external buffer (zero-copy).
 * 
 * This function creates a tensor that directly uses the provided buffer
 * without copying data. The caller is responsible for keeping the buffer
 * alive for the lifetime of the tensor. This is ideal for NumPy integration.
 * 
 * @param buffer Pointer to existing data buffer (must remain valid)
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param dtype Data type (NEURAL_DTYPE_FLOAT32 or NEURAL_DTYPE_FLOAT64)
 * @param strides Array of strides in bytes (or NULL for C-contiguous)
 * @return Pointer to new tensor, or NULL on error
 */
NeuralTensor* neural_tensor_from_buffer(void* buffer, const uint64_t* shape, uint64_t ndim, int dtype, const int64_t* strides);

/**
 * @brief Get the stride array of a tensor (in bytes).
 * @param tensor Tensor to query
 * @return Pointer to stride array, or NULL if tensor is NULL
 */
const int64_t* neural_tensor_stride(NeuralTensor* tensor);

/**
 * @brief Check if a tensor is C-contiguous (row-major).
 * @param tensor Tensor to check
 * @return 1 if contiguous, 0 if not, -1 on error
 */
int neural_tensor_is_contiguous(NeuralTensor* tensor);

/**
 * @brief Return a contiguous copy of the tensor if needed.
 * @param tensor Input tensor
 * @return Contiguous tensor (may be same as input if already contiguous), or NULL on error
 */
NeuralTensor* neural_tensor_make_contiguous(NeuralTensor* tensor);

/**
 * @brief Buffer info structure for NumPy buffer protocol integration.
 */
typedef struct {
    void* data;             /**< Pointer to buffer data */
    uint64_t itemsize;      /**< Size of each element in bytes */
    uint64_t ndim;          /**< Number of dimensions */
    const uint64_t* shape;  /**< Shape array pointer */
    const int64_t* strides; /**< Strides array pointer (in bytes) */
    int readonly;           /**< Whether buffer is read-only */
    char format[8];         /**< Format string ('f' for float32, 'd' for float64) */
} NeuralBufferInfo;

/**
 * @brief Get buffer protocol info for NumPy integration.
 * 
 * This function fills a NeuralBufferInfo structure with the information
 * needed to implement Python's buffer protocol for zero-copy NumPy access.
 * 
 * @param tensor Tensor to get info for
 * @param info Pointer to NeuralBufferInfo structure to fill
 * @return NEURAL_OK on success, error code on failure
 */
int neural_buffer_info(NeuralTensor* tensor, NeuralBufferInfo* info);

/**
 * @brief Free a tensor and its data.
 * @param tensor Tensor to free
 */
void neural_tensor_free(NeuralTensor* tensor);

/**
 * @brief Get the data pointer of a tensor.
 * @param tensor Tensor to get data from
 * @return Pointer to tensor data, or NULL if tensor is NULL
 */
void* neural_tensor_data(NeuralTensor* tensor);

/**
 * @brief Get the number of dimensions of a tensor.
 * @param tensor Tensor to query
 * @return Number of dimensions, or 0 if tensor is NULL
 */
uint64_t neural_tensor_ndim(NeuralTensor* tensor);

/**
 * @brief Get the shape of a tensor.
 * @param tensor Tensor to query
 * @return Pointer to shape array, or NULL if tensor is NULL
 */
const uint64_t* neural_tensor_shape(NeuralTensor* tensor);

/**
 * @brief Get the total number of elements in a tensor.
 * @param tensor Tensor to query
 * @return Number of elements, or 0 if tensor is NULL
 */
uint64_t neural_tensor_numel(NeuralTensor* tensor);

/**
 * @brief Get the data type of a tensor.
 * @param tensor Tensor to query
 * @return Data type, or -1 if tensor is NULL
 */
int neural_tensor_dtype(NeuralTensor* tensor);

/**
 * @brief Get the size in bytes of the tensor data.
 * @param tensor Tensor to query
 * @return Size in bytes, or 0 if tensor is NULL
 */
uint64_t neural_tensor_bytes(NeuralTensor* tensor);

/**
 * @brief Fill a tensor with a scalar value.
 * @param tensor Tensor to fill
 * @param value Value to fill with (as double, will be converted)
 * @return NEURAL_OK on success, error code on failure
 */
int neural_tensor_fill(NeuralTensor* tensor, double value);

/**
 * @brief Copy data from one tensor to another.
 * @param dst Destination tensor
 * @param src Source tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_tensor_copy(NeuralTensor* dst, const NeuralTensor* src);

/**
 * @brief Reshape a tensor (returns a view if possible).
 * @param tensor Tensor to reshape
 * @param new_shape New shape
 * @param new_ndim Number of dimensions in new shape
 * @return Pointer to reshaped tensor (may be same as input), or NULL on error
 */
NeuralTensor* neural_tensor_reshape(NeuralTensor* tensor, const uint64_t* new_shape, uint64_t new_ndim);

/* ============================================================================ */
/* Math Operations (Element-wise) */
/* ============================================================================ */

/**
 * @brief Element-wise addition: out = a + b
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_add(NeuralTensor* out, const NeuralTensor* a, const NeuralTensor* b);

/**
 * @brief Element-wise subtraction: out = a - b
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_sub(NeuralTensor* out, const NeuralTensor* a, const NeuralTensor* b);

/**
 * @brief Element-wise multiplication: out = a * b
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_mul(NeuralTensor* out, const NeuralTensor* a, const NeuralTensor* b);

/**
 * @brief Element-wise division: out = a / b
 * @param out Output tensor
 * @param a First input tensor
 * @param b Second input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_div(NeuralTensor* out, const NeuralTensor* a, const NeuralTensor* b);

/**
 * @brief Matrix multiplication: out = a @ b
 * @param out Output tensor (must be pre-allocated)
 * @param a First matrix (M x K)
 * @param b Second matrix (K x N)
 * @return NEURAL_OK on success, error code on failure
 */
int neural_matmul(NeuralTensor* out, const NeuralTensor* a, const NeuralTensor* b);

/**
 * @brief Sum all elements in a tensor.
 * @param tensor Input tensor
 * @param result Pointer to store result
 * @return NEURAL_OK on success, error code on failure
 */
int neural_sum(const NeuralTensor* tensor, double* result);

/**
 * @brief Mean of all elements in a tensor.
 * @param tensor Input tensor
 * @param result Pointer to store result
 * @return NEURAL_OK on success, error code on failure
 */
int neural_mean(const NeuralTensor* tensor, double* result);

/* ============================================================================ */
/* Activation Functions */
/* ============================================================================ */

/**
 * @brief ReLU activation: out = max(0, input)
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_relu(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Sigmoid activation: out = 1 / (1 + exp(-input))
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_sigmoid(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Tanh activation: out = tanh(input)
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_tanh(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Softmax activation (along last dimension)
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_softmax(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief GELU activation (Gaussian Error Linear Unit)
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_gelu(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Leaky ReLU activation: out = max(alpha * input, input)
 * @param out Output tensor
 * @param input Input tensor
 * @param alpha Negative slope (typically 0.01)
 * @return NEURAL_OK on success, error code on failure
 */
int neural_leaky_relu(NeuralTensor* out, const NeuralTensor* input, double alpha);

/**
 * @brief ELU activation (Exponential Linear Unit)
 * ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
 * @param out Output tensor
 * @param input Input tensor
 * @param alpha Scale for negative values (typically 1.0)
 * @return NEURAL_OK on success, error code on failure
 */
int neural_elu(NeuralTensor* out, const NeuralTensor* input, double alpha);

/**
 * @brief SELU activation (Scaled Exponential Linear Unit)
 * SELU(x) = λ * (x if x > 0, else α * (exp(x) - 1))
 * With λ ≈ 1.0507, α ≈ 1.6733 for self-normalizing properties
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_selu(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Swish/SiLU activation (Sigmoid Linear Unit)
 * Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_swish(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Mish activation
 * Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_mish(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Hard Swish activation (efficient approximation of Swish)
 * HardSwish(x) = 0 if x <= -3, x if x >= 3, else x * (x + 3) / 6
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_hardswish(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Softplus activation (smooth ReLU approximation)
 * Softplus(x) = ln(1 + exp(x))
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_softplus(NeuralTensor* out, const NeuralTensor* input);

/**
 * @brief Hardtanh activation (piecewise linear tanh approximation)
 * Hardtanh(x) = -1 if x < -1, 1 if x > 1, else x
 * @param out Output tensor
 * @param input Input tensor
 * @return NEURAL_OK on success, error code on failure
 */
int neural_hardtanh(NeuralTensor* out, const NeuralTensor* input);

/* ============================================================================ */
/* Neural Network Layers */
/* ============================================================================ */

/**
 * @brief Create a linear (fully connected) layer.
 * @param in_features Number of input features
 * @param out_features Number of output features
 * @param bias Whether to include bias
 * @return Pointer to layer, or NULL on error
 */
NeuralLinear* neural_linear_create(uint64_t in_features, uint64_t out_features, int bias);

/**
 * @brief Free a linear layer.
 * @param layer Linear layer to free
 */
void neural_linear_free(NeuralLinear* layer);

/**
 * @brief Forward pass through linear layer.
 * @param layer Linear layer
 * @param input Input tensor (batch_size x in_features)
 * @param output Output tensor (batch_size x out_features)
 * @return NEURAL_OK on success, error code on failure
 */
int neural_linear_forward(NeuralLinear* layer, const NeuralTensor* input, NeuralTensor* output);

/**
 * @brief Get the weight tensor of a linear layer.
 * @param layer Linear layer
 * @return Pointer to weight tensor
 */
NeuralTensor* neural_linear_weight(NeuralLinear* layer);

/**
 * @brief Get the bias tensor of a linear layer.
 * @param layer Linear layer
 * @return Pointer to bias tensor
 */
NeuralTensor* neural_linear_bias(NeuralLinear* layer);

/* ============================================================================ */
/* Sequential Container */
/* ============================================================================ */

/**
 * @brief Create a sequential container for chaining modules.
 * @param modules Array of modules to add (can be NULL for empty container)
 * @param num_modules Number of modules in the array
 * @return Pointer to sequential container, or NULL on error
 */
NeuralSequential* neural_sequential_create(NeuralLinear** modules, uint64_t num_modules);

/**
 * @brief Free a sequential container and all its modules.
 * @param seq Sequential container to free
 */
void neural_sequential_free(NeuralSequential* seq);

/**
 * @brief Add a module to the end of the sequential container.
 * @param seq Sequential container
 * @param module Module to add
 * @return NEURAL_OK on success, error code on failure
 */
int neural_sequential_add(NeuralSequential* seq, NeuralLinear* module);

/**
 * @brief Forward pass through all modules in the sequential container.
 * @param seq Sequential container
 * @param input Input tensor
 * @param output Output tensor (final output after all modules)
 * @return NEURAL_OK on success, error code on failure
 */
int neural_sequential_forward(NeuralSequential* seq, const NeuralTensor* input, NeuralTensor* output);

/**
 * @brief Get the number of modules in the sequential container.
 * @param seq Sequential container
 * @return Number of modules, or 0 if seq is NULL
 */
uint64_t neural_sequential_size(NeuralSequential* seq);

/**
 * @brief Get a module at the specified index.
 * @param seq Sequential container
 * @param index Index of the module (0-based)
 * @return Pointer to module, or NULL if index is out of bounds
 */
NeuralLinear* neural_sequential_get(NeuralSequential* seq, uint64_t index);

/**
 * @brief Get all parameters from all modules in the container.
 * @param seq Sequential container
 * @param params Array to store parameter tensors (caller must allocate)
 * @param max_params Maximum number of parameters to return
 * @return Number of parameters found, or -1 on error
 */
int64_t neural_sequential_parameters(NeuralSequential* seq, NeuralTensor** params, uint64_t max_params);

/* ============================================================================ */
/* Loss Functions */
/* ============================================================================ */

/**
 * @brief Mean Squared Error loss.
 * @param prediction Predicted values
 * @param target Target values
 * @param loss Pointer to store loss value
 * @return NEURAL_OK on success, error code on failure
 */
int neural_mse_loss(const NeuralTensor* prediction, const NeuralTensor* target, double* loss);

/**
 * @brief Cross-entropy loss.
 * @param prediction Predicted values
 * @param target Target values
 * @param loss Pointer to store loss value
 * @return NEURAL_OK on success, error code on failure
 */
int neural_cross_entropy_loss(const NeuralTensor* prediction, const NeuralTensor* target, double* loss);

/* ============================================================================ */
/* Optimizers */
/* ============================================================================ */

/**
 * @brief Create an SGD optimizer.
 * @param learning_rate Learning rate
 * @param momentum Momentum factor (0 for no momentum)
 * @return Pointer to optimizer, or NULL on error
 */
NeuralOptimizer* neural_sgd_create(double learning_rate, double momentum);

/**
 * @brief Create an Adam optimizer.
 * @param learning_rate Learning rate
 * @param beta1 First moment decay rate
 * @param beta2 Second moment decay rate
 * @param epsilon Numerical stability constant
 * @return Pointer to optimizer, or NULL on error
 */
NeuralOptimizer* neural_adam_create(double learning_rate, double beta1, double beta2, double epsilon);
NeuralOptimizer* neural_adamw_create(double learning_rate, double beta1, double beta2, double epsilon, double weight_decay);

/**
 * @brief Free an optimizer.
 * @param optimizer Optimizer to free
 */
void neural_optimizer_free(NeuralOptimizer* optimizer);

/**
 * @brief Perform one optimization step.
 * @param optimizer Optimizer to use
 * @param params Array of parameter tensors
 * @param grads Array of gradient tensors
 * @param num_params Number of parameters
 * @return NEURAL_OK on success, error code on failure
 */
int neural_optimizer_step(NeuralOptimizer* optimizer, NeuralTensor** params, NeuralTensor** grads, uint64_t num_params);

/**
 * @brief Zero all gradients.
 * @param num_params Number of parameters
 * @param grads Array of gradient tensors
 * @return NEURAL_OK on success, error code on failure
 */
int neural_optimizer_zero_grad(NeuralTensor** grads, uint64_t num_params);

/**
 * @brief Clip gradients by global L2 norm.
 * If the L2 norm of all gradients exceeds max_norm, scale all gradients
 * by (max_norm / current_norm).
 * @param optimizer Optimizer containing parameters with gradients
 * @param max_norm Maximum allowed L2 norm of gradients
 * @return NEURAL_OK on success, error code on failure
 */
int neural_clip_grad_norm(NeuralOptimizer* optimizer, double max_norm);

/**
 * @brief Clip gradient values to a specified range.
 * Clips each gradient element to the range [min_val, max_val].
 * @param optimizer Optimizer containing parameters with gradients
 * @param min_val Minimum allowed gradient value
 * @param max_val Maximum allowed gradient value
 * @return NEURAL_OK on success, error code on failure
 */
int neural_clip_grad_value(NeuralOptimizer* optimizer, double min_val, double max_val);

/* ============================================================================ */
/* Autograd */
/* ============================================================================ */

/**
 * @brief Create a computation graph node.
 * @param tensor Tensor for this node
 * @param requires_grad Whether to track gradients
 * @return Pointer to node, or NULL on error
 */
NeuralNode* neural_node_create(NeuralTensor* tensor, int requires_grad);

/**
 * @brief Free a computation graph node.
 * @param node Node to free
 */
void neural_node_free(NeuralNode* node);

/**
 * @brief Perform backward pass from a loss node.
 * @param loss_node Node containing loss value
 * @return NEURAL_OK on success, error code on failure
 */
int neural_backward(NeuralNode* loss_node);

/**
 * @brief Get the gradient tensor from a node.
 * @param node Node to get gradient from
 * @return Pointer to gradient tensor
 */
NeuralTensor* neural_node_grad(NeuralNode* node);

/* ============================================================================ */
/* Dataset Operations */
/* ============================================================================ */

/**
 * @brief Load a dataset from CSV files.
 * @param data_path Path to data CSV file
 * @param labels_path Path to labels CSV file (can be NULL for unlabeled)
 * @return Pointer to dataset, or NULL on error
 */
NeuralDataset* neural_dataset_load_csv(const char* data_path, const char* labels_path);

/**
 * @brief Free a dataset.
 * @param dataset Dataset to free
 */
void neural_dataset_free(NeuralDataset* dataset);

/**
 * @brief Get the number of samples in a dataset.
 * @param dataset Dataset to query
 * @return Number of samples
 */
uint64_t neural_dataset_size(NeuralDataset* dataset);

/**
 * @brief Get a batch from the dataset.
 * @param dataset Dataset to get batch from
 * @param batch_idx Batch index
 * @param batch_size Batch size
 * @param data Output tensor for data
 * @param labels Output tensor for labels (can be NULL)
 * @return NEURAL_OK on success, error code on failure
 */
int neural_dataset_get_batch(NeuralDataset* dataset, uint64_t batch_idx, uint64_t batch_size, NeuralTensor* data, NeuralTensor* labels);

/* ============================================================================ */
/* Configuration */
/* ============================================================================ */

/**
 * @brief Load configuration from INI file.
 * @param path Path to configuration file
 * @return Pointer to config, or NULL on error
 */
NeuralConfig* neural_config_load(const char* path);

/**
 * @brief Free a configuration.
 * @param config Configuration to free
 */
void neural_config_free(NeuralConfig* config);

/**
 * @brief Get an integer value from config.
 * @param config Configuration
 * @param section Section name
 * @param key Key name
 * @param default_value Default value if key not found
 * @return Integer value
 */
int neural_config_get_int(NeuralConfig* config, const char* section, const char* key, int default_value);

/**
 * @brief Get a float value from config.
 * @param config Configuration
 * @param section Section name
 * @param key Key name
 * @param default_value Default value if key not found
 * @return Float value
 */
double neural_config_get_float(NeuralConfig* config, const char* section, const char* key, double default_value);

/**
 * @brief Get a string value from config.
 * @param config Configuration
 * @param section Section name
 * @param key Key name
 * @param default_value Default value if key not found
 * @return String value (caller must free)
 */
const char* neural_config_get_string(NeuralConfig* config, const char* section, const char* key, const char* default_value);

/* ============================================================================ */
/* TensorBoard Logging (implemented in tb_logger.c)                             */
/* ============================================================================ */

/**
 * @brief Create a TensorBoard event file writer.
 * @param logdir Directory path for event files (created if needed)
 * @return Writer handle (>=0) on success, -1 on error
 */
int tb_create_writer(const char *logdir);

/**
 * @brief Log a scalar value.
 * @param handle Writer handle from tb_create_writer
 * @param tag Metric name (e.g. "loss/train")
 * @param value Scalar value
 * @param step Global step number
 * @return 0 on success, -1 on error
 */
int tb_add_scalar(int handle, const char *tag, float value, int64_t step);

/**
 * @brief Log multiple scalars under a common tag prefix.
 * @param handle Writer handle
 * @param tag_prefix Common prefix (e.g. "loss")
 * @param subtags Array of sub-tag strings (e.g. ["train", "val"])
 * @param values Corresponding float values
 * @param count Number of scalars
 * @param step Global step number
 * @return 0 on success, -1 on error
 */
int tb_add_scalars(int handle, const char *tag_prefix,
                   const char **subtags, const float *values,
                   int count, int64_t step);

/**
 * @brief Log histogram statistics (min/max/mean) as scalars.
 * @param handle Writer handle
 * @param tag Base tag name
 * @param data Float array of values
 * @param count Number of values
 * @param step Global step number
 * @return 0 on success, -1 on error
 */
int tb_add_histogram_stats(int handle, const char *tag,
                           const float *data, int count, int64_t step);

/**
 * @brief Flush pending writes to disk.
 * @param handle Writer handle
 * @return 0 on success
 */
int tb_flush(int handle);

/**
 * @brief Close a writer and release resources.
 * @param handle Writer handle
 * @return 0 on success
 */
int tb_close(int handle);

/**
 * @brief Get the log directory for a writer.
 * @param handle Writer handle
 * @return Log directory string, or NULL
 */
const char *tb_get_logdir(int handle);

/**
 * @brief Get the event file path for a writer.
 * @param handle Writer handle
 * @return File path string, or NULL
 */
const char *tb_get_filepath(int handle);

/* ============================================================================ */
/* Model Pruning (implemented in pruning.c)                                     */
/* ============================================================================ */

/**
 * @brief Unstructured magnitude pruning — zero weights with |w| < threshold.
 * @param weights Float64 weight array (modified in-place)
 * @param mask Output uint8 mask (1=kept, 0=pruned; may be NULL)
 * @param n Number of elements
 * @param threshold Absolute value threshold
 * @return Number of weights pruned, or -1 on error
 */
int64_t prune_magnitude(double *weights, uint8_t *mask, int64_t n, double threshold);

/**
 * @brief Keep only the top-k weights by magnitude, prune the rest.
 * @param weights Float64 weight array (modified in-place)
 * @param mask Output uint8 mask (may be NULL)
 * @param n Number of elements
 * @param keep_ratio Fraction to keep (0.0 to 1.0)
 * @return Number pruned, or -1 on error
 */
int64_t prune_topk(double *weights, uint8_t *mask, int64_t n, double keep_ratio);

/**
 * @brief Structured row pruning — zero entire rows with L2 norm < threshold.
 * @param weights Float64 matrix (rows x cols, row-major)
 * @param rows Number of rows
 * @param cols Number of columns
 * @param threshold L2 norm threshold
 * @param row_mask Output mask per row (may be NULL)
 * @return Number of rows pruned, or -1 on error
 */
int64_t prune_rows(double *weights, int64_t rows, int64_t cols,
                   double threshold, uint8_t *row_mask);

/**
 * @brief Structured column pruning — zero entire columns with L2 norm < threshold.
 */
int64_t prune_cols(double *weights, int64_t rows, int64_t cols,
                   double threshold, uint8_t *col_mask);

/**
 * @brief Compute sparsity ratio (fraction of zeros).
 */
double compute_sparsity(const double *weights, int64_t n);

/**
 * @brief Count non-zero elements.
 */
int64_t count_nonzero(const double *weights, int64_t n);

/**
 * @brief Find threshold that achieves target sparsity.
 * @param weights Float64 array
 * @param n Number of elements
 * @param target_sparsity Desired zero fraction (0.0 to 1.0)
 * @return Magnitude threshold
 */
double compute_threshold_for_sparsity(const double *weights, int64_t n,
                                      double target_sparsity);

/**
 * @brief Re-apply a pruning mask (e.g. after optimizer step).
 * @param weights Float64 array (modified in-place)
 * @param mask Uint8 mask (1=keep, 0=zero)
 * @param n Number of elements
 */
void apply_mask(double *weights, const uint8_t *mask, int64_t n);

/* ============================================================================ */
/* INT8 Quantization (implemented in quantize.c)                                */
/* ============================================================================ */

/** Quantization parameters. */
typedef struct {
    double  scale;
    int32_t zero_point;
    double  min_val;
    double  max_val;
    int     symmetric;
} QuantParams;

/**
 * @brief Calibrate quantization params from min/max of data.
 * @param data Float64 input array
 * @param n Number of elements
 * @param symmetric 1 for symmetric, 0 for affine
 * @param out Output QuantParams
 * @return 0 on success
 */
int calibrate_minmax(const double *data, int64_t n, int symmetric,
                     QuantParams *out);

/**
 * @brief Calibrate with percentile clipping for outlier robustness.
 * @param percentile Clip fraction (e.g. 0.01 clips 1% on each tail)
 */
int calibrate_percentile(const double *data, int64_t n, double percentile,
                         int symmetric, QuantParams *out);

/**
 * @brief Quantize float64 array to int8.
 */
int quantize_tensor(const double *data, int8_t *out, int64_t n,
                    const QuantParams *params);

/**
 * @brief Dequantize int8 array back to float64.
 */
int dequantize_tensor(const int8_t *data, double *out, int64_t n,
                      const QuantParams *params);

/**
 * @brief Quantized int8 matrix multiply with float64 output.
 * C[M,N] = dequant(A[M,K]) @ dequant(B[K,N])
 */
int quantized_matmul(const int8_t *A, const int8_t *B, double *C,
                     int64_t M, int64_t K, int64_t N,
                     const QuantParams *params_a,
                     const QuantParams *params_b);

/**
 * @brief Compute mean squared quantization error.
 */
double quantization_error(const double *original, int64_t n,
                          const QuantParams *params);

/**
 * @brief Compute signal-to-noise ratio of quantization (in dB).
 */
double quantization_snr(const double *original, int64_t n,
                        const QuantParams *params);

/* ── BatchNorm1d (batchnorm.c) ────────────────────────────────────── */

typedef struct BatchNorm1d BatchNorm1d;

BatchNorm1d *batchnorm1d_create(int64_t num_features, double momentum, double eps);
void         batchnorm1d_free(BatchNorm1d *bn);
int          batchnorm1d_forward(BatchNorm1d *bn, const double *input, double *output,
                                  int64_t batch_size, int training);
int          batchnorm1d_backward(BatchNorm1d *bn, const double *grad_output,
                                   double *grad_input, double *grad_gamma,
                                   double *grad_beta, int64_t batch_size);
double      *batchnorm1d_gamma(BatchNorm1d *bn);
double      *batchnorm1d_beta(BatchNorm1d *bn);
double      *batchnorm1d_running_mean(BatchNorm1d *bn);
double      *batchnorm1d_running_var(BatchNorm1d *bn);
int64_t      batchnorm1d_num_features(BatchNorm1d *bn);

/* ── LayerNorm (batchnorm.c) ──────────────────────────────────────── */

typedef struct LayerNorm LayerNorm;

LayerNorm   *layernorm_create(int64_t num_features, double eps);
void         layernorm_free(LayerNorm *ln);
int          layernorm_forward(LayerNorm *ln, const double *input, double *output,
                               int64_t batch_size);
int          layernorm_backward(LayerNorm *ln, const double *grad_output,
                                double *grad_input, double *grad_gamma,
                                double *grad_beta, int64_t batch_size);
double      *layernorm_gamma(LayerNorm *ln);
double      *layernorm_beta(LayerNorm *ln);
int64_t      layernorm_num_features(LayerNorm *ln);

/* ── Label-Smoothed Cross-Entropy (metrics_losses.c) ──────────────── */

int label_smoothing_ce_forward(const double *logits, const int64_t *targets,
                               int64_t batch_size, int64_t num_classes,
                               double smoothing, double *loss_out);
int label_smoothing_ce_backward(const double *logits, const int64_t *targets,
                                int64_t batch_size, int64_t num_classes,
                                double smoothing, double *grad_out);

/* ── ROC-AUC Score (metrics_losses.c) ─────────────────────────────── */

int roc_auc_score(const double *y_true, const double *y_score,
                  int64_t n, double *auc_out);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_API_H */