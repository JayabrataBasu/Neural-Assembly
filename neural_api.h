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

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_API_H */