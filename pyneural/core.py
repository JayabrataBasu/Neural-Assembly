"""
Core functionality and library loading for PyNeural.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

# Error codes matching neural_api.h
class NeuralError:
    OK = 0
    NULL_POINTER = 1
    OUT_OF_MEMORY = 2
    INVALID_ARGUMENT = 3
    SHAPE_MISMATCH = 4
    DTYPE_MISMATCH = 5
    FILE_NOT_FOUND = 6
    FILE_READ = 7
    FILE_WRITE = 8
    PARSE_ERROR = 9
    INVALID_CONFIG = 10
    TENSOR_TOO_LARGE = 11
    INVALID_DTYPE = 12
    DIM_MISMATCH = 13
    NOT_IMPLEMENTED = 14
    INTERNAL = 15
    GRAD_CHECK = 16
    NAN_DETECTED = 17
    INF_DETECTED = 18


class NeuralDtype:
    FLOAT32 = 0
    FLOAT64 = 1


class NeuralException(Exception):
    """Exception raised when a Neural Assembly operation fails."""
    
    def __init__(self, error_code: int, message: str = None):
        self.error_code = error_code
        if message is None:
            message = get_error_message(error_code)
        super().__init__(f"NeuralError({error_code}): {message}")


def _find_library() -> str:
    """Find the libneural.so shared library."""
    # Search paths in order of preference
    search_paths = [
        # Same directory as this module
        Path(__file__).parent.parent / "libneural.so",
        # Current working directory
        Path.cwd() / "libneural.so",
        # System paths
        Path("/usr/local/lib/libneural.so"),
        Path("/usr/lib/libneural.so"),
    ]
    
    # Also check LD_LIBRARY_PATH
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    for path in ld_library_path.split(":"):
        if path:
            search_paths.append(Path(path) / "libneural.so")
    
    for path in search_paths:
        if path.exists():
            return str(path)
    
    # Try to load by name (rely on system library path)
    return "libneural.so"


# Load the shared library
_lib_path = _find_library()
try:
    _lib = ctypes.CDLL(_lib_path)
except OSError as e:
    raise ImportError(
        f"Could not load libneural.so from {_lib_path}. "
        f"Make sure you have built the shared library with 'make lib'. "
        f"Original error: {e}"
    )

# Define function signatures

# Framework initialization
_lib.neural_init.restype = ctypes.c_int
_lib.neural_init.argtypes = []

_lib.neural_shutdown.restype = None
_lib.neural_shutdown.argtypes = []

_lib.neural_version.restype = ctypes.c_char_p
_lib.neural_version.argtypes = []

# Error handling
_lib.neural_get_last_error.restype = ctypes.c_int
_lib.neural_get_last_error.argtypes = []

_lib.neural_get_error_message.restype = ctypes.c_char_p
_lib.neural_get_error_message.argtypes = [ctypes.c_int]

_lib.neural_clear_error.restype = None
_lib.neural_clear_error.argtypes = []

# SIMD
_lib.neural_get_simd_level.restype = ctypes.c_int
_lib.neural_get_simd_level.argtypes = []

_lib.neural_get_simd_name.restype = ctypes.c_char_p
_lib.neural_get_simd_name.argtypes = []

# Tensor operations
_lib.neural_tensor_create.restype = ctypes.c_void_p
_lib.neural_tensor_create.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64,
    ctypes.c_int,
]

_lib.neural_tensor_zeros.restype = ctypes.c_void_p
_lib.neural_tensor_zeros.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64,
    ctypes.c_int,
]

_lib.neural_tensor_ones.restype = ctypes.c_void_p
_lib.neural_tensor_ones.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_uint64,
    ctypes.c_int,
]

_lib.neural_tensor_free.restype = None
_lib.neural_tensor_free.argtypes = [ctypes.c_void_p]

_lib.neural_tensor_data.restype = ctypes.c_void_p
_lib.neural_tensor_data.argtypes = [ctypes.c_void_p]

_lib.neural_tensor_ndim.restype = ctypes.c_uint64
_lib.neural_tensor_ndim.argtypes = [ctypes.c_void_p]

_lib.neural_tensor_shape.restype = ctypes.POINTER(ctypes.c_uint64)
_lib.neural_tensor_shape.argtypes = [ctypes.c_void_p]

_lib.neural_tensor_numel.restype = ctypes.c_uint64
_lib.neural_tensor_numel.argtypes = [ctypes.c_void_p]

_lib.neural_tensor_dtype.restype = ctypes.c_int
_lib.neural_tensor_dtype.argtypes = [ctypes.c_void_p]

_lib.neural_tensor_bytes.restype = ctypes.c_uint64
_lib.neural_tensor_bytes.argtypes = [ctypes.c_void_p]

_lib.neural_tensor_fill.restype = ctypes.c_int
_lib.neural_tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_double]

_lib.neural_tensor_copy.restype = ctypes.c_int
_lib.neural_tensor_copy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# Math operations
_lib.neural_add.restype = ctypes.c_int
_lib.neural_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_lib.neural_sub.restype = ctypes.c_int
_lib.neural_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_lib.neural_mul.restype = ctypes.c_int
_lib.neural_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_lib.neural_div.restype = ctypes.c_int
_lib.neural_div.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_lib.neural_matmul.restype = ctypes.c_int
_lib.neural_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

# Activations
_lib.neural_relu.restype = ctypes.c_int
_lib.neural_relu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.neural_sigmoid.restype = ctypes.c_int
_lib.neural_sigmoid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.neural_softmax.restype = ctypes.c_int
_lib.neural_softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# C-level activations (activations_c.c) — bypass the autograd node layer
_lib.act_tanh.restype = ctypes.c_int
_lib.act_tanh.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.act_gelu.restype = ctypes.c_int
_lib.act_gelu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.act_leaky_relu.restype = ctypes.c_int
_lib.act_leaky_relu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double]

_lib.act_elu.restype = ctypes.c_int
_lib.act_elu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double]

_lib.act_selu.restype = ctypes.c_int
_lib.act_selu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.act_swish.restype = ctypes.c_int
_lib.act_swish.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.act_mish.restype = ctypes.c_int
_lib.act_mish.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.act_hardswish.restype = ctypes.c_int
_lib.act_hardswish.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.act_softplus.restype = ctypes.c_int
_lib.act_softplus.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_lib.act_hardtanh.restype = ctypes.c_int
_lib.act_hardtanh.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# Linear layer
_lib.neural_linear_create.restype = ctypes.c_void_p
_lib.neural_linear_create.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

_lib.neural_linear_free.restype = None
_lib.neural_linear_free.argtypes = [ctypes.c_void_p]

_lib.neural_linear_forward.restype = ctypes.c_int
_lib.neural_linear_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

# Linear layer weight/bias accessors
_lib.neural_linear_weight.restype = ctypes.c_void_p
_lib.neural_linear_weight.argtypes = [ctypes.c_void_p]

_lib.neural_linear_bias.restype = ctypes.c_void_p
_lib.neural_linear_bias.argtypes = [ctypes.c_void_p]

# Optimizers (assembly — internal use only, kept for backward compat)
_lib.neural_sgd_create.restype = ctypes.c_void_p
_lib.neural_sgd_create.argtypes = [ctypes.c_double, ctypes.c_double]

_lib.neural_adam_create.restype = ctypes.c_void_p
_lib.neural_adam_create.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]

_lib.neural_adamw_create.restype = ctypes.c_void_p
_lib.neural_adamw_create.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]

_lib.neural_optimizer_free.restype = None
_lib.neural_optimizer_free.argtypes = [ctypes.c_void_p]

# C-level optimizers (optimizers_c.c) — these actually work from Python
_lib.opt_sgd_create.restype = ctypes.c_void_p
_lib.opt_sgd_create.argtypes = [ctypes.c_double, ctypes.c_double]

_lib.opt_adam_create.restype = ctypes.c_void_p
_lib.opt_adam_create.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
]

_lib.opt_adamw_create.restype = ctypes.c_void_p
_lib.opt_adamw_create.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
]

_lib.opt_step.restype = ctypes.c_int
_lib.opt_step.argtypes = [
    ctypes.c_void_p,                         # optimizer
    ctypes.POINTER(ctypes.c_void_p),         # params
    ctypes.POINTER(ctypes.c_void_p),         # grads
    ctypes.c_int64,                          # n
]

_lib.opt_free.restype = None
_lib.opt_free.argtypes = [ctypes.c_void_p]

_lib.opt_get_lr.restype = ctypes.c_double
_lib.opt_get_lr.argtypes = [ctypes.c_void_p]

_lib.opt_set_lr.restype = None
_lib.opt_set_lr.argtypes = [ctypes.c_void_p, ctypes.c_double]

# RNN layers (rnn.c)
_lib.lstm_create.restype = ctypes.c_void_p
_lib.lstm_create.argtypes = [ctypes.c_int64, ctypes.c_int64]

_lib.lstm_free.restype = None
_lib.lstm_free.argtypes = [ctypes.c_void_p]

_lib.lstm_forward.restype = ctypes.c_int
_lib.lstm_forward.argtypes = [
    ctypes.c_void_p,                         # layer
    ctypes.POINTER(ctypes.c_double),         # input
    ctypes.POINTER(ctypes.c_double),         # output
    ctypes.POINTER(ctypes.c_double),         # h_out
    ctypes.POINTER(ctypes.c_double),         # c_out
    ctypes.POINTER(ctypes.c_double),         # h_init (nullable)
    ctypes.POINTER(ctypes.c_double),         # c_init (nullable)
    ctypes.c_int64,                          # batch
    ctypes.c_int64,                          # seq_len
]

_lib.lstm_weight_ih.restype = ctypes.POINTER(ctypes.c_double)
_lib.lstm_weight_ih.argtypes = [ctypes.c_void_p]
_lib.lstm_weight_hh.restype = ctypes.POINTER(ctypes.c_double)
_lib.lstm_weight_hh.argtypes = [ctypes.c_void_p]
_lib.lstm_bias_ih.restype = ctypes.POINTER(ctypes.c_double)
_lib.lstm_bias_ih.argtypes = [ctypes.c_void_p]
_lib.lstm_bias_hh.restype = ctypes.POINTER(ctypes.c_double)
_lib.lstm_bias_hh.argtypes = [ctypes.c_void_p]
_lib.lstm_input_size.restype = ctypes.c_int64
_lib.lstm_input_size.argtypes = [ctypes.c_void_p]
_lib.lstm_hidden_size.restype = ctypes.c_int64
_lib.lstm_hidden_size.argtypes = [ctypes.c_void_p]

_lib.gru_create.restype = ctypes.c_void_p
_lib.gru_create.argtypes = [ctypes.c_int64, ctypes.c_int64]

_lib.gru_free.restype = None
_lib.gru_free.argtypes = [ctypes.c_void_p]

_lib.gru_forward.restype = ctypes.c_int
_lib.gru_forward.argtypes = [
    ctypes.c_void_p,                         # layer
    ctypes.POINTER(ctypes.c_double),         # input
    ctypes.POINTER(ctypes.c_double),         # output
    ctypes.POINTER(ctypes.c_double),         # h_out
    ctypes.POINTER(ctypes.c_double),         # h_init (nullable)
    ctypes.c_int64,                          # batch
    ctypes.c_int64,                          # seq_len
]

_lib.gru_weight_ih.restype = ctypes.POINTER(ctypes.c_double)
_lib.gru_weight_ih.argtypes = [ctypes.c_void_p]
_lib.gru_weight_hh.restype = ctypes.POINTER(ctypes.c_double)
_lib.gru_weight_hh.argtypes = [ctypes.c_void_p]
_lib.gru_bias_ih.restype = ctypes.POINTER(ctypes.c_double)
_lib.gru_bias_ih.argtypes = [ctypes.c_void_p]
_lib.gru_bias_hh.restype = ctypes.POINTER(ctypes.c_double)
_lib.gru_bias_hh.argtypes = [ctypes.c_void_p]
_lib.gru_input_size.restype = ctypes.c_int64
_lib.gru_input_size.argtypes = [ctypes.c_void_p]
_lib.gru_hidden_size.restype = ctypes.c_int64
_lib.gru_hidden_size.argtypes = [ctypes.c_void_p]

# Loss functions
_lib.neural_mse_loss.restype = ctypes.c_int
_lib.neural_mse_loss.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
]

_lib.neural_cross_entropy_loss.restype = ctypes.c_int
_lib.neural_cross_entropy_loss.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
]

# Dataset
_lib.neural_dataset_load_csv.restype = ctypes.c_void_p
_lib.neural_dataset_load_csv.argtypes = [
    ctypes.c_char_p,   # data_path
    ctypes.c_char_p,   # labels_path
    ctypes.c_uint64,   # n_features
    ctypes.c_uint32,   # dtype
]

_lib.neural_dataset_free.restype = None
_lib.neural_dataset_free.argtypes = [ctypes.c_void_p]

_lib.neural_dataset_size.restype = ctypes.c_uint64
_lib.neural_dataset_size.argtypes = [ctypes.c_void_p]

_lib.neural_dataset_get_batch.restype = ctypes.c_int
_lib.neural_dataset_get_batch.argtypes = [
    ctypes.c_void_p,                         # dataset
    ctypes.c_uint64,                         # batch_index
    ctypes.c_uint64,                         # batch_size
    ctypes.POINTER(ctypes.c_void_p),         # out_x
    ctypes.POINTER(ctypes.c_void_p),         # out_y
]

# Config
_lib.neural_config_load.restype = ctypes.c_void_p
_lib.neural_config_load.argtypes = [ctypes.c_char_p]

_lib.neural_config_free.restype = None
_lib.neural_config_free.argtypes = [ctypes.c_void_p]

# --- Training Operations (training_ops.asm) ---

# Confusion Matrix
_lib.neural_confusion_matrix_update.restype = ctypes.c_int
_lib.neural_confusion_matrix_update.argtypes = [
    ctypes.c_void_p,   # int32_t* matrix
    ctypes.c_void_p,   # int32_t* targets
    ctypes.c_void_p,   # int32_t* predictions
    ctypes.c_uint64,   # n
    ctypes.c_uint64,   # num_classes
]

_lib.neural_compute_class_precision.restype = ctypes.c_double
_lib.neural_compute_class_precision.argtypes = [
    ctypes.c_void_p,   # int32_t* matrix
    ctypes.c_int,      # class_idx
    ctypes.c_int,      # num_classes
]

_lib.neural_compute_class_recall.restype = ctypes.c_double
_lib.neural_compute_class_recall.argtypes = [
    ctypes.c_void_p,   # int32_t* matrix
    ctypes.c_int,      # class_idx
    ctypes.c_int,      # num_classes
]

_lib.neural_compute_class_f1.restype = ctypes.c_double
_lib.neural_compute_class_f1.argtypes = [
    ctypes.c_void_p,   # int32_t* matrix
    ctypes.c_int,      # class_idx
    ctypes.c_int,      # num_classes
]

_lib.neural_compute_accuracy_from_matrix.restype = ctypes.c_double
_lib.neural_compute_accuracy_from_matrix.argtypes = [
    ctypes.c_void_p,   # int32_t* matrix
    ctypes.c_int,      # num_classes
]

# LR Schedules
_lib.neural_lr_step_decay.restype = ctypes.c_double
_lib.neural_lr_step_decay.argtypes = [
    ctypes.c_double,   # initial_lr (xmm0)
    ctypes.c_int,      # epoch (edi)
    ctypes.c_int,      # step_size (esi)
    ctypes.c_double,   # gamma (xmm1)
]

_lib.neural_lr_exponential_decay.restype = ctypes.c_double
_lib.neural_lr_exponential_decay.argtypes = [
    ctypes.c_double,   # initial_lr (xmm0)
    ctypes.c_int,      # epoch (edi)
    ctypes.c_double,   # gamma (xmm1)
]

_lib.neural_lr_cosine_annealing.restype = ctypes.c_double
_lib.neural_lr_cosine_annealing.argtypes = [
    ctypes.c_double,   # initial_lr (xmm0)
    ctypes.c_int,      # epoch (edi)
    ctypes.c_int,      # T_max (esi)
    ctypes.c_double,   # eta_min (xmm1)
]

# Tensor Inspection
_lib.neural_tensor_has_nan.restype = ctypes.c_int
_lib.neural_tensor_has_nan.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
]

_lib.neural_tensor_has_inf.restype = ctypes.c_int
_lib.neural_tensor_has_inf.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
]

_lib.neural_tensor_grad_l2_norm.restype = ctypes.c_float
_lib.neural_tensor_grad_l2_norm.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
]

# Dropout
_lib.neural_dropout_forward.restype = ctypes.c_int
_lib.neural_dropout_forward.argtypes = [
    ctypes.c_void_p,   # float* input
    ctypes.c_void_p,   # float* output
    ctypes.c_void_p,   # uint8_t* mask
    ctypes.c_uint64,   # num_elements
    ctypes.c_float,    # p (dropout probability)
]

_lib.neural_dropout_backward.restype = ctypes.c_int
_lib.neural_dropout_backward.argtypes = [
    ctypes.c_void_p,   # float* grad_output
    ctypes.c_void_p,   # float* grad_input
    ctypes.c_void_p,   # uint8_t* mask
    ctypes.c_uint64,   # num_elements
    ctypes.c_float,    # p
]

# Weight Initialization
_lib.neural_init_uniform_range.restype = ctypes.c_int
_lib.neural_init_uniform_range.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
    ctypes.c_float,    # lo (xmm0)
    ctypes.c_float,    # hi (xmm1)
    ctypes.c_int,      # seed (edx)
]

_lib.neural_init_normal_range.restype = ctypes.c_int
_lib.neural_init_normal_range.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
    ctypes.c_float,    # mean (xmm0)
    ctypes.c_float,    # std (xmm1)
    ctypes.c_int,      # seed (edx)
]

_lib.neural_init_he_uniform.restype = ctypes.c_int
_lib.neural_init_he_uniform.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
    ctypes.c_uint64,   # fan
    ctypes.c_int,      # seed
]

_lib.neural_init_xavier_uniform.restype = ctypes.c_int
_lib.neural_init_xavier_uniform.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
    ctypes.c_uint64,   # fan_in
    ctypes.c_uint64,   # fan_out
    ctypes.c_int,      # seed (r8d)
]

_lib.neural_init_he_normal.restype = ctypes.c_int
_lib.neural_init_he_normal.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
    ctypes.c_uint64,   # fan
    ctypes.c_int,      # seed
]

_lib.neural_init_xavier_normal.restype = ctypes.c_int
_lib.neural_init_xavier_normal.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
    ctypes.c_uint64,   # fan_in
    ctypes.c_uint64,   # fan_out
    ctypes.c_int,      # seed (r8d)
]

_lib.neural_init_kaiming_uniform.restype = ctypes.c_int
_lib.neural_init_kaiming_uniform.argtypes = [
    ctypes.c_void_p,   # float* data
    ctypes.c_uint64,   # num_elements
    ctypes.c_uint64,   # fan_in
    ctypes.c_uint64,   # fan_out
    ctypes.c_int,      # mode (0=fan_in, 1=fan_out)
    ctypes.c_int,      # seed
]

# Optimizer LR access
_lib.neural_optimizer_set_lr.restype = None
_lib.neural_optimizer_set_lr.argtypes = [
    ctypes.c_void_p,   # Optimizer* opt
    ctypes.c_double,   # lr
]

_lib.neural_optimizer_get_lr.restype = ctypes.c_double
_lib.neural_optimizer_get_lr.argtypes = [
    ctypes.c_void_p,   # Optimizer* opt
]

# Gradient clipping (optimizers.asm via neural_api.asm)
_lib.neural_clip_grad_norm.restype = ctypes.c_int
_lib.neural_clip_grad_norm.argtypes = [
    ctypes.c_void_p,   # NeuralOptimizer* opt
    ctypes.c_double,    # max_norm
]

# Class-balanced sampling
_lib.neural_compute_class_weights.restype = ctypes.c_int
_lib.neural_compute_class_weights.argtypes = [
    ctypes.c_void_p,   # int32_t* labels
    ctypes.c_uint64,   # n
    ctypes.c_uint64,   # num_classes
    ctypes.c_void_p,   # double* weights_out
]

_lib.neural_compute_sample_weights.restype = ctypes.c_int
_lib.neural_compute_sample_weights.argtypes = [
    ctypes.c_void_p,   # int32_t* labels
    ctypes.c_uint64,   # n
    ctypes.c_void_p,   # double* class_weights
    ctypes.c_uint64,   # num_classes
    ctypes.c_void_p,   # double* sample_weights_out
]

_lib.neural_weighted_sample_indices.restype = ctypes.c_int
_lib.neural_weighted_sample_indices.argtypes = [
    ctypes.c_void_p,   # double* sample_weights
    ctypes.c_uint64,   # n_in
    ctypes.c_uint64,   # n_out
    ctypes.c_void_p,   # int32_t* indices_out
    ctypes.c_int,      # seed
]

# ── TensorBoard logging (tb_logger.c) ──────────────────────────
_lib.tb_create_writer.restype = ctypes.c_int
_lib.tb_create_writer.argtypes = [ctypes.c_char_p]

_lib.tb_add_scalar.restype = ctypes.c_int
_lib.tb_add_scalar.argtypes = [
    ctypes.c_int,       # handle
    ctypes.c_char_p,    # tag
    ctypes.c_float,     # value
    ctypes.c_int64,     # step
]

_lib.tb_add_scalars.restype = ctypes.c_int
_lib.tb_add_scalars.argtypes = [
    ctypes.c_int,       # handle
    ctypes.c_char_p,    # tag_prefix
    ctypes.c_void_p,    # const char** subtags
    ctypes.c_void_p,    # const float* values
    ctypes.c_int,       # count
    ctypes.c_int64,     # step
]

_lib.tb_add_histogram_stats.restype = ctypes.c_int
_lib.tb_add_histogram_stats.argtypes = [
    ctypes.c_int,       # handle
    ctypes.c_char_p,    # tag
    ctypes.c_void_p,    # const float* data
    ctypes.c_int,       # count
    ctypes.c_int64,     # step
]

_lib.tb_flush.restype = ctypes.c_int
_lib.tb_flush.argtypes = [ctypes.c_int]

_lib.tb_close.restype = ctypes.c_int
_lib.tb_close.argtypes = [ctypes.c_int]

_lib.tb_get_logdir.restype = ctypes.c_char_p
_lib.tb_get_logdir.argtypes = [ctypes.c_int]

_lib.tb_get_filepath.restype = ctypes.c_char_p
_lib.tb_get_filepath.argtypes = [ctypes.c_int]

# ── Model pruning (pruning.c) ──────────────────────────────────
_lib.prune_magnitude.restype = ctypes.c_int64
_lib.prune_magnitude.argtypes = [
    ctypes.c_void_p,    # double* weights
    ctypes.c_void_p,    # uint8_t* mask (may be NULL)
    ctypes.c_int64,     # n
    ctypes.c_double,    # threshold
]

_lib.prune_topk.restype = ctypes.c_int64
_lib.prune_topk.argtypes = [
    ctypes.c_void_p,    # double* weights
    ctypes.c_void_p,    # uint8_t* mask (may be NULL)
    ctypes.c_int64,     # n
    ctypes.c_double,    # keep_ratio
]

_lib.prune_rows.restype = ctypes.c_int64
_lib.prune_rows.argtypes = [
    ctypes.c_void_p,    # double* weights
    ctypes.c_int64,     # rows
    ctypes.c_int64,     # cols
    ctypes.c_double,    # threshold
    ctypes.c_void_p,    # uint8_t* row_mask (may be NULL)
]

_lib.prune_cols.restype = ctypes.c_int64
_lib.prune_cols.argtypes = [
    ctypes.c_void_p,    # double* weights
    ctypes.c_int64,     # rows
    ctypes.c_int64,     # cols
    ctypes.c_double,    # threshold
    ctypes.c_void_p,    # uint8_t* col_mask (may be NULL)
]

_lib.compute_sparsity.restype = ctypes.c_double
_lib.compute_sparsity.argtypes = [ctypes.c_void_p, ctypes.c_int64]

_lib.count_nonzero.restype = ctypes.c_int64
_lib.count_nonzero.argtypes = [ctypes.c_void_p, ctypes.c_int64]

_lib.compute_threshold_for_sparsity.restype = ctypes.c_double
_lib.compute_threshold_for_sparsity.argtypes = [
    ctypes.c_void_p,    # const double* weights
    ctypes.c_int64,     # n
    ctypes.c_double,    # target_sparsity
]

_lib.apply_mask.restype = None
_lib.apply_mask.argtypes = [
    ctypes.c_void_p,    # double* weights
    ctypes.c_void_p,    # const uint8_t* mask
    ctypes.c_int64,     # n
]

# ── INT8 Quantization (quantize.c) ─────────────────────────────

# QuantParams struct: scale(f64), zero_point(i32), min_val(f64), max_val(f64), symmetric(i32)
class QuantParamsC(ctypes.Structure):
    _fields_ = [
        ("scale", ctypes.c_double),
        ("zero_point", ctypes.c_int32),
        ("min_val", ctypes.c_double),
        ("max_val", ctypes.c_double),
        ("symmetric", ctypes.c_int),
    ]

_lib.calibrate_minmax.restype = ctypes.c_int
_lib.calibrate_minmax.argtypes = [
    ctypes.c_void_p,                   # const double* data
    ctypes.c_int64,                    # n
    ctypes.c_int,                      # symmetric
    ctypes.POINTER(QuantParamsC),      # QuantParams* out
]

_lib.calibrate_percentile.restype = ctypes.c_int
_lib.calibrate_percentile.argtypes = [
    ctypes.c_void_p,                   # const double* data
    ctypes.c_int64,                    # n
    ctypes.c_double,                   # percentile
    ctypes.c_int,                      # symmetric
    ctypes.POINTER(QuantParamsC),      # QuantParams* out
]

_lib.quantize_tensor.restype = ctypes.c_int
_lib.quantize_tensor.argtypes = [
    ctypes.c_void_p,                   # const double* data
    ctypes.c_void_p,                   # int8_t* out
    ctypes.c_int64,                    # n
    ctypes.POINTER(QuantParamsC),      # const QuantParams* params
]

_lib.dequantize_tensor.restype = ctypes.c_int
_lib.dequantize_tensor.argtypes = [
    ctypes.c_void_p,                   # const int8_t* data
    ctypes.c_void_p,                   # double* out
    ctypes.c_int64,                    # n
    ctypes.POINTER(QuantParamsC),      # const QuantParams* params
]

_lib.quantized_matmul.restype = ctypes.c_int
_lib.quantized_matmul.argtypes = [
    ctypes.c_void_p,                   # const int8_t* A
    ctypes.c_void_p,                   # const int8_t* B
    ctypes.c_void_p,                   # double* C
    ctypes.c_int64,                    # M
    ctypes.c_int64,                    # K
    ctypes.c_int64,                    # N
    ctypes.POINTER(QuantParamsC),      # params_a
    ctypes.POINTER(QuantParamsC),      # params_b
]

_lib.quantization_error.restype = ctypes.c_double
_lib.quantization_error.argtypes = [
    ctypes.c_void_p,                   # const double* original
    ctypes.c_int64,                    # n
    ctypes.POINTER(QuantParamsC),      # const QuantParams* params
]

_lib.quantization_snr.restype = ctypes.c_double
_lib.quantization_snr.argtypes = [
    ctypes.c_void_p,                   # const double* original
    ctypes.c_int64,                    # n
    ctypes.POINTER(QuantParamsC),      # const QuantParams* params
]

# ── BatchNorm1d (batchnorm.c) ──────────────────────────────────────
_lib.batchnorm1d_create.restype = ctypes.c_void_p
_lib.batchnorm1d_create.argtypes = [ctypes.c_int64, ctypes.c_double, ctypes.c_double]

_lib.batchnorm1d_free.restype = None
_lib.batchnorm1d_free.argtypes = [ctypes.c_void_p]

_lib.batchnorm1d_forward.restype = ctypes.c_int
_lib.batchnorm1d_forward.argtypes = [
    ctypes.c_void_p,   # BatchNorm1d*
    ctypes.c_void_p,   # const double* input
    ctypes.c_void_p,   # double* output
    ctypes.c_int64,    # batch_size
    ctypes.c_int,      # training
]

_lib.batchnorm1d_backward.restype = ctypes.c_int
_lib.batchnorm1d_backward.argtypes = [
    ctypes.c_void_p,   # BatchNorm1d*
    ctypes.c_void_p,   # const double* grad_output
    ctypes.c_void_p,   # double* grad_input
    ctypes.c_void_p,   # double* grad_gamma
    ctypes.c_void_p,   # double* grad_beta
    ctypes.c_int64,    # batch_size
]

_lib.batchnorm1d_gamma.restype = ctypes.c_void_p
_lib.batchnorm1d_gamma.argtypes = [ctypes.c_void_p]

_lib.batchnorm1d_beta.restype = ctypes.c_void_p
_lib.batchnorm1d_beta.argtypes = [ctypes.c_void_p]

_lib.batchnorm1d_running_mean.restype = ctypes.c_void_p
_lib.batchnorm1d_running_mean.argtypes = [ctypes.c_void_p]

_lib.batchnorm1d_running_var.restype = ctypes.c_void_p
_lib.batchnorm1d_running_var.argtypes = [ctypes.c_void_p]

_lib.batchnorm1d_num_features.restype = ctypes.c_int64
_lib.batchnorm1d_num_features.argtypes = [ctypes.c_void_p]

# ── LayerNorm (batchnorm.c) ────────────────────────────────────────
_lib.layernorm_create.restype = ctypes.c_void_p
_lib.layernorm_create.argtypes = [ctypes.c_int64, ctypes.c_double]

_lib.layernorm_free.restype = None
_lib.layernorm_free.argtypes = [ctypes.c_void_p]

_lib.layernorm_forward.restype = ctypes.c_int
_lib.layernorm_forward.argtypes = [
    ctypes.c_void_p,   # LayerNorm*
    ctypes.c_void_p,   # const double* input
    ctypes.c_void_p,   # double* output
    ctypes.c_int64,    # batch_size
]

_lib.layernorm_backward.restype = ctypes.c_int
_lib.layernorm_backward.argtypes = [
    ctypes.c_void_p,   # LayerNorm*
    ctypes.c_void_p,   # const double* grad_output
    ctypes.c_void_p,   # double* grad_input
    ctypes.c_void_p,   # double* grad_gamma
    ctypes.c_void_p,   # double* grad_beta
    ctypes.c_int64,    # batch_size
]

_lib.layernorm_gamma.restype = ctypes.c_void_p
_lib.layernorm_gamma.argtypes = [ctypes.c_void_p]

_lib.layernorm_beta.restype = ctypes.c_void_p
_lib.layernorm_beta.argtypes = [ctypes.c_void_p]

_lib.layernorm_num_features.restype = ctypes.c_int64
_lib.layernorm_num_features.argtypes = [ctypes.c_void_p]

# ── Label-Smoothed Cross-Entropy (metrics_losses.c) ───────────────
_lib.label_smoothing_ce_forward.restype = ctypes.c_int
_lib.label_smoothing_ce_forward.argtypes = [
    ctypes.c_void_p,   # const double* logits
    ctypes.c_void_p,   # const int64_t* targets
    ctypes.c_int64,    # batch_size
    ctypes.c_int64,    # num_classes
    ctypes.c_double,   # smoothing
    ctypes.POINTER(ctypes.c_double),  # loss_out
]

_lib.label_smoothing_ce_backward.restype = ctypes.c_int
_lib.label_smoothing_ce_backward.argtypes = [
    ctypes.c_void_p,   # const double* logits
    ctypes.c_void_p,   # const int64_t* targets
    ctypes.c_int64,    # batch_size
    ctypes.c_int64,    # num_classes
    ctypes.c_double,   # smoothing
    ctypes.c_void_p,   # double* grad_out
]

# ── ROC-AUC Score (metrics_losses.c) ──────────────────────────────
_lib.roc_auc_score.restype = ctypes.c_int
_lib.roc_auc_score.argtypes = [
    ctypes.c_void_p,   # const double* y_true
    ctypes.c_void_p,   # const double* y_score
    ctypes.c_int64,    # n
    ctypes.POINTER(ctypes.c_double),  # auc_out
]

# ── Data Transforms (transforms.c) ────────────────────────────────
_lib.transform_compute_stats.restype = ctypes.c_int
_lib.transform_compute_stats.argtypes = [
    ctypes.c_void_p,   # const double* data
    ctypes.c_int64,    # batch_size
    ctypes.c_int64,    # num_features
    ctypes.c_void_p,   # double* mean_out (nullable)
    ctypes.c_void_p,   # double* std_out  (nullable)
    ctypes.c_void_p,   # double* min_out  (nullable)
    ctypes.c_void_p,   # double* max_out  (nullable)
]

_lib.transform_normalize.restype = ctypes.c_int
_lib.transform_normalize.argtypes = [
    ctypes.c_void_p,   # const double* data
    ctypes.c_void_p,   # double* output
    ctypes.c_int64,    # batch_size
    ctypes.c_int64,    # num_features
    ctypes.c_void_p,   # const double* mean
    ctypes.c_void_p,   # const double* std
    ctypes.c_double,   # eps
]

_lib.transform_unnormalize.restype = ctypes.c_int
_lib.transform_unnormalize.argtypes = [
    ctypes.c_void_p,   # const double* data
    ctypes.c_void_p,   # double* output
    ctypes.c_int64,    # batch_size
    ctypes.c_int64,    # num_features
    ctypes.c_void_p,   # const double* mean
    ctypes.c_void_p,   # const double* std
    ctypes.c_double,   # eps
]

_lib.transform_minmax.restype = ctypes.c_int
_lib.transform_minmax.argtypes = [
    ctypes.c_void_p,   # const double* data
    ctypes.c_void_p,   # double* output
    ctypes.c_int64,    # batch_size
    ctypes.c_int64,    # num_features
    ctypes.c_void_p,   # const double* min_val
    ctypes.c_void_p,   # const double* max_val
    ctypes.c_double,   # eps
]

# ── Embedding Layer (embedding.c) ─────────────────────────────────
_lib.embedding_create.restype = ctypes.c_void_p
_lib.embedding_create.argtypes = [ctypes.c_int64, ctypes.c_int64]

_lib.embedding_free.restype = None
_lib.embedding_free.argtypes = [ctypes.c_void_p]

_lib.embedding_forward.restype = ctypes.c_int
_lib.embedding_forward.argtypes = [
    ctypes.c_void_p,   # const Embedding*
    ctypes.c_void_p,   # const int64_t* indices
    ctypes.c_int64,    # seq_len
    ctypes.c_void_p,   # double* output
]

_lib.embedding_backward.restype = ctypes.c_int
_lib.embedding_backward.argtypes = [
    ctypes.c_void_p,   # const Embedding*
    ctypes.c_void_p,   # const int64_t* indices
    ctypes.c_int64,    # seq_len
    ctypes.c_void_p,   # const double* grad_output
    ctypes.c_void_p,   # double* grad_weight
]

_lib.embedding_weight.restype = ctypes.c_void_p
_lib.embedding_weight.argtypes = [ctypes.c_void_p]

_lib.embedding_num_embeddings.restype = ctypes.c_int64
_lib.embedding_num_embeddings.argtypes = [ctypes.c_void_p]

_lib.embedding_dim.restype = ctypes.c_int64
_lib.embedding_dim.argtypes = [ctypes.c_void_p]

# ── Fuzzy Logic Engine (fuzzy.c) ──────────────────────────────────

# Membership functions
_lib.fuzzy_triangular.restype = ctypes.c_double
_lib.fuzzy_triangular.argtypes = [
    ctypes.c_double,  # x
    ctypes.c_double,  # a
    ctypes.c_double,  # b
    ctypes.c_double,  # c
]

_lib.fuzzy_trapezoidal.restype = ctypes.c_double
_lib.fuzzy_trapezoidal.argtypes = [
    ctypes.c_double,  # x
    ctypes.c_double,  # a
    ctypes.c_double,  # b
    ctypes.c_double,  # c
    ctypes.c_double,  # d
]

_lib.fuzzy_gaussian.restype = ctypes.c_double
_lib.fuzzy_gaussian.argtypes = [
    ctypes.c_double,  # x
    ctypes.c_double,  # mean
    ctypes.c_double,  # sigma
]

# Fuzzy operators
_lib.fuzzy_and.restype = ctypes.c_double
_lib.fuzzy_and.argtypes = [ctypes.c_double, ctypes.c_double]

_lib.fuzzy_or.restype = ctypes.c_double
_lib.fuzzy_or.argtypes = [ctypes.c_double, ctypes.c_double]

_lib.fuzzy_not.restype = ctypes.c_double
_lib.fuzzy_not.argtypes = [ctypes.c_double]

# Defuzzification
_lib.fuzzy_defuzz_centroid.restype = ctypes.c_double
_lib.fuzzy_defuzz_centroid.argtypes = [
    ctypes.c_void_p,  # const double* values
    ctypes.c_void_p,  # const double* memberships
    ctypes.c_int64,   # n
]

_lib.fuzzy_defuzz_bisector.restype = ctypes.c_double
_lib.fuzzy_defuzz_bisector.argtypes = [
    ctypes.c_void_p,  # const double* values
    ctypes.c_void_p,  # const double* memberships
    ctypes.c_int64,   # n
]

_lib.fuzzy_defuzz_mom.restype = ctypes.c_double
_lib.fuzzy_defuzz_mom.argtypes = [
    ctypes.c_void_p,  # const double* values
    ctypes.c_void_p,  # const double* memberships
    ctypes.c_int64,   # n
]

# Fuzzy system lifecycle
_lib.fuzzy_system_create.restype = ctypes.c_void_p
_lib.fuzzy_system_create.argtypes = [
    ctypes.c_int,  # n_inputs
    ctypes.c_int,  # resolution
    ctypes.c_int,  # defuzz_method
]

_lib.fuzzy_system_free.restype = None
_lib.fuzzy_system_free.argtypes = [ctypes.c_void_p]

# System configuration
_lib.fuzzy_system_set_input_range.restype = ctypes.c_int
_lib.fuzzy_system_set_input_range.argtypes = [
    ctypes.c_void_p,  # FuzzySystem*
    ctypes.c_int,     # var_idx
    ctypes.c_double,  # lo
    ctypes.c_double,  # hi
]

_lib.fuzzy_system_set_output_range.restype = ctypes.c_int
_lib.fuzzy_system_set_output_range.argtypes = [
    ctypes.c_void_p,  # FuzzySystem*
    ctypes.c_double,  # lo
    ctypes.c_double,  # hi
]

_lib.fuzzy_system_add_input_mf.restype = ctypes.c_int
_lib.fuzzy_system_add_input_mf.argtypes = [
    ctypes.c_void_p,  # FuzzySystem*
    ctypes.c_int,     # var_idx
    ctypes.c_int,     # mf_type (0=tri, 1=trap, 2=gauss)
    ctypes.c_double,  # p0
    ctypes.c_double,  # p1
    ctypes.c_double,  # p2
    ctypes.c_double,  # p3
]

_lib.fuzzy_system_add_output_mf.restype = ctypes.c_int
_lib.fuzzy_system_add_output_mf.argtypes = [
    ctypes.c_void_p,  # FuzzySystem*
    ctypes.c_int,     # mf_type
    ctypes.c_double,  # p0
    ctypes.c_double,  # p1
    ctypes.c_double,  # p2
    ctypes.c_double,  # p3
]

_lib.fuzzy_system_add_rule.restype = ctypes.c_int
_lib.fuzzy_system_add_rule.argtypes = [
    ctypes.c_void_p,  # FuzzySystem*
    ctypes.c_void_p,  # const int* input_vars
    ctypes.c_void_p,  # const int* input_terms
    ctypes.c_int,     # n_antecedents
    ctypes.c_int,     # consequent_term
    ctypes.c_double,  # weight
]

_lib.fuzzy_system_evaluate.restype = ctypes.c_int
_lib.fuzzy_system_evaluate.argtypes = [
    ctypes.c_void_p,  # const FuzzySystem*
    ctypes.c_void_p,  # const double* inputs
    ctypes.c_void_p,  # double* output
]

# System accessors
_lib.fuzzy_system_n_inputs.restype = ctypes.c_int
_lib.fuzzy_system_n_inputs.argtypes = [ctypes.c_void_p]

_lib.fuzzy_system_n_rules.restype = ctypes.c_int
_lib.fuzzy_system_n_rules.argtypes = [ctypes.c_void_p]

_lib.fuzzy_system_resolution.restype = ctypes.c_int
_lib.fuzzy_system_resolution.argtypes = [ctypes.c_void_p]

_lib.fuzzy_system_defuzz_method.restype = ctypes.c_int
_lib.fuzzy_system_defuzz_method.argtypes = [ctypes.c_void_p]

# ── Conv2D Layer (conv2d.c) ───────────────────────────────────────

_lib.conv2d_output_size.restype = ctypes.c_int64
_lib.conv2d_output_size.argtypes = [
    ctypes.c_int64,  # input_dim
    ctypes.c_int64,  # kernel_dim
    ctypes.c_int64,  # stride
    ctypes.c_int64,  # padding
]

_lib.conv2d_layer_create.restype = ctypes.c_void_p
_lib.conv2d_layer_create.argtypes = [
    ctypes.c_int64,  # in_channels
    ctypes.c_int64,  # out_channels
    ctypes.c_int64,  # kernel_h
    ctypes.c_int64,  # kernel_w
    ctypes.c_int64,  # stride
    ctypes.c_int64,  # padding
    ctypes.c_int,    # has_bias
]

_lib.conv2d_layer_free.restype = None
_lib.conv2d_layer_free.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_forward.restype = ctypes.c_int
_lib.conv2d_layer_forward.argtypes = [
    ctypes.c_void_p,  # Conv2DLayer*
    ctypes.c_void_p,  # const double* input
    ctypes.c_int64,   # batch
    ctypes.c_int64,   # in_h
    ctypes.c_int64,   # in_w
    ctypes.c_void_p,  # double* output
]

_lib.conv2d_layer_backward.restype = ctypes.c_int
_lib.conv2d_layer_backward.argtypes = [
    ctypes.c_void_p,  # Conv2DLayer*
    ctypes.c_void_p,  # const double* input
    ctypes.c_void_p,  # const double* grad_output
    ctypes.c_int64,   # batch
    ctypes.c_int64,   # in_h
    ctypes.c_int64,   # in_w
    ctypes.c_void_p,  # double* grad_input
    ctypes.c_void_p,  # double* grad_weight
    ctypes.c_void_p,  # double* grad_bias
]

_lib.conv2d_layer_weight.restype = ctypes.c_void_p
_lib.conv2d_layer_weight.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_bias.restype = ctypes.c_void_p
_lib.conv2d_layer_bias.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_out_h.restype = ctypes.c_int64
_lib.conv2d_layer_out_h.argtypes = [ctypes.c_void_p, ctypes.c_int64]

_lib.conv2d_layer_out_w.restype = ctypes.c_int64
_lib.conv2d_layer_out_w.argtypes = [ctypes.c_void_p, ctypes.c_int64]

_lib.conv2d_layer_weight_size.restype = ctypes.c_int64
_lib.conv2d_layer_weight_size.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_in_channels.restype = ctypes.c_int64
_lib.conv2d_layer_in_channels.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_out_channels.restype = ctypes.c_int64
_lib.conv2d_layer_out_channels.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_kernel_h.restype = ctypes.c_int64
_lib.conv2d_layer_kernel_h.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_kernel_w.restype = ctypes.c_int64
_lib.conv2d_layer_kernel_w.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_stride.restype = ctypes.c_int64
_lib.conv2d_layer_stride.argtypes = [ctypes.c_void_p]

_lib.conv2d_layer_padding.restype = ctypes.c_int64
_lib.conv2d_layer_padding.argtypes = [ctypes.c_void_p]

# ── MaxPool2D (conv2d.c) ─────────────────────────────────────────

_lib.maxpool2d_forward.restype = ctypes.c_int
_lib.maxpool2d_forward.argtypes = [
    ctypes.c_void_p,  # const double* input
    ctypes.c_int64,   # batch
    ctypes.c_int64,   # channels
    ctypes.c_int64,   # in_h
    ctypes.c_int64,   # in_w
    ctypes.c_int64,   # pool_h
    ctypes.c_int64,   # pool_w
    ctypes.c_int64,   # stride
    ctypes.c_int64,   # padding
    ctypes.c_void_p,  # double* output
    ctypes.c_void_p,  # int64_t* mask
]

_lib.maxpool2d_backward.restype = ctypes.c_int
_lib.maxpool2d_backward.argtypes = [
    ctypes.c_void_p,  # const double* grad_output
    ctypes.c_void_p,  # const int64_t* mask
    ctypes.c_int64,   # batch
    ctypes.c_int64,   # channels
    ctypes.c_int64,   # in_h
    ctypes.c_int64,   # in_w
    ctypes.c_int64,   # out_h
    ctypes.c_int64,   # out_w
    ctypes.c_void_p,  # double* grad_input
]


def _check_error(result: int, operation: str = "operation"):
    """Check result code and raise exception if error."""
    if result != NeuralError.OK:
        raise NeuralException(result, f"{operation} failed")


# Public API functions

_initialized = False


def init() -> None:
    """Initialize the Neural Assembly framework.
    
    Must be called before using any other functions.
    
    Raises:
        NeuralException: If initialization fails
    """
    global _initialized
    if _initialized:
        return
    
    result = _lib.neural_init()
    if result != NeuralError.OK:
        raise NeuralException(result, "Failed to initialize framework")
    _initialized = True


def shutdown() -> None:
    """Shutdown the Neural Assembly framework and free resources."""
    global _initialized
    if _initialized:
        _lib.neural_shutdown()
        _initialized = False


def version() -> str:
    """Get the framework version string."""
    return _lib.neural_version().decode("utf-8")


def get_last_error() -> int:
    """Get the last error code."""
    return _lib.neural_get_last_error()


def get_error_message(error_code: int) -> str:
    """Get the error message for an error code."""
    msg = _lib.neural_get_error_message(error_code)
    if msg:
        return msg.decode("utf-8")
    return f"Unknown error ({error_code})"


def clear_error() -> None:
    """Clear the last error."""
    _lib.neural_clear_error()


def get_simd_level() -> int:
    """Get the available SIMD level.
    
    Returns:
        0: Scalar (no SIMD)
        1: SSE2
        2: AVX
        3: AVX2
        4: AVX-512
    """
    return _lib.neural_get_simd_level()


def get_simd_name() -> str:
    """Get the SIMD level name as a string."""
    return _lib.neural_get_simd_name().decode("utf-8")


# Auto-initialize when imported
try:
    init()
except Exception:
    pass  # Will fail if library not found, user can call init() manually
