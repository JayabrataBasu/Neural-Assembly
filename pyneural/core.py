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

# Linear layer
_lib.neural_linear_create.restype = ctypes.c_void_p
_lib.neural_linear_create.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]

_lib.neural_linear_free.restype = None
_lib.neural_linear_free.argtypes = [ctypes.c_void_p]

_lib.neural_linear_forward.restype = ctypes.c_int
_lib.neural_linear_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

# Optimizers
_lib.neural_sgd_create.restype = ctypes.c_void_p
_lib.neural_sgd_create.argtypes = [ctypes.c_double, ctypes.c_double]

_lib.neural_adam_create.restype = ctypes.c_void_p
_lib.neural_adam_create.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]

_lib.neural_optimizer_free.restype = None
_lib.neural_optimizer_free.argtypes = [ctypes.c_void_p]

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
