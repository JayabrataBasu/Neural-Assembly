"""
Tensor class for PyNeural.
"""

import ctypes
from typing import List, Sequence, Union, Optional
import array

from .core import (
    _lib,
    NeuralError,
    NeuralDtype,
    NeuralException,
    _check_error,
)

# Try to import numpy for interop
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Buffer info structure for NumPy buffer protocol
class NeuralBufferInfo(ctypes.Structure):
    """Buffer info structure for NumPy integration."""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("itemsize", ctypes.c_uint64),
        ("ndim", ctypes.c_uint64),
        ("shape", ctypes.POINTER(ctypes.c_uint64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("readonly", ctypes.c_int),
        ("format", ctypes.c_char * 8),
    ]


# Set up additional function signatures for zero-copy support
_lib.neural_tensor_from_buffer.restype = ctypes.c_void_p
_lib.neural_tensor_from_buffer.argtypes = [
    ctypes.c_void_p,                          # buffer
    ctypes.POINTER(ctypes.c_uint64),          # shape
    ctypes.c_uint64,                          # ndim
    ctypes.c_int,                             # dtype
    ctypes.POINTER(ctypes.c_int64),           # strides (or NULL)
]

_lib.neural_buffer_info.restype = ctypes.c_int
_lib.neural_buffer_info.argtypes = [
    ctypes.c_void_p,                          # tensor
    ctypes.POINTER(NeuralBufferInfo),         # info out
]

_lib.neural_tensor_is_contiguous.restype = ctypes.c_int
_lib.neural_tensor_is_contiguous.argtypes = [ctypes.c_void_p]

_lib.neural_tensor_stride.restype = ctypes.POINTER(ctypes.c_int64)
_lib.neural_tensor_stride.argtypes = [ctypes.c_void_p]


class Tensor:
    """
    Multi-dimensional array for neural network computations.
    
    Attributes:
        shape: Tuple of dimension sizes
        ndim: Number of dimensions
        dtype: Data type (float32 or float64)
        numel: Total number of elements
    
    Example:
        >>> t = Tensor.zeros([2, 3])
        >>> t.shape
        (2, 3)
        >>> t.fill(1.0)
        >>> print(t.numpy())
        [[1. 1. 1.]
         [1. 1. 1.]]
    """
    
    def __init__(self, ptr: ctypes.c_void_p, owns_data: bool = True):
        """
        Create a Tensor wrapper around a native tensor pointer.
        
        Args:
            ptr: Pointer to native NeuralTensor
            owns_data: Whether this wrapper owns the data (will free on del)
        """
        if ptr is None or ptr == 0:
            raise NeuralException(NeuralError.NULL_POINTER, "Null tensor pointer")
        self._ptr = ptr
        self._owns_data = owns_data
    
    def __del__(self):
        """Free the native tensor when garbage collected."""
        if hasattr(self, "_ptr") and self._ptr and self._owns_data:
            _lib.neural_tensor_free(self._ptr)
            self._ptr = None
    
    @classmethod
    def create(
        cls,
        shape: Sequence[int],
        dtype: int = NeuralDtype.FLOAT32
    ) -> "Tensor":
        """
        Create a new tensor with uninitialized data.
        
        Args:
            shape: Sequence of dimension sizes
            dtype: Data type (NeuralDtype.FLOAT32 or FLOAT64)
        
        Returns:
            New Tensor instance
        """
        shape_arr = (ctypes.c_uint64 * len(shape))(*shape)
        ptr = _lib.neural_tensor_create(shape_arr, len(shape), dtype)
        if ptr is None or ptr == 0:
            raise NeuralException(_lib.neural_get_last_error())
        return cls(ptr)
    
    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        dtype: int = NeuralDtype.FLOAT32
    ) -> "Tensor":
        """
        Create a tensor filled with zeros.
        
        Args:
            shape: Sequence of dimension sizes
            dtype: Data type
        
        Returns:
            New Tensor instance filled with zeros
        """
        shape_arr = (ctypes.c_uint64 * len(shape))(*shape)
        ptr = _lib.neural_tensor_zeros(shape_arr, len(shape), dtype)
        if ptr is None or ptr == 0:
            raise NeuralException(_lib.neural_get_last_error())
        return cls(ptr)
    
    @classmethod
    def ones(
        cls,
        shape: Sequence[int],
        dtype: int = NeuralDtype.FLOAT32
    ) -> "Tensor":
        """
        Create a tensor filled with ones.
        
        Args:
            shape: Sequence of dimension sizes
            dtype: Data type
        
        Returns:
            New Tensor instance filled with ones
        """
        shape_arr = (ctypes.c_uint64 * len(shape))(*shape)
        ptr = _lib.neural_tensor_ones(shape_arr, len(shape), dtype)
        if ptr is None or ptr == 0:
            raise NeuralException(_lib.neural_get_last_error())
        return cls(ptr)
    
    @classmethod
    def from_numpy(cls, arr: "np.ndarray") -> "Tensor":
        """
        Create a tensor from a NumPy array (copies data).
        
        Args:
            arr: NumPy array to convert
        
        Returns:
            New Tensor instance with copied data
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for from_numpy()")
        
        # Ensure contiguous and correct dtype
        if arr.dtype == np.float32:
            dtype = NeuralDtype.FLOAT32
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        elif arr.dtype == np.float64:
            dtype = NeuralDtype.FLOAT64
            arr = np.ascontiguousarray(arr, dtype=np.float64)
        else:
            # Convert to float32 by default
            dtype = NeuralDtype.FLOAT32
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        
        # Create tensor with same shape
        tensor = cls.create(arr.shape, dtype)
        
        # Copy data
        data_ptr = _lib.neural_tensor_data(tensor._ptr)
        if data_ptr is None:
            raise NeuralException(NeuralError.NULL_POINTER, "Tensor data is null")
        
        # Copy bytes from numpy array to tensor
        nbytes = arr.nbytes
        ctypes.memmove(data_ptr, arr.ctypes.data, nbytes)
        
        return tensor
    
    @classmethod
    def from_numpy_zero_copy(cls, arr: "np.ndarray") -> "Tensor":
        """
        Create a tensor from a NumPy array without copying data (zero-copy).
        
        The tensor directly wraps the NumPy array's buffer. The caller must
        ensure the NumPy array remains alive for the lifetime of the tensor.
        
        Args:
            arr: NumPy array to wrap (must be contiguous and float32/float64)
        
        Returns:
            New Tensor instance sharing memory with the array
        
        Raises:
            ValueError: If array is not contiguous or has unsupported dtype
            
        Example:
            >>> arr = np.random.randn(100, 100).astype(np.float32)
            >>> t = Tensor.from_numpy_zero_copy(arr)
            >>> # Tensor shares memory with arr - modifications affect both
            >>> arr[0, 0] = 42.0
            >>> assert t.numpy()[0, 0] == 42.0
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for from_numpy_zero_copy()")
        
        # Check dtype
        if arr.dtype == np.float32:
            dtype = NeuralDtype.FLOAT32
        elif arr.dtype == np.float64:
            dtype = NeuralDtype.FLOAT64
        else:
            raise ValueError(f"Unsupported dtype {arr.dtype}. Must be float32 or float64.")
        
        # Check contiguity
        if not arr.flags['C_CONTIGUOUS']:
            raise ValueError("Array must be C-contiguous for zero-copy. Use np.ascontiguousarray() first.")
        
        # Get shape
        shape = arr.shape
        shape_arr = (ctypes.c_uint64 * len(shape))(*shape)
        
        # Get buffer pointer
        buffer_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        
        # Create tensor wrapping the buffer (NULL strides = C-contiguous)
        ptr = _lib.neural_tensor_from_buffer(
            buffer_ptr,
            shape_arr,
            len(shape),
            dtype,
            None  # NULL strides for C-contiguous
        )
        
        if ptr is None or ptr == 0:
            raise NeuralException(_lib.neural_get_last_error(), "Failed to create zero-copy tensor")
        
        # Create tensor wrapper - owns_data=False since buffer is external
        tensor = cls(ptr, owns_data=True)  # Still True because struct needs freeing
        # Keep reference to original array to prevent garbage collection
        tensor._numpy_ref = arr
        
        return tensor
    
    @property
    def __array_interface__(self) -> dict:
        """
        NumPy array interface for zero-copy array creation from tensor.
        
        This allows NumPy to create an array that shares memory with the tensor.
        
        Example:
            >>> t = Tensor.zeros([2, 3])
            >>> arr = np.asarray(t)  # Zero-copy if possible
            >>> arr[0, 0] = 42.0
            >>> # Changes visible in tensor
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for __array_interface__")
        
        data_ptr = _lib.neural_tensor_data(self._ptr)
        if data_ptr is None or data_ptr == 0:
            raise NeuralException(NeuralError.NULL_POINTER, "Tensor data is null")
        
        # Determine typestr
        if self.dtype == NeuralDtype.FLOAT32:
            typestr = "<f4"  # little-endian float32
        else:
            typestr = "<f8"  # little-endian float64
        
        # Get strides in bytes
        strides = self.strides
        
        return {
            "version": 3,
            "data": (data_ptr, False),  # (ptr, read-only)
            "shape": self.shape,
            "typestr": typestr,
            "strides": strides,
        }
    
    @property
    def strides(self) -> tuple:
        """Get the strides of the tensor in bytes."""
        stride_ptr = _lib.neural_tensor_stride(self._ptr)
        if stride_ptr is None:
            # Compute default C-contiguous strides
            shape = self.shape
            itemsize = 4 if self.dtype == NeuralDtype.FLOAT32 else 8
            strides = []
            stride = itemsize
            for dim in reversed(shape):
                strides.insert(0, stride)
                stride *= dim
            return tuple(strides)
        return tuple(stride_ptr[i] for i in range(self.ndim))
    
    @property
    def is_contiguous(self) -> bool:
        """Check if the tensor is C-contiguous."""
        result = _lib.neural_tensor_is_contiguous(self._ptr)
        return result == 1
    
    def numpy_zero_copy(self) -> "np.ndarray":
        """
        Create a NumPy array that shares memory with this tensor (zero-copy).
        
        The array directly references the tensor's data buffer. Modifications
        to either the array or tensor will affect the other.
        
        Returns:
            NumPy array sharing memory with the tensor
            
        Warning:
            The returned array is only valid while the tensor exists.
            Do not use after the tensor is deleted.
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for numpy_zero_copy()")
        
        # Use the array interface to create a zero-copy view
        return np.asarray(self)
    
    @classmethod
    def from_list(
        cls,
        data: List,
        dtype: int = NeuralDtype.FLOAT32
    ) -> "Tensor":
        """
        Create a tensor from a nested Python list.
        
        Args:
            data: Nested list of numbers
            dtype: Data type
        
        Returns:
            New Tensor instance
        """
        if HAS_NUMPY:
            arr = np.array(data, dtype=np.float32 if dtype == NeuralDtype.FLOAT32 else np.float64)
            return cls.from_numpy(arr)
        
        # Manual flattening for non-numpy case
        def get_shape(lst):
            shape = []
            while isinstance(lst, list):
                shape.append(len(lst))
                lst = lst[0] if lst else []
            return shape
        
        def flatten(lst):
            if isinstance(lst, list):
                for item in lst:
                    yield from flatten(item)
            else:
                yield lst
        
        shape = get_shape(data)
        flat_data = list(flatten(data))
        
        tensor = cls.create(shape, dtype)
        
        # Fill data
        data_ptr = _lib.neural_tensor_data(tensor._ptr)
        if data_ptr is None:
            raise NeuralException(NeuralError.NULL_POINTER)
        
        # Create ctypes array and copy
        if dtype == NeuralDtype.FLOAT32:
            arr_type = ctypes.c_float * len(flat_data)
        else:
            arr_type = ctypes.c_double * len(flat_data)
        
        arr = arr_type(*flat_data)
        ctypes.memmove(data_ptr, arr, ctypes.sizeof(arr))
        
        return tensor
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the tensor."""
        ndim = _lib.neural_tensor_ndim(self._ptr)
        if ndim == 0:
            return ()
        shape_ptr = _lib.neural_tensor_shape(self._ptr)
        return tuple(shape_ptr[i] for i in range(ndim))
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return _lib.neural_tensor_ndim(self._ptr)
    
    @property
    def dtype(self) -> int:
        """Get the data type."""
        return _lib.neural_tensor_dtype(self._ptr)
    
    @property
    def numel(self) -> int:
        """Get the total number of elements."""
        return _lib.neural_tensor_numel(self._ptr)
    
    @property
    def nbytes(self) -> int:
        """Get the total size in bytes."""
        return _lib.neural_tensor_bytes(self._ptr)
    
    @property
    def data_ptr(self) -> int:
        """Get the raw data pointer as an integer."""
        return _lib.neural_tensor_data(self._ptr)
    
    def fill(self, value: float) -> "Tensor":
        """
        Fill the tensor with a scalar value.
        
        Args:
            value: Value to fill with
        
        Returns:
            self (for chaining)
        """
        result = _lib.neural_tensor_fill(self._ptr, ctypes.c_double(value))
        _check_error(result, "fill")
        return self
    
    def copy(self) -> "Tensor":
        """
        Create a deep copy of this tensor.
        
        Returns:
            New Tensor with copied data
        """
        new_tensor = Tensor.create(self.shape, self.dtype)
        result = _lib.neural_tensor_copy(new_tensor._ptr, self._ptr)
        _check_error(result, "copy")
        return new_tensor
    
    def detach(self) -> "Tensor":
        """
        Create a new tensor with the same data but no gradient history.
        
        The returned tensor shares data with the original but operations on it
        will not be tracked for gradients. This is useful for inference or
        when you want to use tensor values without backpropagating through them.
        
        Returns:
            New Tensor that shares data but without gradient tracking
            
        Example:
            >>> x = Tensor.from_list([[1, 2], [3, 4]], requires_grad=True)
            >>> y = x.detach()  # y shares data but no grad tracking
            >>> # Operations on y won't affect x's gradient computation
        """
        # Create a view that shares data but with requires_grad=False
        detached = Tensor(self._ptr, owns_data=False)
        detached._requires_grad = False
        detached._grad_fn = None
        # Store a reference to keep the original alive
        detached._base = self
        return detached
    
    def detach_(self) -> "Tensor":
        """
        Detach this tensor in-place, removing its gradient history.
        
        After this call, operations on this tensor will not be tracked
        for gradients. This is an in-place operation.
        
        Returns:
            self (for chaining)
            
        Example:
            >>> x = Tensor.from_list([1, 2, 3], requires_grad=True)
            >>> x.detach_()  # x no longer tracks gradients
        """
        self._requires_grad = False
        self._grad_fn = None
        return self
    
    @property
    def grad_fn(self):
        """Get the gradient function (for autograd)."""
        return getattr(self, "_grad_fn", None)
    
    @grad_fn.setter
    def grad_fn(self, fn):
        """Set the gradient function."""
        self._grad_fn = fn
    
    @property
    def grad(self):
        """Get the accumulated gradient tensor."""
        return getattr(self, "_grad", None)
    
    @grad.setter
    def grad(self, value):
        """Set the gradient tensor."""
        self._grad = value
    
    def retain_grad(self) -> None:
        """
        Enable gradient retention for this tensor.
        
        By default, gradients for non-leaf tensors are not retained.
        Call this to keep the gradient after backward().
        """
        self._retain_grad = True
    
    @property
    def is_leaf(self) -> bool:
        """
        Check if this tensor is a leaf in the computation graph.
        
        A tensor is a leaf if it was created directly by the user
        (not as a result of an operation) and requires_grad is True.
        """
        return self.grad_fn is None and self.requires_grad
    
    @property
    def requires_grad(self) -> bool:
        """Check if this tensor requires gradient computation."""
        # Get requires_grad flag from tensor struct (if implemented in native)
        # For now, return False as tensors don't track by default
        return getattr(self, "_requires_grad", False)
    
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set whether this tensor requires gradient computation."""
        self._requires_grad = value
    
    def numpy(self) -> "np.ndarray":
        """
        Convert to a NumPy array (copies data).
        
        Returns:
            NumPy array with copied data
        """
        if not HAS_NUMPY:
            raise ImportError("NumPy is required for numpy()")
        
        # Get data pointer and size
        data_ptr = _lib.neural_tensor_data(self._ptr)
        numel = self.numel
        
        # Create numpy array with correct dtype
        if self.dtype == NeuralDtype.FLOAT32:
            np_dtype = np.float32
            c_type = ctypes.c_float
        else:
            np_dtype = np.float64
            c_type = ctypes.c_double
        
        # Create buffer and copy
        arr = np.empty(numel, dtype=np_dtype)
        src = ctypes.cast(data_ptr, ctypes.POINTER(c_type))
        ctypes.memmove(arr.ctypes.data, src, arr.nbytes)
        
        return arr.reshape(self.shape)
    
    def tolist(self) -> List:
        """Convert to a nested Python list."""
        if HAS_NUMPY:
            return self.numpy().tolist()
        
        # Manual conversion
        data_ptr = _lib.neural_tensor_data(self._ptr)
        numel = self.numel
        
        if self.dtype == NeuralDtype.FLOAT32:
            arr_type = ctypes.c_float * numel
        else:
            arr_type = ctypes.c_double * numel
        
        arr = ctypes.cast(data_ptr, ctypes.POINTER(arr_type)).contents
        flat = list(arr)
        
        # Reshape
        def reshape(flat, shape):
            if len(shape) == 0:
                return flat[0] if flat else 0.0
            if len(shape) == 1:
                return flat[:shape[0]]
            
            chunk_size = 1
            for dim in shape[1:]:
                chunk_size *= dim
            
            result = []
            for i in range(shape[0]):
                start = i * chunk_size
                end = start + chunk_size
                result.append(reshape(flat[start:end], shape[1:]))
            return result
        
        return reshape(flat, self.shape)
    
    # Arithmetic operations
    def __add__(self, other: "Tensor") -> "Tensor":
        """Element-wise addition."""
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot add Tensor and {type(other)}")
        
        result = Tensor.create(self.shape, self.dtype)
        err = _lib.neural_add(result._ptr, self._ptr, other._ptr)
        _check_error(err, "add")
        return result
    
    def __sub__(self, other: "Tensor") -> "Tensor":
        """Element-wise subtraction."""
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot subtract Tensor and {type(other)}")
        
        result = Tensor.create(self.shape, self.dtype)
        err = _lib.neural_sub(result._ptr, self._ptr, other._ptr)
        _check_error(err, "sub")
        return result
    
    def __mul__(self, other: "Tensor") -> "Tensor":
        """Element-wise multiplication."""
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot multiply Tensor and {type(other)}")
        
        result = Tensor.create(self.shape, self.dtype)
        err = _lib.neural_mul(result._ptr, self._ptr, other._ptr)
        _check_error(err, "mul")
        return result
    
    def __truediv__(self, other: "Tensor") -> "Tensor":
        """Element-wise division."""
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot divide Tensor and {type(other)}")
        
        result = Tensor.create(self.shape, self.dtype)
        err = _lib.neural_div(result._ptr, self._ptr, other._ptr)
        _check_error(err, "div")
        return result
    
    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication."""
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot matmul Tensor and {type(other)}")
        
        # Compute output shape
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul requires 2D tensors")
        
        m, k1 = self.shape
        k2, n = other.shape
        if k1 != k2:
            raise ValueError(f"Shape mismatch: {self.shape} @ {other.shape}")
        
        result = Tensor.create([m, n], self.dtype)
        err = _lib.neural_matmul(result._ptr, self._ptr, other._ptr)
        _check_error(err, "matmul")
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tensor(shape={self.shape}, dtype={'float32' if self.dtype == 0 else 'float64'})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        if HAS_NUMPY:
            return str(self.numpy())
        return repr(self)


def zeros(shape: Sequence[int], dtype: int = NeuralDtype.FLOAT32) -> Tensor:
    """Create a tensor filled with zeros."""
    return Tensor.zeros(shape, dtype)


def ones(shape: Sequence[int], dtype: int = NeuralDtype.FLOAT32) -> Tensor:
    """Create a tensor filled with ones."""
    return Tensor.ones(shape, dtype)


def tensor(
    data: Union[List, "np.ndarray"],
    dtype: int = NeuralDtype.FLOAT32
) -> Tensor:
    """
    Create a tensor from data.
    
    Args:
        data: List or NumPy array
        dtype: Data type
    
    Returns:
        New Tensor
    """
    if HAS_NUMPY and isinstance(data, np.ndarray):
        return Tensor.from_numpy(data)
    return Tensor.from_list(data, dtype)
