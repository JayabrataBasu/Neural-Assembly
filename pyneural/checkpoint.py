"""
Checkpoint save/resume for PyNeural models.

Binary format (v1):
  HEADER  (16 bytes)
    magic       : 8 bytes  "NEURCKPT"
    version     : uint32   1
    num_tensors : uint32   count of weight tensors

  METADATA (variable)
    epoch       : uint32
    best_loss   : float64
    lr          : float64

  For each tensor:
    ndim        : uint32
    shape       : ndim × uint64
    dtype       : uint32   (0=float32, 1=float64)
    data        : numel × sizeof(dtype) bytes
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import Optional

from .core import _lib, NeuralDtype

MAGIC = b"NEURCKPT"
VERSION = 1


def _tensor_raw_bytes(tensor) -> bytes:
    """Extract raw bytes from a tensor."""
    numel = int(_lib.neural_tensor_numel(tensor._ptr))
    dtype = int(_lib.neural_tensor_dtype(tensor._ptr))
    data_ptr = _lib.neural_tensor_data(tensor._ptr)

    elem_size = 4 if dtype == NeuralDtype.FLOAT32 else 8
    total = numel * elem_size
    buf = (ctypes.c_uint8 * total).from_address(data_ptr)
    return bytes(buf)


def _tensor_shape(tensor) -> tuple:
    """Return (ndim, shape_tuple, dtype_int) for a tensor."""
    ndim = int(_lib.neural_tensor_ndim(tensor._ptr))
    shape_ptr = _lib.neural_tensor_shape(tensor._ptr)
    shape = tuple(int(shape_ptr[i]) for i in range(ndim))
    dtype = int(_lib.neural_tensor_dtype(tensor._ptr))
    return ndim, shape, dtype


def _write_tensor(f, tensor) -> None:
    ndim, shape, dtype = _tensor_shape(tensor)
    f.write(struct.pack("<I", ndim))
    for s in shape:
        f.write(struct.pack("<Q", s))
    f.write(struct.pack("<I", dtype))
    f.write(_tensor_raw_bytes(tensor))


def _read_tensor_into(f, tensor) -> None:
    """Read tensor data from file and write into existing tensor buffer."""
    ndim_bytes = f.read(4)
    if len(ndim_bytes) < 4:
        raise IOError("Truncated checkpoint: expected ndim")
    ndim = struct.unpack("<I", ndim_bytes)[0]

    shape = []
    for _ in range(ndim):
        s = struct.unpack("<Q", f.read(8))[0]
        shape.append(s)

    dtype = struct.unpack("<I", f.read(4))[0]

    # Verify shape matches
    t_ndim, t_shape, t_dtype = _tensor_shape(tensor)
    if t_ndim != ndim or t_shape != tuple(shape) or t_dtype != dtype:
        raise ValueError(
            f"Checkpoint shape mismatch: file has ({ndim}, {shape}, dtype={dtype}) "
            f"but model has ({t_ndim}, {t_shape}, dtype={t_dtype})"
        )

    numel = 1
    for s in shape:
        numel *= s
    elem_size = 4 if dtype == NeuralDtype.FLOAT32 else 8
    total = numel * elem_size
    raw = f.read(total)
    if len(raw) < total:
        raise IOError("Truncated checkpoint: expected tensor data")

    data_ptr = _lib.neural_tensor_data(tensor._ptr)
    ctypes.memmove(data_ptr, raw, total)


def save_checkpoint(
    filepath: str,
    model,
    epoch: int = 0,
    best_loss: float = float("inf"),
    lr: float = 0.0,
) -> None:
    """
    Save model parameters and training metadata to a binary checkpoint.

    Args:
        filepath: Path to the output file.
        model: A Module (typically Sequential) whose parameters() to save.
        epoch: Current training epoch.
        best_loss: Best validation/training loss so far.
        lr: Current learning rate.
    """
    params = model.parameters()

    with open(filepath, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(params)))

        # Metadata
        f.write(struct.pack("<I", epoch))
        f.write(struct.pack("<d", best_loss))
        f.write(struct.pack("<d", lr))

        # Tensors
        for p in params:
            _write_tensor(f, p)


def load_checkpoint(
    filepath: str,
    model,
) -> dict:
    """
    Load model parameters from a binary checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        model: A Module whose parameters() will be overwritten.

    Returns:
        dict with keys: 'epoch', 'best_loss', 'lr'

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: On magic/version mismatch or shape mismatch.
        IOError: On truncated file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    params = model.parameters()

    with open(filepath, "rb") as f:
        # Header
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Invalid checkpoint magic: {magic!r} (expected {MAGIC!r})")

        ver = struct.unpack("<I", f.read(4))[0]
        if ver != VERSION:
            raise ValueError(f"Unsupported checkpoint version {ver} (expected {VERSION})")

        num_tensors = struct.unpack("<I", f.read(4))[0]
        if num_tensors != len(params):
            raise ValueError(
                f"Parameter count mismatch: checkpoint has {num_tensors}, "
                f"model has {len(params)}"
            )

        # Metadata
        epoch = struct.unpack("<I", f.read(4))[0]
        best_loss = struct.unpack("<d", f.read(8))[0]
        lr = struct.unpack("<d", f.read(8))[0]

        # Tensors
        for p in params:
            _read_tensor_into(f, p)

    return {"epoch": epoch, "best_loss": best_loss, "lr": lr}
