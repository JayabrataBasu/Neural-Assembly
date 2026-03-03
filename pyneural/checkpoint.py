"""
Checkpoint save/resume for PyNeural models.

Binary format (v3):
    HEADER  (16 bytes)
        magic       : 8 bytes  "NEURCKPT"
        version     : uint32   3
        num_tensors : uint32   count of weight tensors

    METADATA
        epoch       : uint32
        best_loss   : float64
        lr          : float64

    OPTIMIZER METADATA (v2+)
        has_optimizer : uint8
        opt_type      : uint8   (0=none, 1=sgd, 2=adam, 3=adamw)
        reserved      : uint16
        momentum      : float64
        beta1         : float64
        beta2         : float64
        epsilon       : float64
        weight_decay  : float64

    OPTIMIZER STATE BLOB (v3)
        state_blob_size : uint64
        state_blob      : bytes (opaque C optimizer state: moments, t, etc.)

    For each tensor:
        ndim        : uint32
        shape       : ndim × uint64
        dtype       : uint32   (0=float32, 1=float64)
        data        : numel × sizeof(dtype) bytes

Backward compatibility:
    - v1 checkpoints (without optimizer metadata) are loadable.
    - v2 checkpoints (without opaque optimizer state blob) are loadable.
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import Optional

from .core import _lib, NeuralDtype

MAGIC = b"NEURCKPT"
VERSION = 3

OPT_NONE = 0
OPT_SGD = 1
OPT_ADAM = 2
OPT_ADAMW = 3


def _optimizer_payload(optimizer) -> tuple[int, float, float, float, float, float]:
    """Return optimizer type and standardized hyper-parameter payload."""
    if optimizer is None:
        return OPT_NONE, 0.0, 0.0, 0.0, 0.0, 0.0

    name = optimizer.__class__.__name__.lower()
    momentum = float(getattr(optimizer, "momentum", 0.0))
    beta1 = float(getattr(optimizer, "beta1", 0.0))
    beta2 = float(getattr(optimizer, "beta2", 0.0))
    epsilon = float(getattr(optimizer, "epsilon", 0.0))
    weight_decay = float(getattr(optimizer, "weight_decay", 0.0))

    if name == "sgd":
        return OPT_SGD, momentum, 0.0, 0.0, 0.0, 0.0
    if name == "adam":
        return OPT_ADAM, 0.0, beta1, beta2, epsilon, 0.0
    if name == "adamw":
        return OPT_ADAMW, 0.0, beta1, beta2, epsilon, weight_decay
    return OPT_NONE, 0.0, 0.0, 0.0, 0.0, 0.0


def _apply_optimizer_payload(optimizer, opt_type: int, payload: tuple[float, float, float, float, float]) -> None:
    """Apply loaded optimizer metadata to an existing optimizer object."""
    if optimizer is None or opt_type == OPT_NONE:
        return

    momentum, beta1, beta2, epsilon, weight_decay = payload

    # Apply only attributes the object actually exposes.
    if hasattr(optimizer, "momentum"):
        optimizer.momentum = momentum
    if hasattr(optimizer, "beta1"):
        optimizer.beta1 = beta1
    if hasattr(optimizer, "beta2"):
        optimizer.beta2 = beta2
    if hasattr(optimizer, "epsilon"):
        optimizer.epsilon = epsilon
    if hasattr(optimizer, "weight_decay"):
        optimizer.weight_decay = weight_decay


def _export_optimizer_state_blob(optimizer) -> bytes:
    """Export opaque optimizer state bytes from C backend, if supported."""
    if optimizer is None or not hasattr(optimizer, "_ptr") or not optimizer._ptr:
        return b""
    if not hasattr(_lib, "opt_state_bytes"):
        return b""

    nbytes = int(_lib.opt_state_bytes(optimizer._ptr))
    if nbytes <= 0:
        return b""

    buf = (ctypes.c_uint8 * nbytes)()
    ret = int(_lib.opt_state_export(optimizer._ptr, ctypes.cast(buf, ctypes.c_void_p), nbytes))
    if ret != 0:
        return b""
    return bytes(buf)


def _import_optimizer_state_blob(optimizer, blob: bytes) -> None:
    """Import opaque optimizer state bytes into C backend, if supported."""
    if not blob:
        return
    if optimizer is None or not hasattr(optimizer, "_ptr") or not optimizer._ptr:
        return
    if not hasattr(_lib, "opt_state_import"):
        return

    nbytes = len(blob)
    buf = (ctypes.c_uint8 * nbytes).from_buffer_copy(blob)
    _lib.opt_state_import(optimizer._ptr, ctypes.cast(buf, ctypes.c_void_p), nbytes)


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
    optimizer=None,
    epoch: int = 0,
    best_loss: float = float("inf"),
    lr: float = 0.0,
) -> None:
    """
    Save model parameters and training metadata to a binary checkpoint.

    Args:
        filepath: Path to the output file.
        model: A Module (typically Sequential) whose parameters() to save.
        optimizer: Optional optimizer object (SGD/Adam/AdamW metadata saved).
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

        # Optimizer metadata (v2+)
        opt_type, momentum, beta1, beta2, epsilon, weight_decay = _optimizer_payload(optimizer)
        has_optimizer = 1 if opt_type != OPT_NONE else 0
        f.write(struct.pack("<BBH", has_optimizer, opt_type, 0))
        f.write(struct.pack("<ddddd", momentum, beta1, beta2, epsilon, weight_decay))

        # Opaque optimizer state (v3)
        state_blob = _export_optimizer_state_blob(optimizer)
        f.write(struct.pack("<Q", len(state_blob)))
        if state_blob:
            f.write(state_blob)

        # Tensors
        for p in params:
            _write_tensor(f, p)


def load_checkpoint(
    filepath: str,
    model,
    optimizer=None,
    strict_optimizer: bool = False,
) -> dict:
    """
    Load model parameters from a binary checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        model: A Module whose parameters() will be overwritten.
        optimizer: Optional optimizer object to update from checkpoint metadata.
        strict_optimizer: If True, raise on optimizer-type mismatch.

    Returns:
        dict with keys: 'epoch', 'best_loss', 'lr', 'optimizer_type'

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
        if ver not in (1, 2, VERSION):
            raise ValueError(f"Unsupported checkpoint version {ver} (expected 1, 2, or {VERSION})")

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

        optimizer_type = OPT_NONE
        state_blob = b""
        if ver >= 2:
            hdr = f.read(4)
            if len(hdr) < 4:
                raise IOError("Truncated checkpoint: expected optimizer header")
            has_optimizer, optimizer_type, _ = struct.unpack("<BBH", hdr)

            vals = f.read(40)
            if len(vals) < 40:
                raise IOError("Truncated checkpoint: expected optimizer payload")
            payload = struct.unpack("<ddddd", vals)

            if has_optimizer and optimizer is not None:
                if strict_optimizer:
                    cls = optimizer.__class__.__name__.lower()
                    expected = {
                        OPT_SGD: "sgd",
                        OPT_ADAM: "adam",
                        OPT_ADAMW: "adamw",
                    }.get(optimizer_type, "unknown")
                    if expected != "unknown" and cls != expected:
                        raise ValueError(
                            f"Optimizer type mismatch: checkpoint={expected}, provided={cls}"
                        )
                _apply_optimizer_payload(optimizer, optimizer_type, payload)

        if ver >= 3:
            size_raw = f.read(8)
            if len(size_raw) < 8:
                raise IOError("Truncated checkpoint: expected optimizer state blob size")
            blob_size = struct.unpack("<Q", size_raw)[0]
            if blob_size > 0:
                state_blob = f.read(blob_size)
                if len(state_blob) < blob_size:
                    raise IOError("Truncated checkpoint: expected optimizer state blob")

        # Apply loaded LR to optimizer when provided.
        if optimizer is not None and hasattr(optimizer, "lr"):
            optimizer.lr = lr

        if optimizer is not None and state_blob:
            _import_optimizer_state_blob(optimizer, state_blob)

        # Tensors
        for p in params:
            _read_tensor_into(f, p)

    return {
        "epoch": epoch,
        "best_loss": best_loss,
        "lr": lr,
        "optimizer_type": optimizer_type,
    }
