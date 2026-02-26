"""
TensorBoard-compatible event logging.

Writes TFEvent files that TensorBoard can read directly.
The heavy lifting (CRC32C, protobuf encoding, file I/O) is done
in C (tb_logger.c) via ctypes.

Usage:
    import pyneural as pn

    writer = pn.SummaryWriter("runs/experiment_1")
    for epoch in range(100):
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
    writer.close()

    # Then: tensorboard --logdir runs/
"""

import ctypes
import os
from . import core


class SummaryWriter:
    """TensorBoard SummaryWriter (backed by C tb_logger)."""

    def __init__(self, logdir: str = "runs"):
        """Open a new event file in the given directory.

        Args:
            logdir: Path to log directory (created automatically).
        """
        self._logdir = logdir
        logdir_bytes = logdir.encode("utf-8")
        self._handle = core._lib.tb_create_writer(logdir_bytes)
        if self._handle < 0:
            raise RuntimeError(f"Failed to create TensorBoard writer at '{logdir}'")

    # ── Scalar logging ───────────────────────────────────────────

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        """Log a single scalar value.

        Args:
            tag:         Metric name (e.g. "loss/train", "accuracy").
            value:       Scalar value.
            global_step: Training step / epoch number.
        """
        rc = core._lib.tb_add_scalar(
            self._handle,
            tag.encode("utf-8"),
            ctypes.c_float(value),
            ctypes.c_int64(global_step),
        )
        if rc != 0:
            raise RuntimeError(f"tb_add_scalar failed for tag='{tag}'")

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict,
                    global_step: int) -> None:
        """Log multiple scalars under a common tag.

        Args:
            main_tag:        Common prefix (e.g. "loss").
            tag_scalar_dict: Dict of {subtag: value}.
            global_step:     Training step.
        """
        for subtag, value in tag_scalar_dict.items():
            full_tag = f"{main_tag}/{subtag}"
            self.add_scalar(full_tag, value, global_step)

    # ── Histogram stats ──────────────────────────────────────────

    def add_histogram(self, tag: str, values, global_step: int) -> None:
        """Log histogram statistics (min/max/mean) for a set of values.

        This writes simplified histogram stats as scalars. Full bucket
        histograms are not yet supported.

        Args:
            tag:         Histogram name (e.g. "weights/layer1").
            values:      List or array of float values.
            global_step: Training step.
        """
        n = len(values)
        if n == 0:
            return
        arr = (ctypes.c_float * n)(*[float(v) for v in values])
        core._lib.tb_add_histogram_stats(
            self._handle,
            tag.encode("utf-8"),
            arr,
            n,
            ctypes.c_int64(global_step),
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush pending writes to disk."""
        core._lib.tb_flush(self._handle)

    def close(self) -> None:
        """Close the writer."""
        if self._handle >= 0:
            core._lib.tb_close(self._handle)
            self._handle = -1

    @property
    def logdir(self) -> str:
        """Return the log directory path."""
        return self._logdir

    @property
    def filepath(self) -> str:
        """Return the event file path."""
        raw = core._lib.tb_get_filepath(self._handle)
        return raw.decode("utf-8") if raw else ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"SummaryWriter(logdir='{self._logdir}')"
