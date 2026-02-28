"""
pyneural.rnn — LSTM and GRU recurrent layers.

Both layers are backed by C implementations in rnn.c.  Data flows as
float64 arrays through ctypes — Python only handles shape bookkeeping
and buffer allocation.

Input convention:
    input  — [batch, seq_len, input_size]
    output — [batch, seq_len, hidden_size]   (full sequence of hidden states)
    h_n    — [batch, hidden_size]            (last hidden state)
    c_n    — [batch, hidden_size]            (last cell state, LSTM only)

For unbatched input [seq_len, input_size] we silently add a batch
dimension and strip it from the output.
"""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

from .core import _lib, _check_error


# double pointer type used everywhere
_dp = ctypes.POINTER(ctypes.c_double)


def _to_double_array(flat_list):
    """Turn a flat Python list of floats into a C double array."""
    n = len(flat_list)
    arr = (ctypes.c_double * n)(*flat_list)
    return arr


def _read_double_array(ptr, n):
    """Read n doubles from a ctypes pointer into a Python list."""
    return [ptr[i] for i in range(n)]


class LSTM:
    """
    Long Short-Term Memory layer.

    Args:
        input_size:  number of expected features in the input.
        hidden_size: number of features in the hidden state.

    Example::

        lstm = LSTM(input_size=10, hidden_size=20)
        # input: batch=2, seq_len=5, input_size=10
        output, (h_n, c_n) = lstm(input_data, batch=2, seq_len=5)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._ptr = _lib.lstm_create(input_size, hidden_size)
        if self._ptr is None or self._ptr == 0:
            raise RuntimeError("Failed to create LSTM layer")

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.lstm_free(self._ptr)
            self._ptr = None

    def __call__(
        self,
        input_data: list,
        batch: int = 1,
        seq_len: Optional[int] = None,
        h_0: Optional[list] = None,
        c_0: Optional[list] = None,
    ) -> Tuple[list, Tuple[list, list]]:
        """
        Run the LSTM on a flat list of doubles.

        Args:
            input_data: flat list of length batch * seq_len * input_size.
            batch: batch size.
            seq_len: sequence length (inferred if None).
            h_0: optional initial hidden state [batch * hidden_size].
            c_0: optional initial cell state [batch * hidden_size].

        Returns:
            (output, (h_n, c_n)) where:
              output — flat list [batch * seq_len * hidden_size]
              h_n    — flat list [batch * hidden_size]
              c_n    — flat list [batch * hidden_size]
        """
        I = self.input_size
        H = self.hidden_size

        if seq_len is None:
            total = len(input_data)
            seq_len = total // (batch * I)

        expected = batch * seq_len * I
        if len(input_data) != expected:
            raise ValueError(
                f"Expected input of length {expected}, got {len(input_data)}"
            )

        # allocate input / output buffers
        inp_buf = _to_double_array(input_data)
        out_buf = (ctypes.c_double * (batch * seq_len * H))()
        h_out   = (ctypes.c_double * (batch * H))()
        c_out   = (ctypes.c_double * (batch * H))()

        h_init = _to_double_array(h_0) if h_0 else None
        c_init = _to_double_array(c_0) if c_0 else None

        rc = _lib.lstm_forward(
            self._ptr,
            ctypes.cast(inp_buf, _dp),
            ctypes.cast(out_buf, _dp),
            ctypes.cast(h_out, _dp),
            ctypes.cast(c_out, _dp),
            ctypes.cast(h_init, _dp) if h_init else None,
            ctypes.cast(c_init, _dp) if c_init else None,
            batch,
            seq_len,
        )
        _check_error(rc, "lstm_forward")

        output = list(out_buf)
        h_n = list(h_out)
        c_n = list(c_out)
        return output, (h_n, c_n)

    def parameters(self) -> list:
        """Return (W_ih, W_hh, b_ih, b_hh) as flat lists of doubles."""
        H = self.hidden_size
        I = self.input_size
        h4 = 4 * H
        return [
            _read_double_array(_lib.lstm_weight_ih(self._ptr), h4 * I),
            _read_double_array(_lib.lstm_weight_hh(self._ptr), h4 * H),
            _read_double_array(_lib.lstm_bias_ih(self._ptr), h4),
            _read_double_array(_lib.lstm_bias_hh(self._ptr), h4),
        ]

    def __repr__(self) -> str:
        return f"LSTM(input_size={self.input_size}, hidden_size={self.hidden_size})"


class GRU:
    """
    Gated Recurrent Unit layer.

    Args:
        input_size:  number of expected features in the input.
        hidden_size: number of features in the hidden state.

    Example::

        gru = GRU(input_size=10, hidden_size=20)
        output, h_n = gru(input_data, batch=2, seq_len=5)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._ptr = _lib.gru_create(input_size, hidden_size)
        if self._ptr is None or self._ptr == 0:
            raise RuntimeError("Failed to create GRU layer")

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.gru_free(self._ptr)
            self._ptr = None

    def __call__(
        self,
        input_data: list,
        batch: int = 1,
        seq_len: Optional[int] = None,
        h_0: Optional[list] = None,
    ) -> Tuple[list, list]:
        """
        Run the GRU on a flat list of doubles.

        Args:
            input_data: flat list of length batch * seq_len * input_size.
            batch: batch size.
            seq_len: sequence length (inferred if None).
            h_0: optional initial hidden state [batch * hidden_size].

        Returns:
            (output, h_n) where:
              output — flat list [batch * seq_len * hidden_size]
              h_n    — flat list [batch * hidden_size]
        """
        I = self.input_size
        H = self.hidden_size

        if seq_len is None:
            total = len(input_data)
            seq_len = total // (batch * I)

        expected = batch * seq_len * I
        if len(input_data) != expected:
            raise ValueError(
                f"Expected input of length {expected}, got {len(input_data)}"
            )

        inp_buf = _to_double_array(input_data)
        out_buf = (ctypes.c_double * (batch * seq_len * H))()
        h_out   = (ctypes.c_double * (batch * H))()

        h_init = _to_double_array(h_0) if h_0 else None

        rc = _lib.gru_forward(
            self._ptr,
            ctypes.cast(inp_buf, _dp),
            ctypes.cast(out_buf, _dp),
            ctypes.cast(h_out, _dp),
            ctypes.cast(h_init, _dp) if h_init else None,
            batch,
            seq_len,
        )
        _check_error(rc, "gru_forward")

        output = list(out_buf)
        h_n = list(h_out)
        return output, h_n

    def parameters(self) -> list:
        """Return (W_ih, W_hh, b_ih, b_hh) as flat lists of doubles."""
        H = self.hidden_size
        I = self.input_size
        h3 = 3 * H
        return [
            _read_double_array(_lib.gru_weight_ih(self._ptr), h3 * I),
            _read_double_array(_lib.gru_weight_hh(self._ptr), h3 * H),
            _read_double_array(_lib.gru_bias_ih(self._ptr), h3),
            _read_double_array(_lib.gru_bias_hh(self._ptr), h3),
        ]

    def __repr__(self) -> str:
        return f"GRU(input_size={self.input_size}, hidden_size={self.hidden_size})"
