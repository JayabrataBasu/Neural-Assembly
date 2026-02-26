"""
INT8 quantization with calibration.

Provides affine and symmetric quantization of float64 tensors to int8,
backed by C (quantize.c) via ctypes.

Usage:
    import pyneural as pn

    q = pn.Quantizer(symmetric=True)
    q.calibrate(weight_data, n_elements)
    int8_data = q.quantize(weight_data, n_elements)
    recon = q.dequantize(int8_data, n_elements)
    print(f"MSE: {q.error(weight_data, n_elements):.6f}")
    print(f"SNR: {q.snr(weight_data, n_elements):.1f} dB")
"""

import ctypes
from . import core
from .core import QuantParamsC


class QuantParams:
    """Python-side view of quantization parameters."""

    def __init__(self, scale: float = 1.0, zero_point: int = 0,
                 min_val: float = 0.0, max_val: float = 0.0,
                 symmetric: bool = True):
        self.scale = scale
        self.zero_point = zero_point
        self.min_val = min_val
        self.max_val = max_val
        self.symmetric = symmetric

    def _to_c(self) -> QuantParamsC:
        p = QuantParamsC()
        p.scale = self.scale
        p.zero_point = self.zero_point
        p.min_val = self.min_val
        p.max_val = self.max_val
        p.symmetric = 1 if self.symmetric else 0
        return p

    @classmethod
    def _from_c(cls, p: QuantParamsC) -> "QuantParams":
        return cls(
            scale=p.scale,
            zero_point=p.zero_point,
            min_val=p.min_val,
            max_val=p.max_val,
            symmetric=bool(p.symmetric),
        )

    def __repr__(self):
        mode = "symmetric" if self.symmetric else "affine"
        return (f"QuantParams({mode}, scale={self.scale:.6g}, "
                f"zp={self.zero_point}, range=[{self.min_val:.4g}, {self.max_val:.4g}])")


class Quantizer:
    """INT8 quantizer with calibration."""

    def __init__(self, symmetric: bool = True, percentile: float = 0.0):
        """Create a quantizer.

        Args:
            symmetric:  If True, use symmetric quantization (zero_point=0).
                       If False, use affine (asymmetric) quantization.
            percentile: If > 0, use percentile clipping during calibration
                       (e.g. 0.01 clips 1% on each tail).
        """
        self.symmetric = symmetric
        self.percentile = percentile
        self.params = None  # Set after calibration

    def calibrate(self, data_ptr, n: int) -> QuantParams:
        """Calibrate quantization parameters from data.

        Args:
            data_ptr: ctypes pointer to float64 array.
            n:       Number of elements.

        Returns:
            Calibrated QuantParams.
        """
        c_params = QuantParamsC()
        sym = 1 if self.symmetric else 0

        if self.percentile > 0.0:
            rc = core._lib.calibrate_percentile(
                data_ptr, ctypes.c_int64(n),
                ctypes.c_double(self.percentile),
                sym, ctypes.byref(c_params),
            )
        else:
            rc = core._lib.calibrate_minmax(
                data_ptr, ctypes.c_int64(n),
                sym, ctypes.byref(c_params),
            )

        if rc != 0:
            raise RuntimeError("Calibration failed")

        self.params = QuantParams._from_c(c_params)
        return self.params

    def quantize(self, data_ptr, n: int) -> ctypes.Array:
        """Quantize float64 array to int8.

        Args:
            data_ptr: ctypes pointer to float64 array.
            n:       Number of elements.

        Returns:
            ctypes int8 array of length n.
        """
        if self.params is None:
            raise RuntimeError("Must calibrate before quantizing")

        out = (ctypes.c_int8 * n)()
        c_params = self.params._to_c()
        rc = core._lib.quantize_tensor(
            data_ptr, out, ctypes.c_int64(n),
            ctypes.byref(c_params),
        )
        if rc != 0:
            raise RuntimeError("Quantization failed")
        return out

    def dequantize(self, int8_ptr, n: int) -> ctypes.Array:
        """Dequantize int8 array back to float64.

        Args:
            int8_ptr: ctypes pointer/array of int8 values.
            n:       Number of elements.

        Returns:
            ctypes float64 array of length n.
        """
        if self.params is None:
            raise RuntimeError("Must calibrate before dequantizing")

        out = (ctypes.c_double * n)()
        c_params = self.params._to_c()
        rc = core._lib.dequantize_tensor(
            int8_ptr, out, ctypes.c_int64(n),
            ctypes.byref(c_params),
        )
        if rc != 0:
            raise RuntimeError("Dequantization failed")
        return out

    def error(self, original_ptr, n: int) -> float:
        """Compute mean squared quantization error.

        Args:
            original_ptr: ctypes pointer to original float64 array.
            n:           Number of elements.

        Returns:
            MSE between original and quantize→dequantize reconstruction.
        """
        if self.params is None:
            raise RuntimeError("Must calibrate first")
        c_params = self.params._to_c()
        return core._lib.quantization_error(
            original_ptr, ctypes.c_int64(n),
            ctypes.byref(c_params),
        )

    def snr(self, original_ptr, n: int) -> float:
        """Compute signal-to-noise ratio of quantization (dB).

        Args:
            original_ptr: ctypes pointer to original float64 array.
            n:           Number of elements.

        Returns:
            SNR in dB (higher is better).
        """
        if self.params is None:
            raise RuntimeError("Must calibrate first")
        c_params = self.params._to_c()
        return core._lib.quantization_snr(
            original_ptr, ctypes.c_int64(n),
            ctypes.byref(c_params),
        )

    def __repr__(self):
        mode = "symmetric" if self.symmetric else "affine"
        calib = "calibrated" if self.params else "uncalibrated"
        return f"Quantizer({mode}, {calib})"


def quantized_matmul(a_int8, b_int8, m: int, k: int, n: int,
                     params_a: QuantParams,
                     params_b: QuantParams) -> ctypes.Array:
    """Perform quantized matrix multiplication.

    C[M,N] = dequant(A[M,K]) @ dequant(B[K,N])

    The inner accumulation is done in int32, then scaled to float64.

    Args:
        a_int8:  ctypes int8 array for matrix A (M*K elements).
        b_int8:  ctypes int8 array for matrix B (K*N elements).
        m, k, n: Matrix dimensions.
        params_a, params_b: Quantization parameters for A and B.

    Returns:
        ctypes float64 array of M*N elements.
    """
    c_out = (ctypes.c_double * (m * n))()
    pa = params_a._to_c()
    pb = params_b._to_c()
    rc = core._lib.quantized_matmul(
        a_int8, b_int8, c_out,
        ctypes.c_int64(m), ctypes.c_int64(k), ctypes.c_int64(n),
        ctypes.byref(pa), ctypes.byref(pb),
    )
    if rc != 0:
        raise RuntimeError("Quantized matmul failed")
    return c_out
