"""
Model pruning utilities.

Provides magnitude-based (unstructured) and structured (row/column)
pruning backed by C (pruning.c) via ctypes.

Usage:
    import pyneural as pn

    # Prune 90% of smallest weights
    pruner = pn.Pruner(sparsity=0.9)
    pruner.prune(weight_data, n_elements)
    print(f"Sparsity: {pruner.get_sparsity(weight_data, n_elements):.1%}")
"""

import ctypes
from . import core


class Pruner:
    """Weight pruner with mask tracking."""

    def __init__(self, sparsity: float = 0.5, method: str = "magnitude"):
        """Create a pruner.

        Args:
            sparsity: Target fraction of zeros (0.0 to 1.0).
            method:   "magnitude" (threshold-based) or "topk" (keep top-k).
        """
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0,1], got {sparsity}")
        if method not in ("magnitude", "topk"):
            raise ValueError(f"method must be 'magnitude' or 'topk', got '{method}'")
        self.sparsity = sparsity
        self.method = method
        self._mask = None  # uint8 array, set after pruning

    def prune(self, weights_ptr, n: int) -> int:
        """Prune weights in-place.

        Args:
            weights_ptr: ctypes pointer to float64 weight array.
            n:          Number of elements.

        Returns:
            Number of weights pruned.
        """
        mask = (ctypes.c_uint8 * n)()

        if self.method == "topk":
            pruned = core._lib.prune_topk(
                weights_ptr, mask, ctypes.c_int64(n),
                ctypes.c_double(1.0 - self.sparsity),
            )
        else:
            # Find threshold for target sparsity
            threshold = core._lib.compute_threshold_for_sparsity(
                weights_ptr, ctypes.c_int64(n),
                ctypes.c_double(self.sparsity),
            )
            pruned = core._lib.prune_magnitude(
                weights_ptr, mask, ctypes.c_int64(n),
                ctypes.c_double(threshold),
            )

        self._mask = mask
        return pruned

    def reapply_mask(self, weights_ptr, n: int) -> None:
        """Re-apply the stored mask (e.g. after an optimizer step).

        Args:
            weights_ptr: ctypes pointer to float64 weight array.
            n:          Number of elements.
        """
        if self._mask is None:
            return
        core._lib.apply_mask(weights_ptr, self._mask, ctypes.c_int64(n))

    @property
    def mask(self):
        """Return the current pruning mask (uint8 ctypes array)."""
        return self._mask

    @staticmethod
    def get_sparsity(weights_ptr, n: int) -> float:
        """Compute current sparsity ratio.

        Args:
            weights_ptr: ctypes pointer to float64 weight array.
            n:          Number of elements.

        Returns:
            Fraction of zero elements.
        """
        return core._lib.compute_sparsity(weights_ptr, ctypes.c_int64(n))

    @staticmethod
    def count_nonzero(weights_ptr, n: int) -> int:
        """Count non-zero elements.

        Args:
            weights_ptr: ctypes pointer to float64 weight array.
            n:          Number of elements.

        Returns:
            Number of non-zero elements.
        """
        return core._lib.count_nonzero(weights_ptr, ctypes.c_int64(n))


def prune_magnitude(weights_ptr, n: int, threshold: float,
                    mask=None) -> int:
    """Prune weights below threshold (functional API).

    Args:
        weights_ptr: ctypes pointer to float64 array (modified in-place).
        n:          Number of elements.
        threshold:  Absolute value threshold.
        mask:       Optional uint8 ctypes array to receive mask.

    Returns:
        Number of weights pruned.
    """
    return core._lib.prune_magnitude(
        weights_ptr,
        mask if mask is not None else None,
        ctypes.c_int64(n),
        ctypes.c_double(threshold),
    )


def prune_rows(weights_ptr, rows: int, cols: int, threshold: float,
               row_mask=None) -> int:
    """Structured row pruning (zero entire rows with small L2 norm).

    Args:
        weights_ptr: ctypes pointer to float64 matrix (row-major).
        rows, cols:  Matrix dimensions.
        threshold:   L2 norm threshold.
        row_mask:    Optional uint8 ctypes array of size rows.

    Returns:
        Number of rows pruned.
    """
    return core._lib.prune_rows(
        weights_ptr,
        ctypes.c_int64(rows), ctypes.c_int64(cols),
        ctypes.c_double(threshold),
        row_mask if row_mask is not None else None,
    )


def prune_cols(weights_ptr, rows: int, cols: int, threshold: float,
               col_mask=None) -> int:
    """Structured column pruning.

    Args:
        weights_ptr: ctypes pointer to float64 matrix (row-major).
        rows, cols:  Matrix dimensions.
        threshold:   L2 norm threshold.
        col_mask:    Optional uint8 ctypes array of size cols.

    Returns:
        Number of columns pruned.
    """
    return core._lib.prune_cols(
        weights_ptr,
        ctypes.c_int64(rows), ctypes.c_int64(cols),
        ctypes.c_double(threshold),
        col_mask if col_mask is not None else None,
    )
