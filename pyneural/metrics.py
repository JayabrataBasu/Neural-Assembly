"""
Metrics and evaluation utilities for PyNeural.

Core computations (confusion matrix update, precision/recall/F1)
are implemented in x86-64 assembly (training_ops.asm) and called
via ctypes. Python provides the high-level API and formatting.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional

from .core import _lib


class ConfusionMatrix:
    """
    Confusion matrix for multi-class classification.

    The matrix update and per-class metric computations are
    executed natively in assembly (training_ops.asm).

    Example:
        >>> cm = ConfusionMatrix(num_classes=3)
        >>> cm.update([0, 1, 2, 1], [0, 2, 2, 1])
        >>> print(cm.report())
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        # Flat int32 matrix: row-major, num_classes x num_classes
        self._matrix = (ctypes.c_int32 * (num_classes * num_classes))()
        ctypes.memset(self._matrix, 0, ctypes.sizeof(self._matrix))
        self._total = 0

    def reset(self) -> None:
        """Reset all counts to zero."""
        ctypes.memset(self._matrix, 0, ctypes.sizeof(self._matrix))
        self._total = 0

    def update(self, targets: List[int], predictions: List[int]) -> None:
        """
        Update confusion matrix with a batch of predictions.
        Delegates to assembly ``confusion_matrix_update``.
        """
        if len(targets) != len(predictions):
            raise ValueError(
                f"targets and predictions must have same length, "
                f"got {len(targets)} and {len(predictions)}"
            )
        n = len(targets)
        if n == 0:
            return

        t_arr = (ctypes.c_int32 * n)(*targets)
        p_arr = (ctypes.c_int32 * n)(*predictions)

        valid = _lib.neural_confusion_matrix_update(
            self._matrix, t_arr, p_arr, n, self.num_classes
        )
        self._total += valid

    @property
    def total(self) -> int:
        return self._total

    @property
    def accuracy(self) -> float:
        """Overall accuracy (assembly-backed)."""
        if self._total == 0:
            return 0.0
        return _lib.neural_compute_accuracy_from_matrix(
            self._matrix, self.num_classes
        )

    def precision(self, cls: int) -> float:
        """Precision for class *cls* (assembly)."""
        return _lib.neural_compute_class_precision(
            self._matrix, cls, self.num_classes
        )

    def recall(self, cls: int) -> float:
        """Recall for class *cls* (assembly)."""
        return _lib.neural_compute_class_recall(
            self._matrix, cls, self.num_classes
        )

    def f1_score(self, cls: int) -> float:
        """F1 score for class *cls* (assembly)."""
        return _lib.neural_compute_class_f1(
            self._matrix, cls, self.num_classes
        )

    def support(self, cls: int) -> int:
        """Number of true instances for a class (row sum)."""
        base = cls * self.num_classes
        return sum(self._matrix[base + c] for c in range(self.num_classes))

    def macro_precision(self) -> float:
        vals = [self.precision(c) for c in range(self.num_classes)]
        return sum(vals) / len(vals) if vals else 0.0

    def macro_recall(self) -> float:
        vals = [self.recall(c) for c in range(self.num_classes)]
        return sum(vals) / len(vals) if vals else 0.0

    def macro_f1(self) -> float:
        vals = [self.f1_score(c) for c in range(self.num_classes)]
        return sum(vals) / len(vals) if vals else 0.0

    def weighted_precision(self) -> float:
        if self._total == 0:
            return 0.0
        return sum(
            self.precision(c) * self.support(c) for c in range(self.num_classes)
        ) / self._total

    def weighted_recall(self) -> float:
        if self._total == 0:
            return 0.0
        return sum(
            self.recall(c) * self.support(c) for c in range(self.num_classes)
        ) / self._total

    def weighted_f1(self) -> float:
        if self._total == 0:
            return 0.0
        return sum(
            self.f1_score(c) * self.support(c) for c in range(self.num_classes)
        ) / self._total

    def report(self, digits: int = 4) -> str:
        """Generate a classification report string (sklearn-style)."""
        width = max(len(n) for n in self.class_names) + 2
        header = (
            f"{'':>{width}} {'precision':>10} {'recall':>10} "
            f"{'f1-score':>10} {'support':>10}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for c in range(self.num_classes):
            name = self.class_names[c]
            p = self.precision(c)
            r = self.recall(c)
            f = self.f1_score(c)
            s = self.support(c)
            lines.append(
                f"{name:>{width}} {p:>10.{digits}f} {r:>10.{digits}f} "
                f"{f:>10.{digits}f} {s:>10d}"
            )

        lines.append(sep)
        total_support = self._total

        lines.append(
            f"{'accuracy':>{width}} {'':>10} {'':>10} "
            f"{self.accuracy:>10.{digits}f} {total_support:>10d}"
        )
        lines.append(
            f"{'macro avg':>{width}} {self.macro_precision():>10.{digits}f} "
            f"{self.macro_recall():>10.{digits}f} {self.macro_f1():>10.{digits}f} "
            f"{total_support:>10d}"
        )
        lines.append(
            f"{'weighted avg':>{width}} {self.weighted_precision():>10.{digits}f} "
            f"{self.weighted_recall():>10.{digits}f} {self.weighted_f1():>10.{digits}f} "
            f"{total_support:>10d}"
        )
        lines.append(sep)
        return "\n".join(lines)

    def matrix_str(self, normalize: bool = False) -> str:
        """Format the confusion matrix as a readable string."""
        nc = self.num_classes
        if normalize:
            cell_width = 8
        else:
            max_val = max(self._matrix[i] for i in range(nc * nc)) if self._total > 0 else 0
            cell_width = max(len(str(max_val)), 5) + 1

        name_width = max(len(n) for n in self.class_names) + 2

        header = " " * name_width + " " + " ".join(
            f"{self.class_names[c]:>{cell_width}}" for c in range(nc)
        )
        lines = ["Confusion Matrix (rows=true, cols=predicted):", header]
        lines.append("-" * len(header))

        for r in range(nc):
            base = r * nc
            row_sum = sum(self._matrix[base + c] for c in range(nc))
            cells = []
            for c in range(nc):
                v = self._matrix[base + c]
                if normalize and row_sum > 0:
                    cells.append(f"{v / row_sum:>{cell_width}.3f}")
                else:
                    cells.append(f"{v:>{cell_width}d}")
            lines.append(f"{self.class_names[r]:>{name_width}} " + " ".join(cells))

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ConfusionMatrix(num_classes={self.num_classes}, total={self._total})"


def compute_accuracy(targets: List[int], predictions: List[int]) -> float:
    """Compute classification accuracy."""
    if len(targets) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(targets, predictions) if t == p)
    return correct / len(targets)


def top_k_accuracy(
    targets: List[int],
    prediction_scores: List[List[float]],
    k: int = 5,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        targets: Ground truth labels
        prediction_scores: Per-sample class scores/probabilities
        k: Number of top predictions to consider
    """
    if len(targets) == 0:
        return 0.0
    correct = 0
    for target, scores in zip(targets, prediction_scores):
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_k_classes = [idx for idx, _ in indexed[:k]]
        if target in top_k_classes:
            correct += 1
    return correct / len(targets)
