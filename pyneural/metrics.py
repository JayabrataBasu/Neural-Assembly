"""
Metrics and evaluation utilities for PyNeural.

Provides confusion matrix, per-class precision/recall/F1,
and classification report for multi-class problems.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union


class ConfusionMatrix:
    """
    Confusion matrix for multi-class classification.
    
    Tracks predictions vs ground truth labels and computes
    per-class and aggregate metrics.
    
    Example:
        >>> cm = ConfusionMatrix(num_classes=3)
        >>> cm.update([0, 1, 2, 1], [0, 2, 2, 1])
        >>> print(cm.report())
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.matrix: List[List[int]] = [[0] * num_classes for _ in range(num_classes)]
        self._total = 0
    
    def reset(self) -> None:
        """Reset all counts to zero."""
        self.matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        self._total = 0
    
    def update(self, targets: List[int], predictions: List[int]) -> None:
        """
        Update confusion matrix with a batch of predictions.
        
        Args:
            targets: Ground truth class labels (list of ints)
            predictions: Predicted class labels (list of ints)
        """
        if len(targets) != len(predictions):
            raise ValueError(
                f"targets and predictions must have same length, "
                f"got {len(targets)} and {len(predictions)}"
            )
        for t, p in zip(targets, predictions):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.matrix[t][p] += 1
                self._total += 1
    
    @property
    def total(self) -> int:
        """Total number of samples tracked."""
        return self._total
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        if self._total == 0:
            return 0.0
        correct = sum(self.matrix[i][i] for i in range(self.num_classes))
        return correct / self._total
    
    def precision(self, cls: int) -> float:
        """Precision for a specific class: TP / (TP + FP)."""
        tp = self.matrix[cls][cls]
        col_sum = sum(self.matrix[r][cls] for r in range(self.num_classes))
        return tp / col_sum if col_sum > 0 else 0.0
    
    def recall(self, cls: int) -> float:
        """Recall for a specific class: TP / (TP + FN)."""
        tp = self.matrix[cls][cls]
        row_sum = sum(self.matrix[cls][c] for c in range(self.num_classes))
        return tp / row_sum if row_sum > 0 else 0.0
    
    def f1_score(self, cls: int) -> float:
        """F1 score for a specific class: 2 * P * R / (P + R)."""
        p = self.precision(cls)
        r = self.recall(cls)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def support(self, cls: int) -> int:
        """Number of true instances for a class."""
        return sum(self.matrix[cls][c] for c in range(self.num_classes))
    
    def macro_precision(self) -> float:
        """Macro-averaged precision across all classes."""
        vals = [self.precision(c) for c in range(self.num_classes)]
        return sum(vals) / len(vals) if vals else 0.0
    
    def macro_recall(self) -> float:
        """Macro-averaged recall across all classes."""
        vals = [self.recall(c) for c in range(self.num_classes)]
        return sum(vals) / len(vals) if vals else 0.0
    
    def macro_f1(self) -> float:
        """Macro-averaged F1 score across all classes."""
        vals = [self.f1_score(c) for c in range(self.num_classes)]
        return sum(vals) / len(vals) if vals else 0.0
    
    def weighted_precision(self) -> float:
        """Weighted precision (by class support)."""
        if self._total == 0:
            return 0.0
        return sum(
            self.precision(c) * self.support(c) for c in range(self.num_classes)
        ) / self._total
    
    def weighted_recall(self) -> float:
        """Weighted recall (by class support)."""
        if self._total == 0:
            return 0.0
        return sum(
            self.recall(c) * self.support(c) for c in range(self.num_classes)
        ) / self._total
    
    def weighted_f1(self) -> float:
        """Weighted F1 score (by class support)."""
        if self._total == 0:
            return 0.0
        return sum(
            self.f1_score(c) * self.support(c) for c in range(self.num_classes)
        ) / self._total
    
    def report(self, digits: int = 4) -> str:
        """
        Generate a classification report string (sklearn-style).
        
        Returns:
            Formatted string with per-class and aggregate metrics
        """
        # Header
        width = max(len(n) for n in self.class_names) + 2
        header = f"{'':>{width}} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
        sep = "-" * len(header)
        
        lines = [sep, header, sep]
        
        # Per-class rows
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
        
        # Aggregate rows
        total_support = self._total
        
        # Accuracy
        lines.append(
            f"{'accuracy':>{width}} {'':>10} {'':>10} "
            f"{self.accuracy:>10.{digits}f} {total_support:>10d}"
        )
        
        # Macro average
        lines.append(
            f"{'macro avg':>{width}} {self.macro_precision():>10.{digits}f} "
            f"{self.macro_recall():>10.{digits}f} {self.macro_f1():>10.{digits}f} "
            f"{total_support:>10d}"
        )
        
        # Weighted average
        lines.append(
            f"{'weighted avg':>{width}} {self.weighted_precision():>10.{digits}f} "
            f"{self.weighted_recall():>10.{digits}f} {self.weighted_f1():>10.{digits}f} "
            f"{total_support:>10d}"
        )
        
        lines.append(sep)
        return "\n".join(lines)
    
    def matrix_str(self, normalize: bool = False) -> str:
        """
        Format the confusion matrix as a readable string.
        
        Args:
            normalize: If True, show proportions instead of counts
        """
        # Determine column widths
        if normalize:
            cell_width = 8
        else:
            max_val = max(max(row) for row in self.matrix) if self._total > 0 else 0
            cell_width = max(len(str(max_val)), 5) + 1
        
        name_width = max(len(n) for n in self.class_names) + 2
        
        # Header row
        header = " " * name_width + " " + " ".join(
            f"{self.class_names[c]:>{cell_width}}" for c in range(self.num_classes)
        )
        
        lines = ["Confusion Matrix (rows=true, cols=predicted):", header]
        lines.append("-" * len(header))
        
        for r in range(self.num_classes):
            row_sum = sum(self.matrix[r])
            cells = []
            for c in range(self.num_classes):
                if normalize and row_sum > 0:
                    cells.append(f"{self.matrix[r][c] / row_sum:>{cell_width}.3f}")
                else:
                    cells.append(f"{self.matrix[r][c]:>{cell_width}d}")
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
    
    Returns:
        Top-k accuracy
    """
    if len(targets) == 0:
        return 0.0
    correct = 0
    for target, scores in zip(targets, prediction_scores):
        # Get indices of top-k scores
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_k_classes = [idx for idx, _ in indexed[:k]]
        if target in top_k_classes:
            correct += 1
    return correct / len(targets)
