"""
scikit-learn compatible API for PyNeural.

Provides MLPClassifier and MLPRegressor classes that follow the
sklearn estimator interface (fit/predict/score).

Example:
    >>> from pyneural.sklearn_compat import MLPClassifier, MLPRegressor
    >>>
    >>> # Classification
    >>> clf = MLPClassifier(hidden_layers=[256, 128], epochs=50, learning_rate=0.001)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> score = clf.score(X_test, y_test)
    >>>
    >>> # Regression
    >>> reg = MLPRegressor(hidden_layers=[128, 64], epochs=100)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

from .nn import Linear, ReLU, Sigmoid, Softmax, Sequential, Dropout, Module
from .optim import SGD, Adam, AdamW
from .tensor import Tensor
from .core import _lib


class _BaseEstimator:
    """Base class for sklearn-compatible estimators."""

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        epochs: int = 100,
        batch_size: int = 32,
        dropout: Optional[float] = None,
        weight_decay: float = 0.0,
        verbose: bool = False,
        early_stopping: bool = False,
        patience: int = 10,
        validation_fraction: float = 0.1,
    ):
        self.hidden_layers = hidden_layers or [100]
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction

        self.model_: Optional[Sequential] = None
        self._optimizer = None
        self._is_fitted = False
        self.loss_curve_: List[float] = []
        self.n_iter_: int = 0

    def _get_activation(self) -> type:
        act_map = {
            "relu": ReLU, "sigmoid": Sigmoid, "tanh": lambda: __import__('pyneural.nn', fromlist=['Tanh']).Tanh,
        }
        # avoid circular; import directly
        from .nn import Tanh
        act_map_full = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
        cls = act_map_full.get(self.activation.lower())
        if cls is None:
            raise ValueError(f"Unknown activation: {self.activation}")
        return cls

    def _build_model(self, input_dim: int, output_dim: int, final_activation=None):
        act_cls = self._get_activation()
        layers: List[Module] = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_layers:
            layers.append(Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if self.dropout is not None and self.dropout > 0:
                layers.append(Dropout(p=self.dropout))
            prev_dim = hidden_dim

        layers.append(Linear(prev_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation())

        self.model_ = Sequential(layers)

    def _build_optimizer(self):
        params = self.model_.parameters()
        opt_name = self.optimizer.lower()
        if opt_name == "sgd":
            self._optimizer = SGD(params, lr=self.learning_rate)
        elif opt_name == "adam":
            self._optimizer = Adam(params, lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
        elif opt_name == "adamw":
            self._optimizer = AdamW(params, lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn interface)."""
        return {
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "verbose": self.verbose,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "validation_fraction": self.validation_fraction,
        }

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn interface)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self


class MLPClassifier(_BaseEstimator):
    """
    Multi-layer Perceptron classifier with sklearn-compatible API.

    Uses Softmax output for multi-class classification.

    Args:
        hidden_layers: List of hidden layer sizes (default: [100]).
        activation: Activation function ('relu', 'sigmoid', 'tanh').
        learning_rate: Learning rate (default: 0.001).
        optimizer: Optimizer type ('sgd', 'adam', 'adamw').
        epochs: Number of training epochs (default: 100).
        batch_size: Mini-batch size (default: 32).
        dropout: Dropout rate between layers (None = no dropout).
        weight_decay: L2 regularization (default: 0.0).
        verbose: Print training progress (default: False).
        early_stopping: Enable early stopping (default: False).
        patience: Early stopping patience (default: 10).
        validation_fraction: Fraction for validation (default: 0.1).

    Example:
        >>> clf = MLPClassifier(hidden_layers=[256, 128], epochs=50)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> accuracy = clf.score(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes_: Optional[List[int]] = None
        self.n_classes_: int = 0

    def fit(self, X, y) -> "MLPClassifier":
        """
        Fit the model to training data.

        Args:
            X: Training features (list of lists or 2D array-like).
            y: Training labels (list of integers).

        Returns:
            self
        """
        # Convert inputs
        if hasattr(X, 'tolist'):
            X = X.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0

        # Determine classes
        self.classes_ = sorted(set(y))
        self.n_classes_ = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        # Build model
        self._build_model(n_features, self.n_classes_, Softmax)
        self._build_optimizer()

        # Split validation set if early stopping
        if self.early_stopping:
            split = int(n_samples * (1.0 - self.validation_fraction))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        self.loss_curve_ = []

        for epoch in range(self.epochs):
            # Mini-batch training
            epoch_loss = 0.0
            n_batches = max(1, len(X_train) // self.batch_size)

            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(X_train))
                batch_x = X_train[start:end]
                batch_y = y_train[start:end]

                if len(batch_x) == 0:
                    continue

                # Forward pass
                x_tensor = Tensor.from_list(batch_x)
                output = self.model_(x_tensor)

                # Compute cross-entropy loss (manual)
                out_data = output.tolist() if hasattr(output, 'tolist') else []
                batch_loss = 0.0
                for i, label in enumerate(batch_y):
                    idx = class_to_idx[label]
                    if isinstance(out_data[i], list):
                        prob = max(out_data[i][idx], 1e-10)
                    else:
                        prob = max(out_data[i], 1e-10)
                    batch_loss -= math.log(prob)
                batch_loss /= len(batch_y)
                epoch_loss += batch_loss

            epoch_loss /= n_batches
            self.loss_curve_.append(epoch_loss)

            # Validation
            if self.early_stopping and X_val:
                val_loss = self._compute_loss(X_val, y_val, class_to_idx)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={epoch_loss:.6f}")

            self.n_iter_ = epoch + 1

        self._is_fitted = True
        return self

    def _compute_loss(self, X, y, class_to_idx) -> float:
        x_tensor = Tensor.from_list(X)
        output = self.model_(x_tensor)
        out_data = output.tolist() if hasattr(output, 'tolist') else []
        loss = 0.0
        for i, label in enumerate(y):
            idx = class_to_idx[label]
            if isinstance(out_data[i], list):
                prob = max(out_data[i][idx], 1e-10)
            else:
                prob = max(out_data[i], 1e-10)
            loss -= math.log(prob)
        return loss / len(y)

    def predict(self, X) -> List[int]:
        """
        Predict class labels.

        Args:
            X: Input features.

        Returns:
            List of predicted class labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if hasattr(X, 'tolist'):
            X = X.tolist()

        x_tensor = Tensor.from_list(X)
        self.model_.eval()
        output = self.model_(x_tensor)
        self.model_.train()

        out_data = output.tolist() if hasattr(output, 'tolist') else []
        predictions = []
        for row in out_data:
            if isinstance(row, list):
                pred_idx = row.index(max(row))
            else:
                pred_idx = 0
            predictions.append(self.classes_[pred_idx])
        return predictions

    def predict_proba(self, X) -> List[List[float]]:
        """
        Predict class probabilities.

        Args:
            X: Input features.

        Returns:
            List of probability vectors.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if hasattr(X, 'tolist'):
            X = X.tolist()

        x_tensor = Tensor.from_list(X)
        self.model_.eval()
        output = self.model_(x_tensor)
        self.model_.train()

        out_data = output.tolist() if hasattr(output, 'tolist') else []
        return out_data

    def score(self, X, y) -> float:
        """
        Return classification accuracy.

        Args:
            X: Test features.
            y: True labels.

        Returns:
            Accuracy as a float in [0.0, 1.0].
        """
        if hasattr(y, 'tolist'):
            y = y.tolist()
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y) if len(y) > 0 else 0.0


class MLPRegressor(_BaseEstimator):
    """
    Multi-layer Perceptron regressor with sklearn-compatible API.

    Uses linear output (no final activation) for regression tasks.

    Args:
        hidden_layers: List of hidden layer sizes (default: [100]).
        activation: Activation function ('relu', 'sigmoid', 'tanh').
        learning_rate: Learning rate (default: 0.001).
        optimizer: Optimizer type ('sgd', 'adam', 'adamw').
        epochs: Number of training epochs (default: 100).
        batch_size: Mini-batch size (default: 32).
        dropout: Dropout rate (None = no dropout).
        weight_decay: L2 regularization (default: 0.0).
        verbose: Print progress (default: False).
        early_stopping: Enable early stopping (default: False).
        patience: Early stopping patience (default: 10).
        validation_fraction: Fraction for validation (default: 0.1).

    Example:
        >>> reg = MLPRegressor(hidden_layers=[128, 64], epochs=100)
        >>> reg.fit(X_train, y_train)
        >>> predictions = reg.predict(X_test)
        >>> rmse = reg.rmse(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_outputs_: int = 1

    def fit(self, X, y) -> "MLPRegressor":
        """
        Fit the model to training data.

        Args:
            X: Training features (list of lists or 2D array-like).
            y: Training targets (list of floats or list of lists).

        Returns:
            self
        """
        if hasattr(X, 'tolist'):
            X = X.tolist()
        if hasattr(y, 'tolist'):
            y = y.tolist()

        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0

        # Determine output dimension
        if n_samples > 0 and isinstance(y[0], (list, tuple)):
            self.n_outputs_ = len(y[0])
        else:
            self.n_outputs_ = 1

        # Build model (no final activation for regression)
        self._build_model(n_features, self.n_outputs_)
        self._build_optimizer()

        # Split validation set
        if self.early_stopping:
            split = int(n_samples * (1.0 - self.validation_fraction))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        best_val_loss = float("inf")
        patience_counter = 0
        self.loss_curve_ = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = max(1, len(X_train) // self.batch_size)

            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(X_train))
                batch_x = X_train[start:end]
                batch_y = y_train[start:end]

                if len(batch_x) == 0:
                    continue

                x_tensor = Tensor.from_list(batch_x)
                output = self.model_(x_tensor)

                # MSE loss
                out_data = output.tolist() if hasattr(output, 'tolist') else []
                batch_loss = 0.0
                for i, target in enumerate(batch_y):
                    if isinstance(target, (list, tuple)):
                        for j in range(len(target)):
                            pred = out_data[i][j] if isinstance(out_data[i], list) else out_data[i]
                            batch_loss += (pred - target[j]) ** 2
                    else:
                        pred = out_data[i][0] if isinstance(out_data[i], list) else out_data[i]
                        batch_loss += (pred - target) ** 2
                batch_loss /= len(batch_y)
                epoch_loss += batch_loss

            epoch_loss /= n_batches
            self.loss_curve_.append(epoch_loss)

            if self.early_stopping and X_val:
                val_loss = self._compute_loss(X_val, y_val)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            if self.verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={epoch_loss:.6f}")

            self.n_iter_ = epoch + 1

        self._is_fitted = True
        return self

    def _compute_loss(self, X, y) -> float:
        x_tensor = Tensor.from_list(X)
        output = self.model_(x_tensor)
        out_data = output.tolist() if hasattr(output, 'tolist') else []
        loss = 0.0
        for i, target in enumerate(y):
            if isinstance(target, (list, tuple)):
                for j in range(len(target)):
                    pred = out_data[i][j] if isinstance(out_data[i], list) else out_data[i]
                    loss += (pred - target[j]) ** 2
            else:
                pred = out_data[i][0] if isinstance(out_data[i], list) else out_data[i]
                loss += (pred - target) ** 2
        return loss / len(y)

    def predict(self, X) -> List[float]:
        """
        Predict target values.

        Args:
            X: Input features.

        Returns:
            List of predicted values (flat if single output).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if hasattr(X, 'tolist'):
            X = X.tolist()

        x_tensor = Tensor.from_list(X)
        self.model_.eval()
        output = self.model_(x_tensor)
        self.model_.train()

        out_data = output.tolist() if hasattr(output, 'tolist') else []
        if self.n_outputs_ == 1:
            return [row[0] if isinstance(row, list) else row for row in out_data]
        return out_data

    def score(self, X, y) -> float:
        """
        Return R² score (coefficient of determination).

        Args:
            X: Test features.
            y: True targets.

        Returns:
            R² score.
        """
        if hasattr(y, 'tolist'):
            y = y.tolist()
        predictions = self.predict(X)
        from .metrics import r2_score
        y_flat = [t[0] if isinstance(t, (list, tuple)) else t for t in y]
        p_flat = [p[0] if isinstance(p, (list, tuple)) else p for p in predictions]
        return r2_score(y_flat, p_flat)

    def rmse(self, X, y) -> float:
        """
        Return RMSE for the given data.

        Args:
            X: Test features.
            y: True targets.

        Returns:
            Root mean squared error.
        """
        if hasattr(y, 'tolist'):
            y = y.tolist()
        predictions = self.predict(X)
        from .metrics import root_mean_squared_error
        y_flat = [t[0] if isinstance(t, (list, tuple)) else t for t in y]
        p_flat = [p[0] if isinstance(p, (list, tuple)) else p for p in predictions]
        return root_mean_squared_error(y_flat, p_flat)
