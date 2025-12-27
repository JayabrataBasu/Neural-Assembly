"""
Dataset utilities for PyNeural.
"""

import ctypes
from pathlib import Path
from typing import Optional, Tuple

from .core import _lib, NeuralException, _check_error
from .tensor import Tensor


class Dataset:
    """
    Dataset for loading and batching training data.
    
    Example:
        >>> dataset = Dataset.from_csv("train.csv", "labels.csv")
        >>> print(f"Dataset size: {len(dataset)}")
        >>> data, labels = dataset.get_batch(0, batch_size=32)
    """
    
    def __init__(self, ptr: ctypes.c_void_p):
        """
        Create a Dataset wrapper around a native dataset pointer.
        
        Args:
            ptr: Pointer to native NeuralDataset
        """
        if ptr is None or ptr == 0:
            raise NeuralException(1, "Null dataset pointer")
        self._ptr = ptr
    
    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            _lib.neural_dataset_free(self._ptr)
            self._ptr = None
    
    @classmethod
    def from_csv(
        cls,
        data_path: str,
        labels_path: Optional[str] = None
    ) -> "Dataset":
        """
        Load a dataset from CSV files.
        
        Args:
            data_path: Path to data CSV file
            labels_path: Path to labels CSV file (optional)
        
        Returns:
            Dataset instance
        """
        data_path = str(Path(data_path).resolve())
        labels_bytes = labels_path.encode("utf-8") if labels_path else None
        
        ptr = _lib.neural_dataset_load_csv(
            data_path.encode("utf-8"),
            labels_bytes
        )
        
        if ptr is None or ptr == 0:
            raise NeuralException(
                _lib.neural_get_last_error(),
                f"Failed to load dataset from {data_path}"
            )
        
        return cls(ptr)
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return _lib.neural_dataset_size(self._ptr)
    
    @property
    def size(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self)
    
    def get_batch(
        self,
        batch_idx: int,
        batch_size: int,
        data_shape: Tuple[int, ...] = None,
        labels_shape: Tuple[int, ...] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Get a batch of data from the dataset.
        
        Args:
            batch_idx: Index of the batch
            batch_size: Size of the batch
            data_shape: Shape of data tensor (optional)
            labels_shape: Shape of labels tensor (optional)
        
        Returns:
            Tuple of (data_tensor, labels_tensor)
            labels_tensor may be None if dataset has no labels
        """
        # This is a placeholder - actual implementation depends on native API
        raise NotImplementedError("get_batch not yet implemented in native API")
    
    def __repr__(self) -> str:
        return f"Dataset(size={len(self)})"


class DataLoader:
    """
    Data loader for batching and iterating over datasets.
    
    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data each epoch
    
    Example:
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch_data, batch_labels in loader:
        ...     output = model(batch_data)
        ...     loss = criterion(output, batch_labels)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = list(range(len(dataset)))
    
    def __len__(self) -> int:
        """Get the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        if self.shuffle:
            import random
            random.shuffle(self._indices)
        
        for i in range(len(self)):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            batch_indices = self._indices[start_idx:end_idx]
            
            # Get batch - placeholder for actual implementation
            yield self.dataset.get_batch(i, len(batch_indices))
    
    def __repr__(self) -> str:
        return f"DataLoader(dataset={self.dataset}, batch_size={self.batch_size}, shuffle={self.shuffle})"
