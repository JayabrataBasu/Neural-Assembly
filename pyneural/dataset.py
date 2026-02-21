"""
Dataset and DataLoader utilities for PyNeural.

Provides:
- Dataset: Wrapper for native dataset loading from CSV
- DataLoader: Iterable batching with shuffling and multi-worker support
- TensorDataset: Create dataset directly from Tensors
- DatasetSampler: Various sampling strategies
"""

import ctypes
import random
import queue
import threading
from pathlib import Path
from typing import (
    Optional, Tuple, Iterator, List, Callable, Union, Any
)

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
        labels_path: Optional[str] = None,
        n_features: Optional[int] = None,
        dtype: int = 0  # FLOAT32 = 0
    ) -> "Dataset":
        """
        Load a dataset from CSV files.
        
        Args:
            data_path: Path to data CSV file
            labels_path: Path to labels CSV file (optional)
            n_features: Number of features per sample (auto-detected if None)
            dtype: Data type (0 = float32, 1 = float64)
        
        Returns:
            Dataset instance
        """
        data_path = str(Path(data_path).resolve())
        if labels_path:
            labels_path = str(Path(labels_path).resolve())
        
        # Auto-detect n_features from first line if not provided
        if n_features is None:
            with open(data_path, 'r') as f:
                first_line = f.readline().strip()
                n_features = len(first_line.split(','))
        
        labels_bytes = labels_path.encode("utf-8") if labels_path else None
        
        ptr = _lib.neural_dataset_load_csv(
            data_path.encode("utf-8"),
            labels_bytes,
            n_features,
            dtype
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
        batch_size: int
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Get a batch of data from the dataset.
        
        Args:
            batch_idx: Index of the batch
            batch_size: Size of the batch
        
        Returns:
            Tuple of (data_tensor, labels_tensor)
            labels_tensor may be None if dataset has no labels
        """
        out_x = ctypes.c_void_p()
        out_y = ctypes.c_void_p()
        
        result = _lib.neural_dataset_get_batch(
            self._ptr,
            batch_idx,
            batch_size,
            ctypes.byref(out_x),
            ctypes.byref(out_y)
        )
        
        _check_error(result, "get_batch")
        
        x = Tensor(out_x.value, owns_data=True) if out_x.value else None
        y = Tensor(out_y.value, owns_data=True) if out_y.value else None
        
        return x, y
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Get a single sample by index."""
        return self.get_batch(idx, 1)
    
    def __repr__(self) -> str:
        return f"Dataset(size={len(self)})"


class TensorDataset:
    """
    Dataset wrapping Tensors.
    
    Each sample is retrieved by indexing tensors along the first dimension.
    
    Args:
        *tensors: Tensors that have the same size in the first dimension.
    
    Example:
        >>> x = Tensor.from_list([[1, 2], [3, 4], [5, 6]])
        >>> y = Tensor.from_list([0, 1, 0])
        >>> dataset = TensorDataset(x, y)
        >>> len(dataset)
        3
    """
    
    def __init__(self, *tensors: Tensor):
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one tensor")
        
        # Verify all tensors have the same size in dimension 0
        size = tensors[0].shape[0]
        for i, t in enumerate(tensors[1:], 1):
            if t.shape[0] != size:
                raise ValueError(
                    f"Size mismatch: tensor 0 has {size} samples, "
                    f"tensor {i} has {t.shape[0]} samples"
                )
        
        self.tensors = tensors
        self._size = size
    
    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """Get sample by index - returns tuple of tensor values at that index."""
        if idx < 0:
            idx += self._size
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Index {idx} out of range [0, {self._size})")
        
        # Return values at index as nested lists
        return tuple(t.tolist()[idx] if hasattr(t, 'tolist') else t for t in self.tensors)
    
    def __repr__(self) -> str:
        return f"TensorDataset({len(self.tensors)} tensors, {len(self)} samples)"


class Sampler:
    """Base class for all samplers."""
    
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Sample elements sequentially, always in the same order."""
    
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))
    
    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Sample elements randomly without replacement."""
    
    def __init__(self, data_source, replacement: bool = False, num_samples: int = None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
    
    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.replacement:
            return iter([random.randint(0, n - 1) for _ in range(self.num_samples)])
        return iter(random.sample(range(n), min(n, self.num_samples)))
    
    def __len__(self) -> int:
        return self.num_samples


class WeightedRandomSampler(Sampler):
    """
    Sample elements with class-balanced weighting (assembly-backed).

    Computes inverse-frequency class weights via ``compute_class_weights``
    in training_ops.asm, then draws ``num_samples`` indices using
    ``weighted_sample_indices`` (cumulative-distribution sampling).

    This oversamples minority classes and undersamples majority classes,
    yielding a balanced training distribution even for imbalanced datasets.

    Args:
        labels: List or array of integer class labels for every sample.
        num_classes: Total number of classes.
        num_samples: How many indices to draw per epoch (default: len(labels)).
        replacement: Must be True (weighted sampling always uses replacement).
        seed: Optional RNG seed for reproducibility. 0 = time-based.

    Example:
        >>> sampler = WeightedRandomSampler(train_labels, num_classes=6)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    """

    def __init__(
        self,
        labels: List[int],
        num_classes: int,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
    ):
        self._labels = labels
        self.num_classes = num_classes
        self._num_samples = num_samples if num_samples is not None else len(labels)
        self.seed = seed

        # Pre-compute class weights and per-sample weights (assembly)
        n = len(labels)
        label_arr = (ctypes.c_int32 * n)(*labels)
        class_w = (ctypes.c_double * num_classes)()
        _lib.neural_compute_class_weights(label_arr, n, num_classes, class_w)

        self._sample_weights = (ctypes.c_double * n)()
        _lib.neural_compute_sample_weights(
            label_arr, n, class_w, num_classes, self._sample_weights
        )

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n_in = len(self._labels)
        out = (ctypes.c_int32 * self._num_samples)()
        _lib.neural_weighted_sample_indices(
            self._sample_weights, n_in, self._num_samples, out, self.seed
        )
        return iter(out[i] for i in range(self._num_samples))

    def __len__(self) -> int:
        return self._num_samples


class BatchSampler(Sampler):
    """Wraps another sampler to yield batches of indices."""
    
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class _WorkerResult:
    """Result from a worker thread."""
    def __init__(self, batch_idx: int, data: Any, error: Optional[Exception] = None):
        self.batch_idx = batch_idx
        self.data = data
        self.error = error


def _worker_loop(
    dataset: Dataset,
    index_queue: queue.Queue,
    output_queue: queue.Queue,
    batch_size: int
):
    """Worker thread function for loading data."""
    while True:
        try:
            batch_idx = index_queue.get(timeout=0.1)
            if batch_idx is None:  # Shutdown signal
                break
            
            # Load batch
            data = dataset.get_batch(batch_idx, batch_size)
            output_queue.put(_WorkerResult(batch_idx, data))
            
        except queue.Empty:
            continue
        except Exception as e:
            output_queue.put(_WorkerResult(batch_idx if 'batch_idx' in dir() else -1, None, e))


class DataLoader:
    """
    Data loader combining dataset and sampler for batching and iteration.
    
    Args:
        dataset: Dataset to load from
        batch_size: Number of samples per batch (default: 1)
        shuffle: Whether to shuffle at each epoch (default: False)
        sampler: Custom sampler (mutually exclusive with shuffle)
        batch_sampler: Custom batch sampler (if provided, batch_size is ignored)
        num_workers: Number of worker threads for loading (default: 0 = main thread)
        drop_last: Drop the last incomplete batch (default: False)
        prefetch_factor: Number of batches to prefetch per worker
    
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
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        drop_last: bool = False,
        prefetch_factor: int = 2
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self._indices = list(range(len(dataset)))
        
        # Worker management
        self._workers: List[threading.Thread] = []
        self._index_queue: Optional[queue.Queue] = None
        self._output_queue: Optional[queue.Queue] = None
    
    def __len__(self) -> int:
        """Get the number of batches."""
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size
    
    def _start_workers(self):
        """Start worker threads for prefetching."""
        if self.num_workers == 0:
            return
        
        self._index_queue = queue.Queue(maxsize=self.num_workers * self.prefetch_factor)
        self._output_queue = queue.Queue(maxsize=self.num_workers * self.prefetch_factor)
        
        for _ in range(self.num_workers):
            worker = threading.Thread(
                target=_worker_loop,
                args=(
                    self.dataset,
                    self._index_queue,
                    self._output_queue,
                    self.batch_size
                ),
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def _stop_workers(self):
        """Stop worker threads."""
        if not self._workers:
            return
        
        # Send shutdown signals
        for _ in self._workers:
            try:
                self._index_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=1.0)
        
        self._workers.clear()
        self._index_queue = None
        self._output_queue = None
    
    def __iter__(self) -> Iterator[Tuple[Tensor, Optional[Tensor]]]:
        """Iterate over batches."""
        # Shuffle indices if needed
        if self.shuffle:
            random.shuffle(self._indices)
        
        if self.num_workers > 0:
            return self._iter_multiworker()
        return self._iter_singleworker()
    
    def _iter_singleworker(self) -> Iterator[Tuple[Tensor, Optional[Tensor]]]:
        """Single-threaded iteration."""
        for batch_idx in range(len(self)):
            yield self.dataset.get_batch(batch_idx, self.batch_size)
    
    def _iter_multiworker(self) -> Iterator[Tuple[Tensor, Optional[Tensor]]]:
        """Multi-threaded iteration with prefetching."""
        try:
            self._start_workers()
            
            # Submit initial batch indices
            next_batch_idx = 0
            num_submitted = 0
            
            for batch_idx in range(min(len(self), self.num_workers * self.prefetch_factor)):
                self._index_queue.put(batch_idx)
                num_submitted += 1
                next_batch_idx = batch_idx + 1
            
            # Yield results in order
            output_idx = 0
            results = {}
            
            while output_idx < len(self):
                if output_idx in results:
                    # Already have this result
                    result = results.pop(output_idx)
                    output_idx += 1
                    
                    # Submit more work if available
                    if next_batch_idx < len(self):
                        self._index_queue.put(next_batch_idx)
                        next_batch_idx += 1
                    
                    if result.error:
                        raise result.error
                    yield result.data
                else:
                    # Wait for next result
                    result = self._output_queue.get()
                    if result.batch_idx == output_idx:
                        output_idx += 1
                        
                        if next_batch_idx < len(self):
                            self._index_queue.put(next_batch_idx)
                            next_batch_idx += 1
                        
                        if result.error:
                            raise result.error
                        yield result.data
                    else:
                        # Out of order - store for later
                        results[result.batch_idx] = result
        finally:
            self._stop_workers()
    
    def __repr__(self) -> str:
        return (
            f"DataLoader(dataset={self.dataset}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, num_workers={self.num_workers})"
        )
