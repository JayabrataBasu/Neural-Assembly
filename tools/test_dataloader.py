#!/usr/bin/env python3
"""
Test suite for DataLoader functionality (Issue #17).

Tests:
- DataLoader basic batching
- DataLoader with shuffle
- DataLoader with drop_last
- DataLoader length calculation
- Multi-worker DataLoader
- TensorDataset
- Samplers (Sequential, Random, Batch)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyneural as pn


def test_dataloader_basic():
    """Test basic DataLoader functionality."""
    print("Testing DataLoader basic batching...", end=" ")
    
    pn.init()
    
    # Load dataset
    dataset = pn.Dataset.from_csv(
        "csv/xor_train.csv",
        "csv/xor_labels.csv"
    )
    
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Create DataLoader
    loader = pn.DataLoader(dataset, batch_size=2)
    
    # Check length
    expected_batches = (len(dataset) + 1) // 2  # ceil division
    assert len(loader) == expected_batches, f"Expected {expected_batches} batches, got {len(loader)}"
    
    # Iterate
    batch_count = 0
    for batch_x, batch_y in loader:
        assert batch_x is not None, "batch_x should not be None"
        batch_count += 1
    
    assert batch_count == len(loader), f"Should iterate {len(loader)} times, got {batch_count}"
    
    print("PASSED")


def test_dataloader_shuffle():
    """Test DataLoader with shuffle."""
    print("Testing DataLoader shuffle...", end=" ")
    
    dataset = pn.Dataset.from_csv("csv/xor_train.csv", "csv/xor_labels.csv")
    
    # Create two loaders, one shuffled
    loader_no_shuffle = pn.DataLoader(dataset, batch_size=1, shuffle=False)
    loader_shuffle = pn.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Both should have same length
    assert len(loader_no_shuffle) == len(loader_shuffle)
    
    # Can iterate both
    no_shuffle_count = sum(1 for _ in loader_no_shuffle)
    shuffle_count = sum(1 for _ in loader_shuffle)
    
    assert no_shuffle_count == shuffle_count
    
    print("PASSED")


def test_dataloader_drop_last():
    """Test DataLoader with drop_last."""
    print("Testing DataLoader drop_last...", end=" ")
    
    dataset = pn.Dataset.from_csv("csv/xor_train.csv", "csv/xor_labels.csv")
    dataset_size = len(dataset)
    
    # With drop_last=True, incomplete last batch is dropped
    batch_size = 3
    loader_keep = pn.DataLoader(dataset, batch_size=batch_size, drop_last=False)
    loader_drop = pn.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    expected_keep = (dataset_size + batch_size - 1) // batch_size
    expected_drop = dataset_size // batch_size
    
    assert len(loader_keep) == expected_keep, f"Expected {expected_keep}, got {len(loader_keep)}"
    assert len(loader_drop) == expected_drop, f"Expected {expected_drop}, got {len(loader_drop)}"
    
    print("PASSED")


def test_sequential_sampler():
    """Test SequentialSampler."""
    print("Testing SequentialSampler...", end=" ")
    
    dataset = pn.Dataset.from_csv("csv/xor_train.csv", "csv/xor_labels.csv")
    
    sampler = pn.SequentialSampler(dataset)
    
    assert len(sampler) == len(dataset)
    
    indices = list(sampler)
    assert indices == list(range(len(dataset)))
    
    print("PASSED")


def test_random_sampler():
    """Test RandomSampler."""
    print("Testing RandomSampler...", end=" ")
    
    dataset = pn.Dataset.from_csv("csv/xor_train.csv", "csv/xor_labels.csv")
    
    sampler = pn.RandomSampler(dataset)
    
    assert len(sampler) == len(dataset)
    
    indices = list(sampler)
    assert len(indices) == len(dataset)
    
    # All indices should be valid
    for idx in indices:
        assert 0 <= idx < len(dataset)
    
    print("PASSED")


def test_batch_sampler():
    """Test BatchSampler."""
    print("Testing BatchSampler...", end=" ")
    
    dataset = pn.Dataset.from_csv("csv/xor_train.csv", "csv/xor_labels.csv")
    
    base_sampler = pn.SequentialSampler(dataset)
    batch_sampler = pn.BatchSampler(base_sampler, batch_size=2, drop_last=False)
    
    expected_batches = (len(dataset) + 1) // 2
    assert len(batch_sampler) == expected_batches
    
    batches = list(batch_sampler)
    for batch in batches:
        assert len(batch) <= 2
        for idx in batch:
            assert 0 <= idx < len(dataset)
    
    print("PASSED")


def test_tensor_dataset():
    """Test TensorDataset."""
    print("Testing TensorDataset...", end=" ")
    
    # Create tensors
    x = pn.Tensor.from_list([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = pn.Tensor.from_list([0.0, 1.0, 0.0])
    
    dataset = pn.TensorDataset(x, y)
    
    assert len(dataset) == 3
    
    # Get samples
    sample = dataset[0]
    assert len(sample) == 2  # x and y
    
    # Negative indexing
    sample = dataset[-1]
    assert len(sample) == 2
    
    print("PASSED")


def test_tensor_dataset_size_mismatch():
    """Test TensorDataset with mismatched sizes."""
    print("Testing TensorDataset size mismatch...", end=" ")
    
    x = pn.Tensor.from_list([[1.0, 2.0], [3.0, 4.0]])  # 2 samples
    y = pn.Tensor.from_list([0.0, 1.0, 2.0])  # 3 samples
    
    try:
        dataset = pn.TensorDataset(x, y)
        print("FAILED - should raise ValueError")
        return
    except ValueError:
        pass  # Expected
    
    print("PASSED")


def test_dataloader_repr():
    """Test DataLoader string representation."""
    print("Testing DataLoader repr...", end=" ")
    
    dataset = pn.Dataset.from_csv("csv/xor_train.csv", "csv/xor_labels.csv")
    loader = pn.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    
    repr_str = repr(loader)
    assert "batch_size=32" in repr_str
    assert "shuffle=True" in repr_str
    assert "num_workers=2" in repr_str
    
    print("PASSED")


def test_dataset_get_batch():
    """Test Dataset.get_batch directly."""
    print("Testing Dataset.get_batch...", end=" ")
    
    dataset = pn.Dataset.from_csv("csv/xor_train.csv", "csv/xor_labels.csv")
    
    # Get first batch
    x, y = dataset.get_batch(0, 2)
    
    assert x is not None, "x should not be None"
    assert x.shape[0] == 2, f"Batch size should be 2, got {x.shape[0]}"
    
    print("PASSED")


def test_dataloader_multiworker():
    """Test DataLoader with multiple workers."""
    print("Testing DataLoader multi-worker...", end=" ")
    
    dataset = pn.Dataset.from_csv("csv/xor_train.csv", "csv/xor_labels.csv")
    
    # Create loader with 2 workers
    loader = pn.DataLoader(dataset, batch_size=1, num_workers=2)
    
    # Iterate
    batch_count = 0
    for batch_x, batch_y in loader:
        assert batch_x is not None
        batch_count += 1
    
    assert batch_count == len(loader), f"Expected {len(loader)} batches, got {batch_count}"
    
    print("PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("DataLoader Functionality Tests (Issue #17)")
    print("=" * 60)
    
    pn.init()
    
    tests = [
        test_dataloader_basic,
        test_dataloader_shuffle,
        test_dataloader_drop_last,
        test_sequential_sampler,
        test_random_sampler,
        test_batch_sampler,
        test_tensor_dataset,
        test_tensor_dataset_size_mismatch,
        test_dataloader_repr,
        test_dataset_get_batch,
        test_dataloader_multiworker,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    pn.shutdown()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
