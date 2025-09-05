"""
Tests for utility functions.
"""

import pytest
import numpy as np
import torch
from nri_md.utils import (
    my_softmax, 
    binary_concrete, 
    gumbel_softmax,
    get_offdiag_indices,
    binary_accuracy
)


def test_my_softmax():
    """Test custom softmax function."""
    x = torch.randn(2, 3, 4)
    result = my_softmax(x, axis=1)
    
    # Check shape is preserved
    assert result.shape == x.shape
    
    # Check softmax property (sum to 1 along axis)
    assert torch.allclose(result.sum(dim=1), torch.ones(2, 4), atol=1e-6)


def test_binary_concrete():
    """Test binary concrete distribution."""
    logits = torch.randn(10, 2)
    
    # Test soft sampling
    result_soft = binary_concrete(logits, tau=1.0, hard=False)
    assert result_soft.shape == logits.shape
    assert torch.all(result_soft >= 0) and torch.all(result_soft <= 1)
    
    # Test hard sampling
    result_hard = binary_concrete(logits, tau=1.0, hard=True)
    assert result_hard.shape == logits.shape
    # Hard samples should be binary (0 or 1)
    assert torch.all((result_hard == 0) | (result_hard == 1))


def test_gumbel_softmax():
    """Test Gumbel softmax function."""
    logits = torch.randn(5, 3)
    
    # Test soft sampling
    result_soft = gumbel_softmax(logits, tau=1.0, hard=False)
    assert result_soft.shape == logits.shape
    assert torch.allclose(result_soft.sum(dim=-1), torch.ones(5), atol=1e-6)
    
    # Test hard sampling
    result_hard = gumbel_softmax(logits, tau=1.0, hard=True)
    assert result_hard.shape == logits.shape
    # Each row should sum to 1 and be one-hot
    assert torch.allclose(result_hard.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_get_offdiag_indices():
    """Test off-diagonal indices function."""
    num_nodes = 4
    indices = get_offdiag_indices(num_nodes)
    
    # Should have num_nodes * (num_nodes - 1) indices
    expected_length = num_nodes * (num_nodes - 1)
    assert len(indices) == expected_length
    
    # All indices should be valid
    assert np.all(indices >= 0)
    assert np.all(indices < num_nodes * num_nodes)


def test_binary_accuracy():
    """Test binary accuracy function."""
    # Perfect predictions
    output = torch.tensor([0.1, 0.9, 0.2, 0.8])
    labels = torch.tensor([0, 1, 0, 1])
    acc = binary_accuracy(output, labels)
    assert acc == 1.0
    
    # Half correct predictions
    output = torch.tensor([0.1, 0.1, 0.9, 0.9])
    labels = torch.tensor([0, 1, 0, 1])
    acc = binary_accuracy(output, labels)
    assert acc == 0.5


def test_tensor_operations():
    """Test basic tensor operations work correctly."""
    x = torch.randn(3, 4, 5)
    
    # Test transpose
    y = x.transpose(1, 2)
    assert y.shape == (3, 5, 4)
    
    # Test contiguous
    z = y.contiguous()
    assert z.shape == y.shape
    
    # Test CUDA availability (should work regardless of CUDA availability)
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        assert x_cuda.is_cuda
        x_cpu = x_cuda.cpu()
        assert not x_cpu.is_cuda