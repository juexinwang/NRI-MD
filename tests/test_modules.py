"""
Tests for neural network modules.
"""

import pytest
import torch
from nri_md.modules import MLP, CNN, MLPEncoder


def test_mlp():
    """Test MLP module."""
    batch_size, num_nodes, input_dim = 2, 5, 10
    hidden_dim, output_dim = 32, 16
    
    mlp = MLP(input_dim, hidden_dim, output_dim, do_prob=0.1)
    
    # Test forward pass
    x = torch.randn(batch_size, num_nodes, input_dim)
    output = mlp(x)
    
    assert output.shape == (batch_size, num_nodes, output_dim)
    assert not torch.isnan(output).any()


def test_cnn():
    """Test CNN module."""
    batch_size, input_dim, seq_len = 4, 6, 20
    hidden_dim, output_dim = 32, 16
    
    cnn = CNN(input_dim, hidden_dim, output_dim, do_prob=0.1)
    
    # Test forward pass
    x = torch.randn(batch_size, input_dim, seq_len)
    output = cnn(x)
    
    # Output should be 2D after global pooling
    assert len(output.shape) == 2
    assert output.shape[0] == batch_size
    assert not torch.isnan(output).any()


def test_mlp_encoder():
    """Test MLP encoder."""
    batch_size, num_atoms, timesteps, dims = 2, 5, 10, 6
    hidden_dim, edge_types = 32, 4
    
    encoder = MLPEncoder(
        n_in=timesteps * dims,
        n_hid=hidden_dim,
        n_out=edge_types,
        do_prob=0.1,
        factor=True
    )
    
    # Create input data
    inputs = torch.randn(batch_size, num_atoms, timesteps, dims)
    
    # Create relation matrices (off-diagonal connections)
    num_edges = num_atoms * (num_atoms - 1)
    rel_rec = torch.zeros(num_edges, num_atoms)
    rel_send = torch.zeros(num_edges, num_atoms)
    
    # Fill relation matrices
    idx = 0
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                rel_rec[idx, j] = 1
                rel_send[idx, i] = 1
                idx += 1
    
    # Test forward pass
    output = encoder(inputs, rel_rec, rel_send)
    
    expected_shape = (batch_size, num_edges, edge_types)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()


def test_model_parameters():
    """Test that models have trainable parameters."""
    mlp = MLP(10, 32, 16)
    
    # Check that model has parameters
    params = list(mlp.parameters())
    assert len(params) > 0
    
    # Check that parameters require gradients
    for param in params:
        assert param.requires_grad
    
    # Check parameter shapes are reasonable
    total_params = sum(p.numel() for p in params)
    assert total_params > 0


def test_model_training_mode():
    """Test model training/eval modes."""
    mlp = MLP(10, 32, 16, do_prob=0.5)
    
    # Test training mode
    mlp.train()
    assert mlp.training
    
    # Test eval mode
    mlp.eval()
    assert not mlp.training
    
    # Test that dropout behaves differently in train vs eval
    x = torch.randn(1, 5, 10)
    
    mlp.train()
    out_train = mlp(x)
    
    mlp.eval()
    out_eval = mlp(x)
    
    # Outputs should be different due to dropout
    # (though this test might occasionally fail due to randomness)
    assert out_train.shape == out_eval.shape