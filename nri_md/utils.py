"""
Utility functions for NRI-MD model training and data processing.

Updated for Python 3.10+ and modern PyTorch compatibility.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def my_softmax(input_tensor: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """Apply softmax along specified axis."""
    trans_input = input_tensor.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10) -> torch.Tensor:
    """Binary concrete distribution sampling."""
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        # Use detach() instead of deprecated .data access
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits: torch.Tensor, tau: float = 1, eps: float = 1e-10) -> torch.Tensor:
    """Sample from binary concrete distribution."""
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)


def sample_logistic(shape: torch.Size, eps: float = 1e-10) -> torch.Tensor:
    """Sample from logistic distribution."""
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape: torch.Size, eps: float = 1e-10) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1, eps: float = 1e-10) -> torch.Tensor:
    """Draw a sample from the Gumbel-Softmax distribution."""
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + gumbel_noise
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10) -> torch.Tensor:
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        tau: non-negative scalar temperature
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        eps: small epsilon for numerical stability
    
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape, dtype=y_soft.dtype, device=y_soft.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # Use detach() instead of deprecated .data access
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y


def binary_accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate binary accuracy."""
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_offdiag_indices(num_nodes: int) -> np.ndarray:
    """Get off-diagonal indices for adjacency matrix."""
    ones = np.ones((num_nodes, num_nodes))
    eye = np.eye(num_nodes)
    off_diag_idx = np.ravel_multi_index(
        np.where(ones - eye), [num_nodes, num_nodes]
    )
    return off_diag_idx


def load_dataset_train(
    batch_size: int = 1, 
    suffix: str = '', 
    number_exp: int = 1, 
    dims: int = 6
) -> Tuple[DataLoader, float, float]:
    """Load training dataset."""
    feat_train = np.load('data/features.npy')
    edges_train = np.load('data/edges.npy')

    assert feat_train.shape[0] >= number_exp
    assert feat_train.shape[2] >= dims
    feat_train = feat_train[0:number_exp, :, 0:dims, :]
    edges_train = edges_train[0:number_exp, :, :]

    # [num_samples, num_timesteps, num_dims, num_genes]
    num_genes = feat_train.shape[3]

    feat_max = feat_train.max()
    feat_min = feat_train.min()

    # Normalize to [-1, 1]
    feat_train = (feat_train - feat_min) * 2 / (feat_max - feat_min) - 1

    # Reshape to: [num_samples, num_genes, num_timesteps, num_dims]
    feat_train = np.transpose(feat_train, [0, 3, 1, 2])
    edges_train = np.reshape(edges_train, [-1, num_genes ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    # Exclude self edges
    off_diag_idx = get_offdiag_indices(num_genes)
    edges_train = edges_train[:, off_diag_idx]

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)

    train_data = TensorDataset(feat_train, edges_train)
    train_data_loader = DataLoader(train_data, batch_size=batch_size)

    return train_data_loader, feat_max, feat_min


def load_dataset_train_valid_test(
    batch_size: int = 1, 
    number_exp: int = 1, 
    number_expstart: int = 0, 
    dims: int = 6
) -> Tuple[DataLoader, DataLoader, DataLoader, float, float, float, float]:
    """Load train, validation, and test datasets."""
    feat_train = np.load('data/features.npy')
    edges_train = np.load('data/edges.npy')
    feat_valid = np.load('data/features_valid.npy')
    edges_valid = np.load('data/edges_valid.npy')
    feat_test = np.load('data/features_test.npy')
    edges_test = np.load('data/edges_test.npy')

    assert feat_train.shape[0] >= number_exp
    assert feat_train.shape[2] >= dims
    feat_train = feat_train[number_expstart:number_exp, :, 0:dims, :]
    edges_train = edges_train[number_expstart:number_exp, :, :]
    feat_valid = feat_valid[number_expstart:number_exp, :, 0:dims, :]
    edges_valid = edges_valid[number_expstart:number_exp, :, :]
    feat_test = feat_test[:, :, 0:dims, :]

    # [num_samples, num_timesteps, num_dims, num_genes]
    num_genes = feat_train.shape[3]

    loc_max = feat_train[:, :, 0:3, :].max()
    loc_min = feat_train[:, :, 0:3, :].min()
    vel_max = feat_train[:, :, 3:6, :].max()
    vel_min = feat_train[:, :, 3:6, :].min()

    # Normalize to [-1, 1]
    loc_train = (feat_train[:, :, 0:3, :] - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (feat_train[:, :, 3:6, :] - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (feat_valid[:, :, 0:3, :] - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (feat_valid[:, :, 3:6, :] - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (feat_test[:, :, 0:3, :] - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (feat_test[:, :, 3:6, :] - vel_min) * 2 / (vel_max - vel_min) - 1

    feat_train = np.concatenate((loc_train, vel_train), axis=2)
    feat_valid = np.concatenate((loc_valid, vel_valid), axis=2)
    feat_test = np.concatenate((loc_test, vel_test), axis=2)

    # Reshape to: [num_samples, num_genes, num_timesteps, num_dims]
    feat_train = np.transpose(feat_train, [0, 3, 1, 2])
    feat_valid = np.transpose(feat_valid, [0, 3, 1, 2])
    feat_test = np.transpose(feat_test, [0, 3, 1, 2])

    edges_train = np.reshape(edges_train, [-1, num_genes ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_genes ** 2])
    edges_test = np.reshape(edges_test, [-1, num_genes ** 2])

    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    # Exclude self edges
    off_diag_idx = get_offdiag_indices(num_genes)
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    feat_train = torch.FloatTensor(feat_train)
    feat_valid = torch.FloatTensor(feat_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_train = torch.LongTensor(edges_train)
    edges_valid = torch.LongTensor(edges_valid)
    edges_test = torch.LongTensor(edges_test)

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return (train_data_loader, valid_data_loader, test_data_loader, 
            loc_max, loc_min, vel_max, vel_min)


def encode_onehot(labels: np.ndarray) -> np.ndarray:
    """Encode labels as one-hot vectors."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def get_triu_offdiag_indices(num_atoms: int) -> np.ndarray:
    """Get upper triangular off-diagonal indices."""
    ones = np.ones((num_atoms, num_atoms))
    eye = np.eye(num_atoms)
    triu = np.triu(ones, k=1)
    indices = np.where(triu)
    return np.ravel_multi_index(indices, [num_atoms, num_atoms])


def get_tril_offdiag_indices(num_atoms: int) -> np.ndarray:
    """Get lower triangular off-diagonal indices."""
    ones = np.ones((num_atoms, num_atoms))
    eye = np.eye(num_atoms)
    tril = np.tril(ones, k=-1)
    indices = np.where(tril)
    return np.ravel_multi_index(indices, [num_atoms, num_atoms])