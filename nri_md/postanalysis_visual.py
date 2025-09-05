"""
Visualize the distribution of learned edges between residues.

Updated for Python 3.10+ compatibility.
"""

import argparse
import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize the distribution of learned edges between residues.'
    )
    parser.add_argument('--num-residues', type=int, default=77,
                        help='Number of residues of the PDB.')
    parser.add_argument('--windowsize', type=int, default=56,
                        help='Window size')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold for plotting')
    parser.add_argument('--dist-threshold', type=int, default=12,
                        help='Threshold for shortest distance')
    parser.add_argument('--filename', type=str, default='logs/out_probs_train.npy',
                        help='File name of the probs file.')
    parser.add_argument('--output-dir', type=str, default='logs',
                        help='Output directory for plots')
    
    return parser.parse_args()


def get_edge_results(filename: str, threshold: Optional[float] = None) -> np.ndarray:
    """
    Extract edge probabilities from saved results.
    
    Args:
        filename: Path to the probability file
        threshold: Optional threshold for filtering
        
    Returns:
        Edge probability matrix
    """
    try:
        a = np.load(filename)
        # There are four types of edges, eliminate the first type as the non-edge
        b = a[:, :, 1]
        c = a[:, :, 2]
        d = a[:, :, 3]
        
        # Sum probabilities for edge types 1, 2, 3
        probs = b + c + d
        
        if threshold is not None:
            probs = (probs > threshold).astype(float)
        
        return probs
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return np.array([])
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return np.array([])


def create_adjacency_matrix(probs: np.ndarray, num_residues: int) -> np.ndarray:
    """
    Create adjacency matrix from edge probabilities.
    
    Args:
        probs: Edge probability array
        num_residues: Number of residues
        
    Returns:
        Adjacency matrix
    """
    adj_matrix = np.zeros((num_residues, num_residues))
    
    # Fill upper triangle (excluding diagonal)
    idx = 0
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            adj_matrix[i, j] = probs[idx]
            adj_matrix[j, i] = probs[idx]  # Make symmetric
            idx += 1
    
    return adj_matrix


def plot_adjacency_heatmap(adj_matrix: np.ndarray, output_path: str, 
                          title: str = "Learned Edge Probabilities") -> None:
    """
    Plot adjacency matrix as heatmap.
    
    Args:
        adj_matrix: Adjacency matrix
        output_path: Output file path
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(adj_matrix, 
                cmap='viridis', 
                cbar=True,
                square=True,
                xticklabels=10,
                yticklabels=10,
                cbar_kws={'label': 'Edge Probability'})
    
    plt.title(title, fontsize=16)
    plt.xlabel('Residue Index', fontsize=14)
    plt.ylabel('Residue Index', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved to {output_path}")


def plot_edge_distribution(probs: np.ndarray, output_path: str) -> None:
    """
    Plot distribution of edge probabilities.
    
    Args:
        probs: Edge probability array
        output_path: Output file path
    """
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Edge Probability', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Edge Probabilities', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_prob = np.mean(probs)
    std_prob = np.std(probs)
    plt.axvline(mean_prob, color='red', linestyle='--', 
                label=f'Mean: {mean_prob:.3f}')
    plt.axvline(mean_prob + std_prob, color='orange', linestyle='--', 
                label=f'Mean + Std: {mean_prob + std_prob:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plot saved to {output_path}")


def analyze_long_range_interactions(adj_matrix: np.ndarray, 
                                  dist_threshold: int = 12) -> Tuple[int, int, float]:
    """
    Analyze long-range vs short-range interactions.
    
    Args:
        adj_matrix: Adjacency matrix
        dist_threshold: Distance threshold for long-range interactions
        
    Returns:
        Tuple of (short_range_count, long_range_count, long_range_ratio)
    """
    num_residues = adj_matrix.shape[0]
    short_range_count = 0
    long_range_count = 0
    
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            if adj_matrix[i, j] > 0:
                distance = abs(j - i)
                if distance <= dist_threshold:
                    short_range_count += 1
                else:
                    long_range_count += 1
    
    total_interactions = short_range_count + long_range_count
    long_range_ratio = long_range_count / total_interactions if total_interactions > 0 else 0
    
    return short_range_count, long_range_count, long_range_ratio


def create_contact_map(adj_matrix: np.ndarray, output_path: str, 
                      threshold: float = 0.5) -> None:
    """
    Create contact map visualization.
    
    Args:
        adj_matrix: Adjacency matrix
        output_path: Output file path
        threshold: Threshold for showing contacts
    """
    # Binarize matrix based on threshold
    contact_matrix = (adj_matrix > threshold).astype(int)
    
    plt.figure(figsize=(12, 10))
    
    # Plot contact map
    plt.imshow(contact_matrix, cmap='Blues', origin='lower')
    plt.colorbar(label='Contact')
    
    plt.title(f'Contact Map (threshold = {threshold})', fontsize=16)
    plt.xlabel('Residue Index', fontsize=14)
    plt.ylabel('Residue Index', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Contact map saved to {output_path}")


def main():
    """Main visualization function."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load edge probabilities
    print(f"Loading edge probabilities from {args.filename}")
    probs = get_edge_results(args.filename)
    
    if probs.size == 0:
        print("No data to visualize")
        return
    
    print(f"Loaded {len(probs)} edge probabilities")
    print(f"Mean probability: {np.mean(probs):.4f}")
    print(f"Std probability: {np.std(probs):.4f}")
    print(f"Max probability: {np.max(probs):.4f}")
    print(f"Min probability: {np.min(probs):.4f}")
    
    # Create adjacency matrix
    adj_matrix = create_adjacency_matrix(probs, args.num_residues)
    
    # Generate visualizations
    base_name = os.path.splitext(os.path.basename(args.filename))[0]
    
    # 1. Adjacency heatmap
    heatmap_path = os.path.join(args.output_dir, f'{base_name}_heatmap.png')
    plot_adjacency_heatmap(adj_matrix, heatmap_path)
    
    # 2. Edge probability distribution
    dist_path = os.path.join(args.output_dir, f'{base_name}_distribution.png')
    plot_edge_distribution(probs, dist_path)
    
    # 3. Contact map
    contact_path = os.path.join(args.output_dir, f'{base_name}_contact_map.png')
    create_contact_map(adj_matrix, contact_path, args.threshold)
    
    # 4. Analyze long-range interactions
    short_range, long_range, lr_ratio = analyze_long_range_interactions(
        adj_matrix > args.threshold, args.dist_threshold
    )
    
    print(f"\nInteraction Analysis (threshold = {args.threshold}):")
    print(f"Short-range interactions (≤{args.dist_threshold}): {short_range}")
    print(f"Long-range interactions (>{args.dist_threshold}): {long_range}")
    print(f"Long-range ratio: {lr_ratio:.3f}")
    
    # Save analysis results
    analysis_path = os.path.join(args.output_dir, f'{base_name}_analysis.txt')
    with open(analysis_path, 'w') as f:
        f.write(f"Edge Probability Analysis\n")
        f.write(f"========================\n\n")
        f.write(f"Input file: {args.filename}\n")
        f.write(f"Number of residues: {args.num_residues}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Distance threshold: {args.dist_threshold}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"Mean probability: {np.mean(probs):.4f}\n")
        f.write(f"Std probability: {np.std(probs):.4f}\n")
        f.write(f"Max probability: {np.max(probs):.4f}\n")
        f.write(f"Min probability: {np.min(probs):.4f}\n\n")
        f.write(f"Interaction Analysis:\n")
        f.write(f"Short-range interactions (≤{args.dist_threshold}): {short_range}\n")
        f.write(f"Long-range interactions (>{args.dist_threshold}): {long_range}\n")
        f.write(f"Long-range ratio: {lr_ratio:.3f}\n")
    
    print(f"Analysis saved to {analysis_path}")
    print("Visualization complete!")


if __name__ == '__main__':
    main()