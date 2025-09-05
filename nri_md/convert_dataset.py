"""
Convert PDB trajectory data to training format.

Updated for Python 3.10+ compatibility.
"""

import argparse
import numpy as np
import os
from typing import List, Tuple, Optional


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert PDB trajectory data to training format'
    )
    parser.add_argument('--data-dir', type=str, default='data/pdb',
                        help='Directory containing PDB files')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for processed data')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of data for training')
    parser.add_argument('--valid-ratio', type=float, default=0.15,
                        help='Ratio of data for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Ratio of data for testing')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of timesteps per sample')
    parser.add_argument('--dims', type=int, default=6,
                        help='Number of dimensions (position + velocity)')
    
    return parser.parse_args()


def read_pdb_trajectory(pdb_file: str) -> Tuple[np.ndarray, int]:
    """
    Read PDB trajectory file and extract coordinates.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        Tuple of (coordinates array, number of atoms)
    """
    coordinates = []
    current_frame = []
    num_atoms = 0
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_frame.append([x, y, z])
            elif line.startswith('ENDMDL') or line.startswith('END'):
                if current_frame:
                    coordinates.append(current_frame)
                    if num_atoms == 0:
                        num_atoms = len(current_frame)
                    current_frame = []
    
    # Handle case where file doesn't end with ENDMDL
    if current_frame:
        coordinates.append(current_frame)
        if num_atoms == 0:
            num_atoms = len(current_frame)
    
    return np.array(coordinates), num_atoms


def calculate_velocities(coordinates: np.ndarray) -> np.ndarray:
    """
    Calculate velocities from coordinates.
    
    Args:
        coordinates: Array of shape (timesteps, atoms, 3)
        
    Returns:
        Velocities array of shape (timesteps-1, atoms, 3)
    """
    velocities = np.diff(coordinates, axis=0)
    return velocities


def create_features_and_edges(coordinates: np.ndarray, velocities: np.ndarray, 
                            timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create feature and edge arrays from coordinates and velocities.
    
    Args:
        coordinates: Coordinate array
        velocities: Velocity array
        timesteps: Number of timesteps per sample
        
    Returns:
        Tuple of (features, edges)
    """
    num_frames = min(len(coordinates), len(velocities) + 1)
    num_atoms = coordinates.shape[1]
    
    # Ensure we have enough frames
    if num_frames < timesteps:
        print(f"Warning: Only {num_frames} frames available, but {timesteps} requested")
        timesteps = num_frames
    
    # Create samples with sliding window
    num_samples = max(1, num_frames - timesteps + 1)
    features = np.zeros((num_samples, timesteps, 6, num_atoms))
    
    for i in range(num_samples):
        # Position features
        features[i, :, :3, :] = coordinates[i:i+timesteps].transpose(0, 2, 1)
        # Velocity features (use last available velocity for final timestep)
        vel_end = min(i+timesteps-1, len(velocities)-1)
        for t in range(timesteps):
            vel_idx = min(i+t, vel_end)
            features[i, t, 3:6, :] = velocities[vel_idx].T
    
    # Create dummy edges (all connected for now)
    edges = np.ones((num_samples, num_atoms, num_atoms)) - np.eye(num_atoms)
    
    return features, edges


def split_data(features: np.ndarray, edges: np.ndarray, 
               train_ratio: float, valid_ratio: float, test_ratio: float) -> Tuple[
                   Tuple[np.ndarray, np.ndarray],
                   Tuple[np.ndarray, np.ndarray], 
                   Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        features: Feature array
        edges: Edge array
        train_ratio: Training data ratio
        valid_ratio: Validation data ratio
        test_ratio: Test data ratio
        
    Returns:
        Tuple of (train, valid, test) data tuples
    """
    num_samples = features.shape[0]
    
    # Calculate split indices
    train_end = int(num_samples * train_ratio)
    valid_end = train_end + int(num_samples * valid_ratio)
    
    # Split data
    train_features = features[:train_end]
    train_edges = edges[:train_end]
    
    valid_features = features[train_end:valid_end]
    valid_edges = edges[train_end:valid_end]
    
    test_features = features[valid_end:]
    test_edges = edges[valid_end:]
    
    return ((train_features, train_edges), 
            (valid_features, valid_edges), 
            (test_features, test_edges))


def main():
    """Main conversion function."""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find PDB files
    pdb_files = []
    if os.path.isdir(args.data_dir):
        for file in os.listdir(args.data_dir):
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(args.data_dir, file))
    else:
        print(f"Error: {args.data_dir} is not a directory")
        return
    
    if not pdb_files:
        print(f"No PDB files found in {args.data_dir}")
        return
    
    print(f"Found {len(pdb_files)} PDB files")
    
    all_features = []
    all_edges = []
    
    # Process each PDB file
    for pdb_file in pdb_files:
        print(f"Processing {pdb_file}...")
        
        try:
            coordinates, num_atoms = read_pdb_trajectory(pdb_file)
            print(f"  Found {len(coordinates)} frames with {num_atoms} atoms")
            
            if len(coordinates) < 2:
                print(f"  Skipping {pdb_file}: insufficient frames")
                continue
            
            velocities = calculate_velocities(coordinates)
            features, edges = create_features_and_edges(
                coordinates, velocities, args.timesteps
            )
            
            all_features.append(features)
            all_edges.append(edges)
            
            print(f"  Generated {features.shape[0]} samples")
            
        except Exception as e:
            print(f"  Error processing {pdb_file}: {e}")
            continue
    
    if not all_features:
        print("No valid data generated")
        return
    
    # Combine all data
    features = np.concatenate(all_features, axis=0)
    edges = np.concatenate(all_edges, axis=0)
    
    print(f"Total samples: {features.shape[0]}")
    print(f"Feature shape: {features.shape}")
    print(f"Edge shape: {edges.shape}")
    
    # Split data
    (train_features, train_edges), (valid_features, valid_edges), (test_features, test_edges) = split_data(
        features, edges, args.train_ratio, args.valid_ratio, args.test_ratio
    )
    
    # Save data
    np.save(os.path.join(args.output_dir, 'features.npy'), train_features)
    np.save(os.path.join(args.output_dir, 'edges.npy'), train_edges)
    np.save(os.path.join(args.output_dir, 'features_valid.npy'), valid_features)
    np.save(os.path.join(args.output_dir, 'edges_valid.npy'), valid_edges)
    np.save(os.path.join(args.output_dir, 'features_test.npy'), test_features)
    np.save(os.path.join(args.output_dir, 'edges_test.npy'), test_edges)
    
    print(f"Data saved to {args.output_dir}")
    print(f"Training samples: {train_features.shape[0]}")
    print(f"Validation samples: {valid_features.shape[0]}")
    print(f"Test samples: {test_features.shape[0]}")


if __name__ == '__main__':
    main()