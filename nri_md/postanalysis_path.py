"""
Find shortest paths along domains in residues.

Updated for Python 3.10+ compatibility.
"""

import argparse
import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import networkx as nx


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Find shortest paths along domains in residues.'
    )
    parser.add_argument('--num-residues', type=int, default=77,
                        help='Number of residues of the PDB.')
    parser.add_argument('--windowsize', type=int, default=56,
                        help='Window size')
    parser.add_argument('--dist-threshold', type=int, default=12,
                        help='Threshold for shortest distance')
    parser.add_argument('--filename', type=str, default='logs/out_probs_train.npy',
                        help='File name of the probs file.')
    parser.add_argument('--source-node', type=int, default=46,
                        help='Source residue of the PDB')
    parser.add_argument('--output-filename', type=str, default='logs/source46.txt',
                        help='File of shortest path from source to targets')
    parser.add_argument('--edge-threshold', type=float, default=0.5,
                        help='Threshold for considering edges in graph')
    parser.add_argument('--target-nodes', type=str, default='',
                        help='Comma-separated list of target nodes (empty for all)')
    
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


def create_graph_from_probabilities(probs: np.ndarray, num_residues: int, 
                                  threshold: float = 0.5) -> nx.Graph:
    """
    Create NetworkX graph from edge probabilities.
    
    Args:
        probs: Edge probability array
        num_residues: Number of residues
        threshold: Threshold for including edges
        
    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    
    # Add all nodes
    G.add_nodes_from(range(num_residues))
    
    # Add edges based on probabilities
    idx = 0
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            if probs[idx] > threshold:
                # Use inverse probability as weight (shorter paths for higher probabilities)
                weight = 1.0 / (probs[idx] + 1e-10)
                G.add_edge(i, j, weight=weight, probability=probs[idx])
            idx += 1
    
    return G


def find_shortest_paths(G: nx.Graph, source: int, targets: Optional[List[int]] = None) -> Dict[int, Tuple[List[int], float]]:
    """
    Find shortest paths from source to targets.
    
    Args:
        G: NetworkX graph
        source: Source node
        targets: List of target nodes (None for all nodes)
        
    Returns:
        Dictionary mapping target -> (path, distance)
    """
    if targets is None:
        targets = list(G.nodes())
    
    paths = {}
    
    for target in targets:
        if target == source:
            continue
            
        try:
            # Find shortest path using Dijkstra's algorithm
            path = nx.shortest_path(G, source, target, weight='weight')
            distance = nx.shortest_path_length(G, source, target, weight='weight')
            paths[target] = (path, distance)
        except nx.NetworkXNoPath:
            # No path exists
            paths[target] = ([], float('inf'))
    
    return paths


def analyze_path_properties(G: nx.Graph, path: List[int]) -> Dict[str, float]:
    """
    Analyze properties of a path.
    
    Args:
        G: NetworkX graph
        path: List of nodes in path
        
    Returns:
        Dictionary of path properties
    """
    if len(path) < 2:
        return {'length': 0, 'avg_probability': 0, 'min_probability': 0}
    
    probabilities = []
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        if edge_data:
            probabilities.append(edge_data['probability'])
    
    return {
        'length': len(path) - 1,
        'avg_probability': np.mean(probabilities) if probabilities else 0,
        'min_probability': np.min(probabilities) if probabilities else 0,
        'sequence_distance': abs(path[-1] - path[0])
    }


def classify_interactions(path: List[int], dist_threshold: int = 12) -> str:
    """
    Classify interaction as short-range or long-range.
    
    Args:
        path: Path between residues
        dist_threshold: Distance threshold
        
    Returns:
        Classification string
    """
    if len(path) < 2:
        return "no_path"
    
    sequence_distance = abs(path[-1] - path[0])
    if sequence_distance <= dist_threshold:
        return "short_range"
    else:
        return "long_range"


def save_path_analysis(paths: Dict[int, Tuple[List[int], float]], 
                      G: nx.Graph, source: int, output_file: str,
                      dist_threshold: int = 12) -> None:
    """
    Save path analysis to file.
    
    Args:
        paths: Dictionary of paths
        G: NetworkX graph
        source: Source node
        output_file: Output file path
        dist_threshold: Distance threshold for classification
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(f"Shortest Path Analysis from Source Node {source}\n")
        f.write("=" * 60 + "\n\n")
        
        # Summary statistics
        reachable_targets = [t for t, (p, d) in paths.items() if len(p) > 0]
        unreachable_targets = [t for t, (p, d) in paths.items() if len(p) == 0]
        
        f.write(f"Summary:\n")
        f.write(f"Total targets: {len(paths)}\n")
        f.write(f"Reachable targets: {len(reachable_targets)}\n")
        f.write(f"Unreachable targets: {len(unreachable_targets)}\n\n")
        
        # Classify interactions
        short_range = []
        long_range = []
        
        for target, (path, distance) in paths.items():
            if len(path) > 0:
                classification = classify_interactions(path, dist_threshold)
                if classification == "short_range":
                    short_range.append(target)
                elif classification == "long_range":
                    long_range.append(target)
        
        f.write(f"Interaction Classification (threshold = {dist_threshold}):\n")
        f.write(f"Short-range interactions: {len(short_range)}\n")
        f.write(f"Long-range interactions: {len(long_range)}\n\n")
        
        # Detailed path information
        f.write("Detailed Path Information:\n")
        f.write("-" * 40 + "\n")
        
        for target in sorted(paths.keys()):
            path, distance = paths[target]
            
            if len(path) == 0:
                f.write(f"Target {target}: No path found\n")
                continue
            
            properties = analyze_path_properties(G, path)
            classification = classify_interactions(path, dist_threshold)
            
            f.write(f"Target {target} ({classification}):\n")
            f.write(f"  Path: {' -> '.join(map(str, path))}\n")
            f.write(f"  Path length: {properties['length']} edges\n")
            f.write(f"  Sequence distance: {properties['sequence_distance']}\n")
            f.write(f"  Weighted distance: {distance:.4f}\n")
            f.write(f"  Average edge probability: {properties['avg_probability']:.4f}\n")
            f.write(f"  Minimum edge probability: {properties['min_probability']:.4f}\n")
            f.write("\n")
        
        # Unreachable targets
        if unreachable_targets:
            f.write("Unreachable Targets:\n")
            f.write("-" * 20 + "\n")
            for target in sorted(unreachable_targets):
                f.write(f"Target {target}: No path found\n")


def main():
    """Main path analysis function."""
    args = parse_args()
    
    # Load edge probabilities
    print(f"Loading edge probabilities from {args.filename}")
    probs = get_edge_results(args.filename)
    
    if probs.size == 0:
        print("No data to analyze")
        return
    
    print(f"Loaded {len(probs)} edge probabilities")
    
    # Create graph
    print(f"Creating graph with edge threshold {args.edge_threshold}")
    G = create_graph_from_probabilities(probs, args.num_residues, args.edge_threshold)
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Parse target nodes
    if args.target_nodes:
        target_nodes = [int(x.strip()) for x in args.target_nodes.split(',')]
        print(f"Analyzing paths to specific targets: {target_nodes}")
    else:
        target_nodes = None
        print(f"Analyzing paths to all nodes")
    
    # Find shortest paths
    print(f"Finding shortest paths from source node {args.source_node}")
    paths = find_shortest_paths(G, args.source_node, target_nodes)
    
    # Analyze and save results
    save_path_analysis(paths, G, args.source_node, args.output_filename, args.dist_threshold)
    
    # Print summary
    reachable = sum(1 for p, d in paths.values() if len(p) > 0)
    total = len(paths)
    
    print(f"\nPath Analysis Complete:")
    print(f"Reachable targets: {reachable}/{total}")
    print(f"Results saved to: {args.output_filename}")
    
    # Print some example paths
    print(f"\nExample paths from node {args.source_node}:")
    count = 0
    for target, (path, distance) in sorted(paths.items()):
        if len(path) > 0 and count < 5:
            print(f"  To node {target}: {' -> '.join(map(str, path))} (distance: {distance:.3f})")
            count += 1


if __name__ == '__main__':
    main()