"""
Neural Relational Inference for Molecular Dynamics (NRI-MD)

A PyTorch implementation of neural relational inference to learn allosteric 
long-range interactions in proteins from molecular dynamics simulations.

This package provides tools for:
- Training neural relational inference models on MD trajectories
- Converting PDB trajectories to training data
- Visualizing learned protein interactions
- Finding shortest paths between residues

Authors: Jingxuan Zhu, Juexin Wang, Weiwei Han, Dong Xu
Paper: https://www.nature.com/articles/s41467-022-29331-3
"""

__version__ = "1.0.0"
__author__ = "Jingxuan Zhu, Juexin Wang, Weiwei Han, Dong Xu"

from . import modules
from . import utils

__all__ = ["modules", "utils"]