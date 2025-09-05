#!/usr/bin/env python3
"""
Installation and testing script for NRI-MD with Python 3.10+ support.

This script helps set up the environment and test the installation.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {cmd}")
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result.stdout.strip()
    else:
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            sys.exit(1)


def check_python_version():
    """Check if Python version is 3.10+."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("Error: Python 3.10+ is required")
        print("Please install Python 3.10 or later")
        return False
    
    print("✓ Python version is compatible")
    return True


def check_cuda_availability():
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA is available: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("! CUDA is not available (CPU-only mode)")
        return cuda_available
    except ImportError:
        print("! PyTorch not installed yet")
        return False


def install_dependencies(use_cuda=False):
    """Install dependencies."""
    print("\n" + "="*50)
    print("Installing dependencies...")
    print("="*50)
    
    if use_cuda:
        print("Installing with CUDA 11.8 support...")
        run_command("pip install -r requirements-cuda118.txt")
    else:
        print("Installing CPU-only version...")
        run_command("pip install -r requirements.txt")
    
    print("Installing package in development mode...")
    run_command("pip install -e .")


def run_tests():
    """Run tests to verify installation."""
    print("\n" + "="*50)
    print("Running tests...")
    print("="*50)
    
    # Check if pytest is available
    try:
        import pytest
        run_command("python -m pytest tests/ -v")
    except ImportError:
        print("pytest not available, running basic import tests...")
        
        # Basic import test
        try:
            import nri_md
            print("✓ nri_md package imports successfully")
            
            from nri_md import utils, modules
            print("✓ nri_md.utils imports successfully")
            print("✓ nri_md.modules imports successfully")
            
            import torch
            print(f"✓ PyTorch {torch.__version__} is working")
            
            import numpy as np
            print(f"✓ NumPy {np.__version__} is working")
            
        except ImportError as e:
            print(f"✗ Import error: {e}")
            return False
    
    return True


def test_basic_functionality():
    """Test basic functionality."""
    print("\n" + "="*50)
    print("Testing basic functionality...")
    print("="*50)
    
    try:
        import torch
        from nri_md.utils import my_softmax, gumbel_softmax
        from nri_md.modules import MLP
        
        # Test tensor operations
        x = torch.randn(2, 3, 4)
        result = my_softmax(x, axis=1)
        print("✓ my_softmax function works")
        
        # Test Gumbel softmax
        logits = torch.randn(5, 3)
        result = gumbel_softmax(logits, tau=1.0, hard=False)
        print("✓ gumbel_softmax function works")
        
        # Test MLP module
        mlp = MLP(10, 32, 16)
        x = torch.randn(2, 5, 10)
        output = mlp(x)
        print("✓ MLP module works")
        
        print("✓ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    print("\n" + "="*50)
    print("Creating sample data...")
    print("="*50)
    
    data_dir = Path("data")
    pdb_dir = data_dir / "pdb"
    
    # Create directories
    data_dir.mkdir(exist_ok=True)
    pdb_dir.mkdir(exist_ok=True)
    
    # Create a simple sample PDB file if it doesn't exist
    sample_pdb = pdb_dir / "sample.pdb"
    if not sample_pdb.exists():
        pdb_content = """ATOM      1  CA  ALA A   1      20.154  16.967  10.000  1.00 20.00           C
ATOM      2  CA  VAL A   2      23.000  17.000  12.000  1.00 20.00           C
ATOM      3  CA  GLY A   3      25.000  19.000  14.000  1.00 20.00           C
ENDMDL
ATOM      1  CA  ALA A   1      20.200  17.000  10.100  1.00 20.00           C
ATOM      2  CA  VAL A   2      23.100  17.100  12.100  1.00 20.00           C
ATOM      3  CA  GLY A   3      25.100  19.100  14.100  1.00 20.00           C
ENDMDL
ATOM      1  CA  ALA A   1      20.300  17.100  10.200  1.00 20.00           C
ATOM      2  CA  VAL A   2      23.200  17.200  12.200  1.00 20.00           C
ATOM      3  CA  GLY A   3      25.200  19.200  14.200  1.00 20.00           C
END
"""
        with open(sample_pdb, 'w') as f:
            f.write(pdb_content)
        print(f"✓ Created sample PDB file: {sample_pdb}")
    else:
        print(f"✓ Sample PDB file already exists: {sample_pdb}")


def main():
    """Main installation and testing function."""
    print("NRI-MD Installation and Testing Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check current directory
    if not Path("pyproject.toml").exists():
        print("Error: Please run this script from the NRI-MD root directory")
        return
    
    # Check CUDA availability
    cuda_available = False
    try:
        cuda_available = check_cuda_availability()
    except:
        pass
    
    # Ask user about CUDA installation
    if cuda_available:
        response = input("\nCUDA is available. Install with CUDA 11.8 support? (y/n): ").lower()
        use_cuda = response in ['y', 'yes']
    else:
        use_cuda = False
        print("Installing CPU-only version...")
    
    # Install dependencies
    try:
        install_dependencies(use_cuda)
        print("✓ Dependencies installed successfully")
    except Exception as e:
        print(f"✗ Failed to install dependencies: {e}")
        return
    
    # Run tests
    if not run_tests():
        print("✗ Tests failed")
        return
    
    # Test basic functionality
    if not test_basic_functionality():
        print("✗ Basic functionality tests failed")
        return
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "="*50)
    print("Installation completed successfully!")
    print("="*50)
    print("\nNext steps:")
    print("1. Place your PDB trajectory files in data/pdb/")
    print("2. Convert data: nri-md-convert --data-dir data/pdb --output-dir data")
    print("3. Train model: nri-md-train --num-residues <N> --epochs 500")
    print("4. Visualize results: nri-md-visualize --filename logs/out_probs_train.npy")
    print("5. Analyze paths: nri-md-path --filename logs/out_probs_train.npy --source-node <N>")
    
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch CUDA support: {'Yes' if use_cuda else 'No'}")


if __name__ == "__main__":
    main()