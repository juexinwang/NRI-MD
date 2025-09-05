# Makefile for NRI-MD project

.PHONY: help install install-cuda install-dev clean test format lint type-check all-checks

help:
	@echo "Available commands:"
	@echo "  install      - Install package with CPU-only dependencies"
	@echo "  install-cuda - Install package with CUDA 11.8 dependencies"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  clean        - Clean build artifacts"
	@echo "  test         - Run tests"
	@echo "  format       - Format code with black"
	@echo "  lint         - Lint code with flake8"
	@echo "  type-check   - Type check with mypy"
	@echo "  all-checks   - Run all code quality checks"

install:
	pip install -r requirements.txt
	pip install -e .

install-cuda:
	pip install -r requirements-cuda118.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	pytest tests/ -v --cov=nri_md --cov-report=term-missing

format:
	black nri_md/ tests/

lint:
	flake8 nri_md/ tests/

type-check:
	mypy nri_md/

all-checks: format lint type-check test
	@echo "All checks completed!"

# Data processing commands
convert-data:
	nri-md-convert --data-dir data/pdb --output-dir data

# Training commands
train:
	nri-md-train --num-residues 77 --epochs 500 --save-folder logs

# Analysis commands
visualize:
	nri-md-visualize --filename logs/out_probs_train.npy --num-residues 77 --output-dir logs

analyze-paths:
	nri-md-path --filename logs/out_probs_train.npy --source-node 46 --num-residues 77