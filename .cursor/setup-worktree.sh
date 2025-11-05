#!/bin/bash
# Setup script for new git worktrees
# This installs pre-commit hooks when working in a new worktree

set -e

echo "Setting up worktree with pre-commit hooks..."

# Check if .pre-commit-config.yaml exists
if [ ! -f ".pre-commit-config.yaml" ]; then
    echo "Warning: .pre-commit-config.yaml not found. Skipping pre-commit setup."
    exit 0
fi

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    # Try python3 first, fallback to python
    if command -v python3 &> /dev/null; then
        python3 -m pip install --user pre-commit
    elif command -v python &> /dev/null; then
        python -m pip install --user pre-commit
    else
        echo "Error: Python not found. Cannot install pre-commit."
        exit 1
    fi
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo "Pre-commit hooks installed successfully!"
