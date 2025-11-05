#!/bin/bash
# Setup script for rust-advanced-heaps
# This ensures pre-commit hooks are installed

set -e

echo "Setting up rust-advanced-heaps development environment..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit || python3 -m pip install pre-commit
fi

# Install git hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo "✓ Setup complete! Pre-commit hooks are now active."
