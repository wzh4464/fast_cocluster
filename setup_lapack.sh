#!/bin/bash
# LAPACK Environment Setup for Fast CoCluster
# macOS with Homebrew OpenBLAS

set -e

echo "🔧 Setting up LAPACK environment for Fast CoCluster..."
echo

# Get OpenBLAS path
OPENBLAS_PATH=$(brew --prefix openblas)
echo "✓ OpenBLAS found at: $OPENBLAS_PATH"

# Export environment variables for building
export OPENBLAS_DIR="$OPENBLAS_PATH"
export OPENBLAS_LIB="$OPENBLAS_PATH/lib"
export OPENBLAS_INCLUDE="$OPENBLAS_PATH/include"

# Runtime library path (for macOS)
export DYLD_LIBRARY_PATH="$OPENBLAS_PATH/lib:${DYLD_LIBRARY_PATH:-}"

# For ndarray-linalg (use openblas backend)
export CARGO_FEATURE_OPENBLAS=1

echo "✓ Environment variables set:"
echo "  OPENBLAS_DIR=$OPENBLAS_DIR"
echo "  OPENBLAS_LIB=$OPENBLAS_LIB"
echo "  DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
echo

# Test compilation
echo "🔨 Testing compilation..."
cargo check --all-targets

echo
echo "✅ LAPACK environment configured successfully!"
echo
echo "To use in your shell, run:"
echo "  source setup_lapack.sh"
echo
echo "To run tests:"
echo "  cargo test"
echo
echo "To run benchmarks:"
echo "  cargo bench --bench dimerge_co_benchmarks"
