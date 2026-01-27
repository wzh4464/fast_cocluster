# Fast Co-clustering Library - Project Overview

## Purpose
High-performance Rust library for bi-clustering (co-clustering) large matrices using SVD-based algorithms and flexible scoring methods.

**Key Applications:**
- Gene expression analysis
- Recommendation systems
- Market basket analysis
- Document clustering
- Social network analysis

## Tech Stack

### Core Language
- **Rust**: Nightly version (1.89.0-nightly as of Jan 2025)
- **Edition**: 2021

### Main Dependencies
- **ndarray** (0.15.6): Multi-dimensional array library
- **rayon** (1.10.0): Data parallelism
- **nalgebra** (0.33): Linear algebra
- **linfa** (0.7): Machine learning framework
- **kmeans_smid** (0.3.0): K-means clustering
- **ndarray-linalg** (0.16.0): Linear algebra for ndarray
- **statrs** (0.18): Statistics

### Additional Dependencies
- **rand** (0.9): Random number generation
- **log** (0.4.21): Logging facade
- **serde** (1.0): Serialization/deserialization
- **chrono** (0.4.38): Time/date handling

### Dev Dependencies
- **env_logger** (0.11): Logger implementation for testing

## Key Features
1. **High Performance**: Parallel processing with Rayon
2. **Flexible Algorithms**: SVD-based, spectral, and basic clustering
3. **Multiple Scoring Methods**: Pearson, exponential, compatibility, composite
4. **Builder Pattern**: Easy-to-use configuration
5. **Memory Efficient**: Optimized for large matrices
6. **Type Safe**: Full Rust type safety

## System Requirements
- Rust 1.70+ (nightly recommended)
- BLAS/LAPACK libraries
- macOS (Darwin), Linux (Ubuntu/Debian), or Windows