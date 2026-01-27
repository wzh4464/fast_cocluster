# Fast Co-clustering Library - Codebase Structure

## Directory Structure

```
fast_cocluster/
├── src/               # Source code
│   ├── lib.rs         # Library entry point
│   ├── main.rs        # Binary entry point
│   ├── cocluster.rs   # Core co-clustering implementation
│   ├── matrix.rs      # Matrix wrapper
│   ├── pipeline.rs    # Pipeline orchestration
│   ├── scoring.rs     # Scoring methods
│   ├── spectral_cocluster.rs  # Spectral clustering
│   ├── parallel_cocluster.rs  # Parallel processing
│   ├── submatrix.rs   # Submatrix data structure
│   ├── union_dc.rs    # Union divide-and-conquer
│   ├── config.rs      # Configuration
│   └── util.rs        # Utilities
├── docs/              # Documentation
│   ├── coclusterer.puml     # PlantUML diagram
│   └── union_dc.puml        # PlantUML diagram
├── .github/workflows/ # CI/CD
│   └── rust.yml       # GitHub Actions workflow
├── target/            # Build output (gitignored)
├── Cargo.toml         # Package manifest
├── Cargo.lock         # Dependency lock file
└── README.md          # Project documentation

```

## Core Modules

### pipeline.rs
- **Traits**: `Clusterer`, `Pipeline`
- **Main Types**: `CoclusterPipeline`, `PipelineBuilder`, `PipelineConfig`
- **Clusterers**: `SVDClusterer`, `BasicCoclusterer`, `SpectralCoclustererHook`
- **Result Types**: `StepResult`, `PipelineStats`, `ScoreDistribution`

### scoring.rs
- **Trait**: `Scorer`
- **Implementations**: `PearsonScorer`, `ExponentialScorer`, `CompatibilityScorer`, `CompositeScorer`

### cocluster.rs
- Core co-clustering algorithm implementation
- SVD-based clustering

### matrix.rs
- Matrix wrapper around ndarray::Array2

### submatrix.rs
- Submatrix data structure with row/column indices
- View into larger matrix

### spectral_cocluster.rs
- Spectral co-clustering implementation

### union_dc.rs
- Union divide-and-conquer utilities

## Design Patterns

### 1. Trait-Based Abstraction
- `Clusterer` trait for different clustering algorithms
- `Scorer` trait for different scoring methods
- `Pipeline` trait for processing workflows

### 2. Builder Pattern
- `PipelineBuilder` for configuring pipelines
- Fluent API with method chaining

### 3. Lifetime Management
- Extensive use of lifetimes for zero-copy views
- `Submatrix<'matrix_life, T>` borrows from matrix

### 4. Error Handling
- Result types throughout
- Custom error types (e.g., `PipelineError`)

## Entry Points

### Library Usage
```rust
use fast_cocluster::pipeline::*;
use fast_cocluster::scoring::*;
use fast_cocluster::Matrix;
```

### Binary Entry Point
- `src/main.rs`: Command-line interface (if needed)