# Fast Co-clustering Library - Architecture & Design Patterns

## Core Design Principles

### 1. Trait-Based Polymorphism

The library uses Rust traits to provide pluggable algorithms and scoring methods:

**Clusterer Trait**
```rust
pub trait Clusterer: Send + Sync {
    fn cluster<'matrix_life>(
        &self,
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>>;
    fn name(&self) -> &str;
}
```

**Implementations:**
- `SVDClusterer`: SVD-based clustering (recommended)
- `BasicCoclusterer`: Simple partitioning
- `SpectralCoclustererHook`: Spectral clustering

**Scorer Trait**
```rust
pub trait Scorer: Send + Sync {
    fn score(&self, matrix: &Matrix<f64>, submatrix: &Submatrix<f64>) -> f64;
    fn score_all(&self, matrix: &Matrix<f64>, submatrices: &[Submatrix<f64>]) -> Vec<f64>;
}
```

**Implementations:**
- `PearsonScorer`: Linear correlation
- `ExponentialScorer`: Exponential decay
- `CompatibilityScorer`: Variance-based
- `CompositeScorer`: Weighted combination

### 2. Builder Pattern

The `PipelineBuilder` provides a fluent API for configuration:

```rust
let pipeline = CoclusterPipeline::builder()
    .with_clusterer(Box::new(SVDClusterer::new(5, 0.1)))
    .with_scorer(Box::new(PearsonScorer::new(3, 3)))
    .min_score(0.6)
    .max_submatrices(10)
    .min_submatrix_size(5, 5)
    .parallel(true)
    .build()?;
```

**Advantages:**
- Sensible defaults
- Optional configuration
- Type-safe construction
- Clear API

### 3. Zero-Copy Views

The `Submatrix` structure provides views into matrices without copying:

```rust
pub struct Submatrix<'a, T> {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    // References parent matrix
}
```

**Benefits:**
- Memory efficient
- Fast submatrix operations
- No unnecessary allocations

### 4. Pipeline Architecture

The processing flow follows a pipeline pattern:

```
Input Matrix → Clustering → Scoring → Filtering → Output Results
```

**Components:**
1. **Clusterer**: Generates candidate co-clusters
2. **Scorer**: Evaluates quality of co-clusters
3. **Filter**: Removes low-quality results
4. **Sort**: Orders by score

### 5. Parallel Processing

Using Rayon for data parallelism:

```rust
scores.par_iter()
    .map(|submatrix| scorer.score(matrix, submatrix))
    .collect()
```

**Requirements:**
- Types must implement `Send + Sync`
- Thread-safe shared state uses `Arc<Mutex<T>>`

## Key Architectural Decisions

### 1. Separation of Concerns

**Clustering**: Separate from scoring
- Clusterers focus on finding structure
- Scorers evaluate quality
- Pipeline orchestrates the process

**Benefits:**
- Easy to add new algorithms
- Mix and match components
- Clear responsibilities

### 2. Configuration vs. Code

**PipelineConfig** separates configuration from logic:
- min_score: Quality threshold
- max_submatrices: Result limit
- parallel: Enable parallelism
- collect_stats: Performance metrics

### 3. Error Handling Strategy

**Result Types Throughout:**
```rust
Result<Vec<Submatrix>, Box<dyn Error>>
```

**Custom Error Types:**
```rust
pub enum PipelineError {
    // Specific error variants
}
```

**Error Propagation:**
- Use `?` operator
- Return informative errors
- No panics in library code

### 4. Performance Optimizations

**Memory:**
- Avoid cloning large matrices
- Use references and borrowing
- Views instead of copies

**Computation:**
- Parallel scoring
- Efficient linear algebra (nalgebra, ndarray-linalg)
- Early filtering

**Caching:**
- Reuse intermediate results where possible

### 5. Type Safety

**Strong Types:**
- `Matrix<T>`: Wrapper around Array2
- `Submatrix<'a, T>`: Borrowed view
- Generic parameters for flexibility

**Lifetime Annotations:**
```rust
fn cluster<'matrix_life>(
    &self,
    matrix: &'matrix_life Matrix<f64>
) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn Error>>
```

## Module Responsibilities

### pipeline.rs
- **Responsibility**: Orchestration
- **Exports**: `CoclusterPipeline`, `PipelineBuilder`, traits
- **Dependencies**: cocluster, scoring, matrix, submatrix

### scoring.rs
- **Responsibility**: Quality evaluation
- **Exports**: `Scorer` trait, scorer implementations
- **Dependencies**: matrix, submatrix

### cocluster.rs
- **Responsibility**: Core clustering algorithm
- **Exports**: `Coclusterer` implementation
- **Dependencies**: matrix, nalgebra, kmeans

### matrix.rs
- **Responsibility**: Matrix abstraction
- **Exports**: `Matrix<T>` wrapper
- **Dependencies**: ndarray

### submatrix.rs
- **Responsibility**: Submatrix views
- **Exports**: `Submatrix<'a, T>`
- **Dependencies**: ndarray

## Extension Points

### Adding New Clusterers

1. Implement `Clusterer` trait
2. Add configuration parameters as struct fields
3. Implement `cluster` method with algorithm
4. Add to builder options

### Adding New Scorers

1. Implement `Scorer` trait
2. Define parameters in struct
3. Implement `score` and `score_all` methods
4. Optionally parallelize `score_all`

### Adding New Pipeline Steps

1. Define trait for new step
2. Add to pipeline execution flow
3. Update builder with configuration
4. Update result types if needed

## Data Flow

```
User Code
    ↓
PipelineBuilder (configuration)
    ↓
CoclusterPipeline::run()
    ↓
Clusterer::cluster() → Vec<Submatrix>
    ↓
Scorer::score_all() → Vec<f64>
    ↓
Filter & Sort
    ↓
StepResult {submatrices, scores, stats}
    ↓
User Code
```

## Concurrency Model

### Thread Safety
- All traits require `Send + Sync`
- Scoring can be parallelized
- Shared state uses `Arc<Mutex<T>>`

### Rayon Configuration
```rust
// User can configure thread count
std::env::set_var("RAYON_NUM_THREADS", "8");
```

### Parallel vs Sequential
- Controlled by `PipelineConfig::parallel`
- Sequential for debugging
- Parallel for production

## Testing Strategy

### Unit Tests
- Test individual scorers
- Test clusterers
- Test utility functions

### Integration Tests
- Test full pipeline
- Test with different configurations
- Test error conditions

### Performance Tests
- Benchmarks for algorithms
- Memory profiling
- Scalability tests

## Future Extensibility

**Designed for:**
- New clustering algorithms
- New scoring methods
- New preprocessing steps
- New postprocessing steps
- Different matrix types (sparse, distributed)
- GPU acceleration

**Extension Pattern:**
1. Define trait
2. Implement for new type
3. Add to builder
4. Document usage