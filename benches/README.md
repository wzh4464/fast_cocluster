# DiMergeCo Benchmarks

Comprehensive performance benchmarks for the DiMergeCo divide-merge co-clustering algorithm.

## Running Benchmarks

### Run All Benchmarks
```bash
cargo bench --bench dimerge_co_benchmarks
```

### Run Specific Benchmark Group
```bash
# Probabilistic partitioning only
cargo bench --bench dimerge_co_benchmarks -- probabilistic_partitioning

# Hierarchical merging only
cargo bench --bench dimerge_co_benchmarks -- hierarchical_merging

# Full pipeline
cargo bench --bench dimerge_co_benchmarks -- full_dimerge_co_pipeline

# Parallelism comparison
cargo bench --bench dimerge_co_benchmarks -- parallelism_comparison
```

### View HTML Reports
After running benchmarks, open the generated HTML report:
```bash
open target/criterion/report/index.html
```

## Benchmark Groups

### 1. **Probabilistic Partitioning** (`bench_probabilistic_partitioning`)
- Tests partitioning performance across matrix sizes: 50×40, 100×80, 200×150, 500×400
- Measures SVD computation and partition formation overhead
- Expected: O(k × min(n, m)²) complexity

### 2. **Hierarchical Merging** (`bench_hierarchical_merging`)
- Tests binary tree merging with 2, 4, 8, 16 partitions
- Measures tree construction and merge operation overhead
- Expected: O(log P) communication rounds where P = number of partitions

### 3. **Merge Strategies** (`bench_merge_strategies`)
- Compares performance of 4 strategies:
  - **Union**: Combines all clusters (fastest)
  - **Adaptive**: Smart strategy selection (moderate)
  - **Intersection**: Overlapping clusters only (moderate)
  - **Weighted**: Score-weighted combination (slowest)

### 4. **Full Pipeline** (`bench_full_pipeline`)
- End-to-end DiMergeCo execution on matrices: 60×50, 100×80, 150×120
- Includes all 3 phases: partition → cluster → merge
- Most comprehensive performance test

### 5. **Parallelism Comparison** (`bench_parallelism_comparison`)
- Measures speedup with 1, 2, 4, 8 threads
- Shows scaling efficiency of parallel implementation
- Expected: Near-linear speedup up to 4 threads on typical hardware

### 6. **Partition Extraction** (`bench_partition_extraction`)
- Tests partition data extraction for small (10×10), medium (50×40), large (100×80) partitions
- Measures index mapping overhead

### 7. **Theoretical Validation** (`bench_theoretical_validation`)
- Benchmarks validation operations:
  - Preservation probability calculation (Jaccard similarity)
  - Spectral gap validation
- Typically negligible overhead

## Expected Performance

### Baseline (Single-threaded)
| Operation | 100×80 Matrix | 200×150 Matrix |
|-----------|---------------|----------------|
| Partitioning | ~50ms | ~200ms |
| Local Clustering | ~100ms | ~400ms |
| Merging (4 partitions) | ~20ms | ~50ms |
| **Total** | ~170ms | ~650ms |

### Parallel (4 threads)
| Operation | 100×80 Matrix | 200×150 Matrix | Speedup |
|-----------|---------------|----------------|---------|
| Partitioning | ~50ms | ~200ms | 1.0x (SVD limited) |
| Local Clustering | ~30ms | ~120ms | 3.3x |
| Merging (4 partitions) | ~10ms | ~25ms | 2.0x |
| **Total** | ~90ms | ~345ms | **1.9x** |

*Note: Actual results depend on hardware, BLAS library, and data characteristics.*

## Optimization Tips

### For Large Matrices (>500×400)
- Increase `num_partitions` to reduce local clustering overhead
- Use `parallel_level: 3` for deeper parallel tree merging
- Consider `MergeStrategy::Union` for fastest merging

### For Small Matrices (<100×80)
- Set `parallel_level: 0` to disable parallelization overhead
- Use `num_partitions: 2` to minimize partition count
- `MergeStrategy::Adaptive` works well

### For Real-Time Applications
- Pre-partition data offline
- Cache partitioner results
- Use `rescore_merged: false` to skip rescoring

## Profiling

For detailed profiling with flamegraphs:
```bash
# Install flamegraph
cargo install flamegraph

# Run with profiling
cargo flamegraph --bench dimerge_co_benchmarks -- --bench
```

## Continuous Benchmarking

To track performance regressions:
```bash
# Baseline
cargo bench --bench dimerge_co_benchmarks -- --save-baseline main

# After changes
cargo bench --bench dimerge_co_benchmarks -- --baseline main
```

Criterion will show performance differences compared to baseline.
