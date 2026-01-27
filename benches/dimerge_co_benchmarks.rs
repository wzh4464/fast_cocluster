//! Benchmarks for DiMergeCo algorithm components and full pipeline
//!
//! Run with: cargo bench --bench dimerge_co_benchmarks
//! HTML reports: target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, AxisScale};
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;
use std::time::Duration;

/// Create a synthetic matrix with planted co-cluster structure
fn create_benchmark_matrix(n_rows: usize, n_cols: usize, n_clusters: usize) -> Array2<f64> {
    let mut matrix = Array2::random((n_rows, n_cols), Uniform::new(0.0, 1.0));

    let rows_per_cluster = n_rows / n_clusters;
    let cols_per_cluster = n_cols / n_clusters;

    for k in 0..n_clusters {
        let row_start = k * rows_per_cluster;
        let row_end = ((k + 1) * rows_per_cluster).min(n_rows);
        let col_start = k * cols_per_cluster;
        let col_end = ((k + 1) * cols_per_cluster).min(n_cols);

        for i in row_start..row_end {
            for j in col_start..col_end {
                matrix[[i, j]] += 2.0;
            }
        }
    }

    matrix
}

/// Simple local clusterer for benchmarking
struct BenchmarkLocalClusterer {
    k: usize,
}

impl LocalClusterer for BenchmarkLocalClusterer {
    fn cluster_local<'a>(
        &self,
        matrix: &'a Array2<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
        let n_rows = matrix.nrows();
        let n_cols = matrix.ncols();

        if n_rows < 2 || n_cols < 2 {
            return Ok(vec![]);
        }

        // Create simple clusters by splitting
        let mut clusters = Vec::new();
        let rows_per_cluster = (n_rows / self.k).max(1);

        for i in 0..self.k.min(n_rows / rows_per_cluster) {
            let row_start = i * rows_per_cluster;
            let row_end = ((i + 1) * rows_per_cluster).min(n_rows);
            let rows: Vec<usize> = (row_start..row_end).collect();
            let cols: Vec<usize> = (0..n_cols).collect();

            if let Some(sub) = Submatrix::from_indices(matrix, &rows, &cols) {
                clusters.push(sub);
            }
        }

        Ok(clusters)
    }
}

/// Benchmark probabilistic partitioning with different matrix sizes
fn bench_probabilistic_partitioning(c: &mut Criterion) {
    let mut group = c.benchmark_group("probabilistic_partitioning");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.sample_size(20); // Reduce sample size for large matrices

    let sizes = vec![
        (50, 40),
        (100, 80),
        (200, 150),
        (500, 400),
    ];

    for (n_rows, n_cols) in sizes {
        let matrix = create_benchmark_matrix(n_rows, n_cols, 3);
        let partitioner = ProbabilisticPartitioner::new(3, n_rows, 0.05, 4).unwrap();

        group.bench_with_input(
            BenchmarkId::new("partition", format!("{}x{}", n_rows, n_cols)),
            &matrix,
            |b, mat| {
                b.iter(|| {
                    partitioner.partition(black_box(mat)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark hierarchical merging with different number of partitions
fn bench_hierarchical_merging(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_merging");
    group.sample_size(50);

    let matrix_data = create_benchmark_matrix(100, 80, 3);
    let matrix = Matrix {
        data: matrix_data.clone(),
        rows: 100,
        cols: 80,
    };

    let num_partitions_list = vec![2, 4, 8, 16];

    for num_partitions in num_partitions_list {
        // Create mock partition results
        let mut partition_results = Vec::new();
        let rows_per_partition = 100 / num_partitions;

        for i in 0..num_partitions {
            let row_start = i * rows_per_partition;
            let row_end = ((i + 1) * rows_per_partition).min(100);
            let rows: Vec<usize> = (row_start..row_end).collect();
            let cols: Vec<usize> = (0..80).collect();

            if let Some(sub) = Submatrix::from_indices(&matrix.data, &rows, &cols) {
                partition_results.push(vec![sub]);
            }
        }

        let config = HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Union,
            merge_threshold: 0.5,
            rescore_merged: false,
            parallel_level: 2,
        };

        group.bench_with_input(
            BenchmarkId::new("merge", num_partitions),
            &partition_results,
            |b, results| {
                b.iter(|| {
                    let merger = HierarchicalMerger::new(config.clone());
                    merger.execute_parallel(black_box(results.clone()), &matrix).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different merge strategies
fn bench_merge_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_strategies");
    group.sample_size(50);

    let matrix_data = create_benchmark_matrix(80, 60, 3);
    let matrix = Matrix {
        data: matrix_data.clone(),
        rows: 80,
        cols: 60,
    };

    // Create partition results
    let sub1 = Submatrix::from_indices(&matrix.data, &vec![0, 1, 2, 3], &vec![0, 1, 2]).unwrap();
    let sub2 = Submatrix::from_indices(&matrix.data, &vec![4, 5, 6, 7], &vec![3, 4, 5]).unwrap();
    let partition_results = vec![vec![sub1], vec![sub2]];

    let strategies = vec![
        ("Union", MergeStrategy::Union),
        ("Adaptive", MergeStrategy::Adaptive),
        ("Intersection", MergeStrategy::Intersection { overlap_threshold: 0.5 }),
        ("Weighted", MergeStrategy::Weighted { left_weight: 0.6, right_weight: 0.4 }),
    ];

    for (name, strategy) in strategies {
        let config = HierarchicalMergeConfig {
            merge_strategy: strategy,
            merge_threshold: 0.5,
            rescore_merged: false,
            parallel_level: 1,
        };

        group.bench_with_input(
            BenchmarkId::new("strategy", name),
            &config,
            |b, cfg| {
                b.iter(|| {
                    let merger = HierarchicalMerger::new(cfg.clone());
                    merger.execute_parallel(black_box(partition_results.clone()), &matrix).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full DiMergeCo pipeline with different configurations
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_dimerge_co_pipeline");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.sample_size(10); // Very expensive operation
    group.measurement_time(Duration::from_secs(30));

    let matrix_sizes = vec![
        (60, 50),
        (100, 80),
        (150, 120),
    ];

    for (n_rows, n_cols) in matrix_sizes {
        let matrix_data = create_benchmark_matrix(n_rows, n_cols, 3);
        let matrix = Matrix {
            data: matrix_data,
            rows: n_rows,
            cols: n_cols,
        };

        let local_clusterer = BenchmarkLocalClusterer { k: 2 };
        let merge_config = HierarchicalMergeConfig::default();

        let clusterer = DiMergeCoClusterer::with_adaptive(
            3,
            n_rows,
            0.05,
            local_clusterer,
            merge_config,
            2,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("pipeline", format!("{}x{}", n_rows, n_cols)),
            &matrix,
            |b, mat| {
                b.iter(|| {
                    clusterer.run(black_box(mat)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel vs sequential execution
fn bench_parallelism_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallelism_comparison");
    group.sample_size(20);

    let matrix_data = create_benchmark_matrix(100, 80, 3);
    let matrix = Matrix {
        data: matrix_data,
        rows: 100,
        cols: 80,
    };

    let thread_counts = vec![1, 2, 4, 8];

    for num_threads in thread_counts {
        let local_clusterer = BenchmarkLocalClusterer { k: 2 };
        let merge_config = HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Union,
            merge_threshold: 0.5,
            rescore_merged: false,
            parallel_level: if num_threads > 1 { 2 } else { 0 },
        };

        let clusterer = DiMergeCoClusterer::with_adaptive(
            3,
            100,
            0.05,
            local_clusterer,
            merge_config,
            num_threads,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &matrix,
            |b, mat| {
                b.iter(|| {
                    clusterer.run(black_box(mat)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark partition extraction performance
fn bench_partition_extraction(c: &mut Criterion) {
    use fast_cocluster::dimerge_co::pipeline_integration::extract_partition_matrix;

    let mut group = c.benchmark_group("partition_extraction");
    group.sample_size(100);

    let matrix = create_benchmark_matrix(200, 150, 3);

    let partition_sizes = vec![
        (10, 10, "small"),
        (50, 40, "medium"),
        (100, 80, "large"),
    ];

    for (n_rows, n_cols, label) in partition_sizes {
        let partition = Partition {
            row_indices: (0..n_rows).collect(),
            col_indices: (0..n_cols).collect(),
            id: 0,
        };

        group.bench_with_input(
            BenchmarkId::new("extract", label),
            &partition,
            |b, part| {
                b.iter(|| {
                    extract_partition_matrix(black_box(&matrix), black_box(part))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark theoretical validation operations
fn bench_theoretical_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("theoretical_validation");
    group.sample_size(100);

    let matrix_data = create_benchmark_matrix(100, 80, 3);

    // Create clusters for validation
    let ground_truth: Vec<Submatrix<f64>> = (0..5)
        .map(|i| {
            let row_start = i * 10;
            let rows: Vec<usize> = (row_start..row_start + 10).collect();
            let cols: Vec<usize> = (0..20).collect();
            Submatrix::from_indices(&matrix_data, &rows, &cols).unwrap()
        })
        .collect();

    let recovered = ground_truth.clone();

    group.bench_function("preservation_validation", |b| {
        b.iter(|| {
            TheoreticalValidator::validate_preservation(
                black_box(&ground_truth),
                black_box(&recovered),
                black_box(0.05),
            )
        });
    });

    // Benchmark spectral gap validation
    let singular_values = vec![10.0, 8.0, 5.0, 2.0, 1.0, 0.5, 0.2];

    group.bench_function("spectral_gap_validation", |b| {
        b.iter(|| {
            TheoreticalValidator::validate_spectral_gap(
                black_box(&singular_values),
                black_box(3),
                black_box(1.0),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_probabilistic_partitioning,
    bench_hierarchical_merging,
    bench_merge_strategies,
    bench_full_pipeline,
    bench_parallelism_comparison,
    bench_partition_extraction,
    bench_theoretical_validation,
);

criterion_main!(benches);
