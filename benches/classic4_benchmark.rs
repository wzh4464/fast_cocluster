//! Benchmark for Classic4 dataset using Pipeline API
//!
//! Classic4 is a standard document clustering dataset with 4 categories:
//! - CISI: Information Retrieval (1,460 docs)
//! - CRANFIELD: Aeronautics (1,400 docs)
//! - MEDLINE: Medical (1,033 docs)
//! - CACM: Computer Science (3,204 docs)
//! Total: 7,095 documents
//!
//! Run with: cargo bench --bench classic4_benchmark
//! Prepare data: python3 scripts/download_classic4.py

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fast_cocluster::dimerge_co::*;
use fast_cocluster::pipeline::*;
use fast_cocluster::scoring::{CompatibilityScorer, PearsonScorer};
use fast_cocluster::matrix::Matrix;
use ndarray::Array2;
use std::error::Error;
use std::path::Path;
use std::time::Duration;

/// Load Classic4 dataset from .npy file
fn load_classic4_subset() -> Result<(Array2<f64>, String), Box<dyn Error>> {
    let data_path = Path::new("data/classic4_subset_1000.npy");

    if !data_path.exists() {
        eprintln!("\n{}", "=".repeat(70));
        eprintln!("Classic4 dataset not found!");
        eprintln!("{}", "=".repeat(70));
        eprintln!("Please run: python3 scripts/download_classic4.py");
        eprintln!("This will download and prepare the Classic4 dataset.\n");

        // Create a synthetic dataset for benchmarking
        eprintln!("Creating synthetic dataset for benchmark...");
        let matrix = create_synthetic_classic4(1000, 500);
        return Ok((matrix, "synthetic".to_string()));
    }

    // Load using ndarray-npy
    let array: Array2<f64> = ndarray_npy::read_npy(data_path)
        .map_err(|e| format!("Failed to load Classic4 dataset: {}", e))?;

    Ok((array, "real".to_string()))
}

/// Create a synthetic dataset resembling Classic4 structure
fn create_synthetic_classic4(n_docs: usize, n_features: usize) -> Array2<f64> {
    use ndarray_rand::rand_distr::{Distribution, Exp};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(42);
    let exp_strong = Exp::new(0.3).unwrap();  // For cluster-specific features
    let exp_weak = Exp::new(1.0).unwrap();    // For background noise

    let mut matrix = Array2::zeros((n_docs, n_features));

    // 4 clusters with clear structure
    let docs_per_cluster = n_docs / 4;
    let features_per_cluster = n_features / 4;

    for cluster_id in 0..4 {
        let doc_start = cluster_id * docs_per_cluster;
        let doc_end = if cluster_id < 3 {
            (cluster_id + 1) * docs_per_cluster
        } else {
            n_docs
        };

        let feature_start = cluster_id * features_per_cluster;
        let feature_end = if cluster_id < 3 {
            (cluster_id + 1) * features_per_cluster
        } else {
            n_features
        };

        // Cluster-specific high values
        for doc in doc_start..doc_end {
            // Strong cluster-specific features
            for feat in feature_start..feature_end {
                if exp_strong.sample(&mut rng) < 0.3 {  // 30% density in cluster
                    matrix[[doc, feat]] = exp_strong.sample(&mut rng) * 3.0 + 1.0;
                }
            }

            // Weak background noise across all features
            for feat in 0..n_features {
                if exp_weak.sample(&mut rng) < 0.05 {  // 5% global noise
                    matrix[[doc, feat]] += exp_weak.sample(&mut rng) * 0.5;
                }
            }
        }
    }

    matrix
}

/// Benchmark traditional SVD-based pipeline on Classic4
fn bench_classic4_traditional_pipeline(c: &mut Criterion) {
    let (matrix_data, data_type) = match load_classic4_subset() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading dataset: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group(format!("classic4_{}_traditional", data_type));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("svd_pipeline_k4", |b| {
        b.iter(|| {
            // Create Matrix from Array2 using public new() method
            let matrix = Matrix::new(matrix_data.clone());

            let pipeline = CoclusterPipeline::builder()
                .with_clusterer(Box::new(SVDClusterer::new(4, 0.1)))
                .with_scorer(Box::new(PearsonScorer::new(3, 3)))
                .min_score(0.3)
                .max_submatrices(10)
                .parallel(true)
                .build()
                .unwrap();

            let _ = pipeline.run(black_box(&matrix)).unwrap();
        });
    });

    group.finish();
}

/// Benchmark DiMergeCo pipeline with different configurations
fn bench_classic4_dimerge_co_pipeline(c: &mut Criterion) {
    let (matrix_data, data_type) = match load_classic4_subset() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading dataset: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group(format!("classic4_{}_dimerge_co", data_type));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let configs = vec![
        ("p4_t1", 4, 1),  // 4 partitions, 1 thread
        ("p4_t4", 4, 4),  // 4 partitions, 4 threads
        ("p8_t4", 8, 4),  // 8 partitions, 4 threads
    ];

    for (name, num_partitions, num_threads) in configs {
        group.bench_with_input(
            BenchmarkId::new("config", name),
            &(num_partitions, num_threads),
            |b, &(partitions, threads)| {
                b.iter(|| {
                    let matrix = Matrix::new(matrix_data.clone());
                    let local_clusterer = ClustererAdapter::new(SVDClusterer::new(4, 0.1));

                    let pipeline = CoclusterPipeline::builder()
                        .with_dimerge_co_explicit(
                            4,                      // k clusters
                            matrix_data.nrows(),    // n samples
                            0.05,                   // delta
                            partitions,
                            local_clusterer,
                            HierarchicalMergeConfig::default(),
                            threads,
                        )
                        .unwrap()
                        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
                        .min_score(0.3)
                        .max_submatrices(10)
                        .build()
                        .unwrap();

                    let _ = pipeline.run(black_box(&matrix)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different merge strategies on Classic4
fn bench_classic4_merge_strategies(c: &mut Criterion) {
    let (matrix_data, data_type) = match load_classic4_subset() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading dataset: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group(format!("classic4_{}_merge_strategies", data_type));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let strategies = vec![
        ("union", MergeStrategy::Union),
        ("adaptive", MergeStrategy::Adaptive),
        ("intersection_03", MergeStrategy::Intersection { overlap_threshold: 0.3 }),
    ];

    for (name, strategy) in strategies {
        group.bench_with_input(
            BenchmarkId::new("strategy", name),
            &strategy,
            |b, strat| {
                b.iter(|| {
                    let matrix = Matrix::new(matrix_data.clone());
                    let local_clusterer = ClustererAdapter::new(SVDClusterer::new(4, 0.1));
                    let merge_config = HierarchicalMergeConfig {
                        merge_strategy: strat.clone(),
                        merge_threshold: 0.5,
                        rescore_merged: true,
                        parallel_level: 2,
                    };

                    let pipeline = CoclusterPipeline::builder()
                        .with_dimerge_co_explicit(
                            4,
                            matrix_data.nrows(),
                            0.05,
                            4,
                            local_clusterer,
                            merge_config,
                            4,
                        )
                        .unwrap()
                        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
                        .min_score(0.3)
                        .build()
                        .unwrap();

                    let _ = pipeline.run(black_box(&matrix)).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark comparison: Traditional vs DiMergeCo
fn bench_classic4_comparison(c: &mut Criterion) {
    let (matrix_data, data_type) = match load_classic4_subset() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading dataset: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group(format!("classic4_{}_comparison", data_type));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    // Traditional approach
    group.bench_function("traditional", |b| {
        b.iter(|| {
            let matrix = Matrix::new(matrix_data.clone());
            let pipeline = CoclusterPipeline::builder()
                .with_clusterer(Box::new(SVDClusterer::new(4, 0.1)))
                .with_scorer(Box::new(CompatibilityScorer::new(0.3, 0.3)))
                .min_score(0.4)
                .parallel(true)
                .build()
                .unwrap();

            let _ = pipeline.run(black_box(&matrix)).unwrap();
        });
    });

    // DiMergeCo approach (optimized)
    group.bench_function("dimerge_co_optimized", |b| {
        b.iter(|| {
            let matrix = Matrix::new(matrix_data.clone());
            let local_clusterer = ClustererAdapter::new(SVDClusterer::new(4, 0.1));

            let pipeline = CoclusterPipeline::builder()
                .with_dimerge_co(
                    4,
                    matrix_data.nrows(),
                    0.05,
                    local_clusterer,
                    4,  // 4 threads
                )
                .unwrap()
                .with_scorer(Box::new(CompatibilityScorer::new(0.3, 0.3)))
                .min_score(0.4)
                .build()
                .unwrap();

            let _ = pipeline.run(black_box(&matrix)).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    classic4_benches,
    bench_classic4_traditional_pipeline,
    bench_classic4_dimerge_co_pipeline,
    bench_classic4_merge_strategies,
    bench_classic4_comparison,
);

criterion_main!(classic4_benches);
