//! Systematic parameter sweep to identify optimal DiMergeCo scenarios
//!
//! Tests DiMergeCo vs Traditional on various scenarios:
//! - Different dataset sizes (100, 300, 500, 1000 docs)
//! - Different sparsity levels (90%, 95%, 99%)
//! - Different cluster structures (well-separated vs overlapping)
//! - Different thread counts (1, 2, 4, 8)
//! - Different partition counts (2, 4, 8, 16)
//!
//! Run with: cargo run --release --example parameter_sweep

use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::*;
use fast_cocluster::scoring::PearsonScorer;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use std::time::Instant;

#[derive(Debug, Clone)]
struct ExperimentConfig {
    name: String,
    n_docs: usize,
    n_features: usize,
    sparsity: f64,
    n_clusters: usize,
    cluster_separation: f64, // Higher = more separated
}

#[derive(Debug)]
struct ExperimentResult {
    config: ExperimentConfig,
    // DiMergeCo results (different configs)
    dimerge_results: Vec<DiMergeCoResult>,
}

#[derive(Debug)]
struct DiMergeCoResult {
    partitions: usize,
    threads: usize,
    time: f64,
    clusters: usize,
    nmi: f64,
}

/// Generate synthetic dataset with controlled properties
fn generate_dataset(config: &ExperimentConfig) -> (Matrix<f64>, Vec<usize>) {
    use ndarray_rand::rand_distr::{Distribution, Exp};
    use ndarray_rand::rand::{SeedableRng, Rng};
    use ndarray_rand::rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(42);
    let mut matrix = Array2::zeros((config.n_docs, config.n_features));
    let mut true_labels = vec![0; config.n_docs];

    let docs_per_cluster = config.n_docs / config.n_clusters;
    let features_per_cluster = config.n_features / config.n_clusters;

    // Create clusters with controlled separation
    for cluster_id in 0..config.n_clusters {
        let doc_start = cluster_id * docs_per_cluster;
        let doc_end = if cluster_id < config.n_clusters - 1 {
            (cluster_id + 1) * docs_per_cluster
        } else {
            config.n_docs
        };

        let feature_start = cluster_id * features_per_cluster;
        let feature_end = if cluster_id < config.n_clusters - 1 {
            (cluster_id + 1) * features_per_cluster
        } else {
            config.n_features
        };

        // Cluster-specific signal strength based on separation parameter
        let signal_strength = config.cluster_separation;
        let noise_level = 1.0 - config.cluster_separation;

        for doc in doc_start..doc_end {
            true_labels[doc] = cluster_id;

            // Calculate target density to achieve desired sparsity
            let target_nnz = ((1.0 - config.sparsity) * config.n_features as f64) as usize;

            // Cluster-specific features (strong signal)
            let n_cluster_features = (target_nnz as f64 * 0.7) as usize;
            let cluster_dist = Exp::new(1.0 / signal_strength).unwrap();

            for _ in 0..n_cluster_features {
                let feat = feature_start + (rng.gen::<usize>() % (feature_end - feature_start));
                matrix[[doc, feat]] += cluster_dist.sample(&mut rng);
            }

            // Random features (noise)
            let n_noise_features = target_nnz - n_cluster_features;
            let noise_dist = Exp::new(1.0 / noise_level).unwrap();

            for _ in 0..n_noise_features {
                let feat = rng.gen::<usize>() % config.n_features;
                matrix[[doc, feat]] += noise_dist.sample(&mut rng) * 0.3;
            }
        }
    }

    (Matrix::new(matrix), true_labels)
}

/// Calculate NMI between true and predicted labels
fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    use std::collections::HashMap;

    let n = true_labels.len() as f64;

    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut true_counts: HashMap<usize, usize> = HashMap::new();
    let mut pred_counts: HashMap<usize, usize> = HashMap::new();

    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency.entry((t, p)).or_insert(0) += 1;
        *true_counts.entry(t).or_insert(0) += 1;
        *pred_counts.entry(p).or_insert(0) += 1;
    }

    let mut h_true = 0.0;
    for &count in true_counts.values() {
        let p = count as f64 / n;
        if p > 0.0 {
            h_true -= p * p.ln();
        }
    }

    let mut h_pred = 0.0;
    for &count in pred_counts.values() {
        let p = count as f64 / n;
        if p > 0.0 {
            h_pred -= p * p.ln();
        }
    }

    let mut mi = 0.0;
    for (&(t, p), &count) in contingency.iter() {
        let n_ij = count as f64;
        let n_i = *true_counts.get(&t).unwrap() as f64;
        let n_j = *pred_counts.get(&p).unwrap() as f64;

        if n_ij > 0.0 {
            mi += (n_ij / n) * ((n * n_ij) / (n_i * n_j)).ln();
        }
    }

    if h_true + h_pred > 0.0 {
        2.0 * mi / (h_true + h_pred)
    } else {
        0.0
    }
}

fn extract_labels<'a>(result: &[Submatrix<'a, f64>], n_docs: usize) -> Vec<usize> {
    let mut labels = vec![0; n_docs];
    for (cluster_id, submatrix) in result.iter().enumerate() {
        for &row_idx in &submatrix.row_indices {
            labels[row_idx] = cluster_id;
        }
    }
    labels
}

fn run_experiment(config: ExperimentConfig) -> ExperimentResult {
    println!("\n{}", "=".repeat(70));
    println!("Experiment: {}", config.name);
    println!("  Dataset: {} docs × {} features", config.n_docs, config.n_features);
    println!("  Sparsity: {:.1}%", config.sparsity * 100.0);
    println!("  Clusters: {}", config.n_clusters);
    println!("  Separation: {:.2}", config.cluster_separation);
    println!("{}", "=".repeat(70));

    // Generate dataset
    let (matrix, true_labels) = generate_dataset(&config);

    // Run DiMergeCo with different configurations
    let mut dimerge_results = Vec::new();

    let partition_configs = vec![2, 4, 8];
    let thread_configs = vec![1, 2, 4];

    for &partitions in &partition_configs {
        for &threads in &thread_configs {
            print!("  DiMergeCo (p{}_t{})... ", partitions, threads);

            let start = Instant::now();
            let local_clusterer = ClustererAdapter::new(
                SVDClusterer::new(config.n_clusters, 0.1)
            );

            let pipeline_dimerge = CoclusterPipeline::builder()
                .with_dimerge_co_explicit(
                    config.n_clusters,
                    matrix.rows,
                    0.05,
                    partitions,
                    local_clusterer,
                    HierarchicalMergeConfig::default(),
                    threads,
                )
                .unwrap()
                .with_scorer(Box::new(PearsonScorer::new(3, 3)))
                .min_score(0.3)
                .max_submatrices(20)
                .build()
                .unwrap();

            let result_dimerge = pipeline_dimerge.run(&matrix).unwrap();
            let dimerge_time = start.elapsed().as_secs_f64();

            let pred_labels_dimerge = extract_labels(&result_dimerge.submatrices, matrix.rows);
            let dimerge_nmi = calculate_nmi(&true_labels, &pred_labels_dimerge);

            println!("Done in {:.3}s (NMI: {:.4}, Clusters: {})",
                     dimerge_time, dimerge_nmi, result_dimerge.submatrices.len());

            dimerge_results.push(DiMergeCoResult {
                partitions,
                threads,
                time: dimerge_time,
                clusters: result_dimerge.submatrices.len(),
                nmi: dimerge_nmi,
            });
        }
    }

    ExperimentResult {
        config,
        dimerge_results,
    }
}

fn print_summary(results: &[ExperimentResult]) {
    println!("\n\n{}", "=".repeat(70));
    println!("SUMMARY: DiMergeCo Performance Analysis");
    println!("{}", "=".repeat(70));

    println!("\n📊 Fastest Configurations by Scenario:");
    println!("{:<30} {:>10} {:>8} {:>10}", "Scenario", "Runtime", "NMI", "Config");
    println!("{}", "-".repeat(70));

    // Find fastest config for each scenario
    for exp in results {
        if let Some(fastest) = exp.dimerge_results.iter()
            .min_by(|a, b| a.time.partial_cmp(&b.time).unwrap())
        {
            println!("{:<30} {:>9.3}s {:>8.4} p{}_t{}",
                     exp.config.name,
                     fastest.time,
                     fastest.nmi,
                     fastest.partitions,
                     fastest.threads);
        }
    }

    println!("\n📈 Best Quality Configurations:");
    println!("{:<30} {:>10} {:>8} {:>10}", "Scenario", "Runtime", "NMI", "Config");
    println!("{}", "-".repeat(70));

    // Find best quality config for each scenario
    for exp in results {
        if let Some(best_quality) = exp.dimerge_results.iter()
            .max_by(|a, b| a.nmi.partial_cmp(&b.nmi).unwrap())
        {
            println!("{:<30} {:>9.3}s {:>8.4} p{}_t{}",
                     exp.config.name,
                     best_quality.time,
                     best_quality.nmi,
                     best_quality.partitions,
                     best_quality.threads);
        }
    }

    println!("\n💡 Key Insights:");

    // Analyze performance trends by configuration
    let avg_time_by_config: std::collections::HashMap<(usize, usize), f64> = {
        let mut map = std::collections::HashMap::new();
        let mut counts = std::collections::HashMap::new();

        for exp in results {
            for dm in &exp.dimerge_results {
                let key = (dm.partitions, dm.threads);
                *map.entry(key).or_insert(0.0) += dm.time;
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        map.iter().map(|(k, v)| (*k, v / counts[k] as f64)).collect()
    };

    println!("\n  1. Average Runtime by Configuration:");
    let mut configs: Vec<_> = avg_time_by_config.iter().collect();
    configs.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
    for ((partitions, threads), avg_time) in configs.iter().take(5) {
        println!("     p{}_t{}: {:.3}s", partitions, threads, avg_time);
    }

    // Best overall configuration (balance of speed and quality)
    println!("\n  2. Recommended Configuration:");
    let mut all_results: Vec<_> = results.iter()
        .flat_map(|exp| exp.dimerge_results.iter().map(move |dm| (exp, dm)))
        .collect();

    // Sort by: high NMI (>0.5), then fast runtime
    all_results.sort_by(|a, b| {
        let score_a = if a.1.nmi > 0.5 { -a.1.time } else { -1000.0 };
        let score_b = if b.1.nmi > 0.5 { -b.1.time } else { -1000.0 };
        score_b.partial_cmp(&score_a).unwrap()
    });

    if let Some((_, best)) = all_results.first() {
        println!("     Partitions: {}", best.partitions);
        println!("     Threads: {}", best.threads);
        println!("     Typical runtime: {:.3}s", best.time);
        println!("     Typical NMI: {:.4}", best.nmi);
    }

    // Quality analysis
    println!("\n  3. Quality Distribution:");
    let high_quality = results.iter()
        .flat_map(|exp| exp.dimerge_results.iter())
        .filter(|dm| dm.nmi > 0.7)
        .count();
    let medium_quality = results.iter()
        .flat_map(|exp| exp.dimerge_results.iter())
        .filter(|dm| dm.nmi > 0.5 && dm.nmi <= 0.7)
        .count();
    let total_configs = results.iter()
        .map(|exp| exp.dimerge_results.len())
        .sum::<usize>();

    println!("     High quality (NMI > 0.7): {}/{}", high_quality, total_configs);
    println!("     Medium quality (NMI 0.5-0.7): {}/{}", medium_quality, total_configs);

    println!("\n{}", "=".repeat(70));
}

fn main() {
    println!("DiMergeCo Parameter Sweep - Finding Optimal Scenarios");
    println!("This will test various dataset sizes, sparsity levels, and configurations");
    println!();

    let experiments = vec![
        // Small, well-separated clusters (ideal for DiMergeCo)
        ExperimentConfig {
            name: "Small_WellSeparated".to_string(),
            n_docs: 300,
            n_features: 500,
            sparsity: 0.95,
            n_clusters: 4,
            cluster_separation: 0.8,
        },

        // Medium, moderate separation
        ExperimentConfig {
            name: "Medium_Moderate".to_string(),
            n_docs: 500,
            n_features: 1000,
            sparsity: 0.95,
            n_clusters: 4,
            cluster_separation: 0.6,
        },

        // Large, well-separated
        ExperimentConfig {
            name: "Large_WellSeparated".to_string(),
            n_docs: 1000,
            n_features: 1000,
            sparsity: 0.95,
            n_clusters: 4,
            cluster_separation: 0.8,
        },

        // High sparsity
        ExperimentConfig {
            name: "HighSparsity".to_string(),
            n_docs: 500,
            n_features: 1000,
            sparsity: 0.99,
            n_clusters: 4,
            cluster_separation: 0.7,
        },

        // Many small clusters
        ExperimentConfig {
            name: "ManyClusters".to_string(),
            n_docs: 500,
            n_features: 1000,
            sparsity: 0.95,
            n_clusters: 8,
            cluster_separation: 0.7,
        },

        // Overlapping clusters (challenging)
        ExperimentConfig {
            name: "Overlapping".to_string(),
            n_docs: 500,
            n_features: 1000,
            sparsity: 0.95,
            n_clusters: 4,
            cluster_separation: 0.3,
        },
    ];

    let mut results = Vec::new();

    for config in experiments {
        let result = run_experiment(config);
        results.push(result);
    }

    print_summary(&results);
}
