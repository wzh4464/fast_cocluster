//! Ablation study: compare merge strategies for DiMergeCo
//!
//! Tests all 4 Rust merge strategies (Union, Intersection, Weighted, Adaptive)
//! on Classic4 and BCW datasets, measuring NMI, ARI, and merge time.
//!
//! Since partition seeds are deterministic (iter_id), all strategies see the
//! same local co-clusters — this isolates the effect of the merge strategy.
//!
//! Run with: RUST_LOG=info cargo run --release --example ablation_merge_strategy

use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::{Clusterer, SVDClusterer};
use fast_cocluster::submatrix::Submatrix;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;

/// Load Classic4 (6460 x 4667, k=4)
fn load_classic4() -> Result<(Matrix<f64>, Vec<usize>, usize), Box<dyn std::error::Error>> {
    let array: Array2<f64> = ndarray_npy::read_npy("data/classic4_paper.npy")
        .map_err(|e| format!("Failed to load classic4_paper.npy: {}", e))?;
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/classic4_paper_labels.npy")
        .map_err(|e| format!("Failed to load classic4_paper_labels.npy: {}", e))?;
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| x as usize).collect();
    let matrix = Matrix::new(array);
    println!("Loaded Classic4: {} x {}, k=4", matrix.rows, matrix.cols);
    Ok((matrix, true_labels, 4))
}

/// Load BCW (569 x 30, k=2)
fn load_bcw() -> Result<(Matrix<f64>, Vec<usize>, usize), Box<dyn std::error::Error>> {
    let array: Array2<f64> = ndarray_npy::read_npy("data/bcw.npy")
        .map_err(|e| format!("Failed to load bcw.npy: {}", e))?;
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/bcw_labels.npy")
        .map_err(|e| format!("Failed to load bcw_labels.npy: {}", e))?;
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| x as usize).collect();
    let matrix = Matrix::new(array);
    println!("Loaded BCW: {} x {}, k=2", matrix.rows, matrix.cols);
    Ok((matrix, true_labels, 2))
}

/// Calculate NMI between true and predicted labels
fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len() as f64;
    if n < 2.0 { return 0.0; }

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
        if p > 0.0 { h_true -= p * p.ln(); }
    }

    let mut h_pred = 0.0;
    for &count in pred_counts.values() {
        let p = count as f64 / n;
        if p > 0.0 { h_pred -= p * p.ln(); }
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

    if h_true + h_pred > 0.0 { 2.0 * mi / (h_true + h_pred) } else { 0.0 }
}

/// Calculate ARI between true and predicted labels
fn calculate_ari(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len();
    if n < 2 { return 0.0; }

    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut true_counts: HashMap<usize, usize> = HashMap::new();
    let mut pred_counts: HashMap<usize, usize> = HashMap::new();

    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency.entry((t, p)).or_insert(0) += 1;
        *true_counts.entry(t).or_insert(0) += 1;
        *pred_counts.entry(p).or_insert(0) += 1;
    }

    let comb2 = |x: usize| -> f64 { if x < 2 { 0.0 } else { (x * (x - 1)) as f64 / 2.0 } };

    let sum_comb_nij: f64 = contingency.values().map(|&v| comb2(v)).sum();
    let sum_comb_ai: f64 = true_counts.values().map(|&v| comb2(v)).sum();
    let sum_comb_bj: f64 = pred_counts.values().map(|&v| comb2(v)).sum();
    let comb_n = comb2(n);

    let expected = sum_comb_ai * sum_comb_bj / comb_n;
    let max_index = (sum_comb_ai + sum_comb_bj) / 2.0;

    if (max_index - expected).abs() < 1e-12 { 0.0 }
    else { (sum_comb_nij - expected) / (max_index - expected) }
}

/// Extract document labels via consensus clustering (membership matrix + k-means)
fn extract_labels<'a>(result: &[Submatrix<'a, f64>], n_docs: usize, k: usize) -> Vec<usize> {
    if result.is_empty() { return vec![0; n_docs]; }

    let n_coclusters = result.len();
    let mut membership = Array2::<f64>::zeros((n_docs, n_coclusters));
    for (cid, submatrix) in result.iter().enumerate() {
        for &row_idx in &submatrix.row_indices {
            if row_idx < n_docs { membership[[row_idx, cid]] = 1.0; }
        }
    }

    let dataset = DatasetBase::from(membership);
    let model = KMeans::params(k)
        .max_n_iterations(200)
        .fit(&dataset)
        .expect("K-means on membership vectors failed");
    let predictions = model.predict(dataset);
    predictions.targets.to_vec()
}

/// All merge strategy configs to test
fn merge_strategies() -> Vec<(String, HierarchicalMergeConfig)> {
    vec![
        ("Adaptive (Ours)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Adaptive,
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Union".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Union,
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Intersection(0.3)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Intersection { overlap_threshold: 0.3 },
            merge_threshold: 0.3,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Intersection(0.5)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Intersection { overlap_threshold: 0.5 },
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Weighted(0.7/0.3)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Weighted { left_weight: 0.7, right_weight: 0.3 },
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Weighted(0.5/0.5)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Weighted { left_weight: 0.5, right_weight: 0.5 },
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
    ]
}

/// Run ablation on one dataset
fn run_ablation(
    name: &str,
    matrix: &Matrix<f64>,
    true_labels: &[usize],
    k: usize,
    m_blocks: usize,
    n_blocks: usize,
    t_p: usize,
    num_threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("ABLATION — {} ({}x{}, T_p={}, threads={})", name, m_blocks, n_blocks, t_p, num_threads);
    println!("{}", "=".repeat(80));

    // Baseline: SVD on full matrix
    {
        let start = Instant::now();
        let clusterer = SVDClusterer::new(k, 0.1);
        let submatrices = clusterer.cluster(matrix)?;
        let runtime = start.elapsed().as_secs_f64();

        let mut labels = vec![0usize; matrix.rows];
        for (cid, sub) in submatrices.iter().enumerate() {
            for &r in &sub.row_indices {
                if r < matrix.rows { labels[r] = cid; }
            }
        }
        let nmi = calculate_nmi(true_labels, &labels);
        let ari = calculate_ari(true_labels, &labels);
        println!("\n  Baseline SCC:  NMI={:.4}  ARI={:.4}  Time={:.3}s", nmi, ari, runtime);
    }

    println!("\n  {:<22} {:>8} {:>8} {:>10} {:>8} {:>8}",
        "Strategy", "NMI", "ARI", "CoClusters", "MergeT", "TotalT");
    println!("  {}", "-".repeat(72));

    for (strategy_name, config) in merge_strategies() {
        let start = Instant::now();
        let local_clusterer = ClustererAdapter::new(SVDClusterer::new(k, 0.1));

        let clusterer = DiMergeCoClusterer::new(
            k, matrix.rows, 0.05, local_clusterer, config,
            num_threads, t_p, m_blocks, n_blocks,
        )?;

        let result = clusterer.run(matrix)?;
        let total_time = start.elapsed().as_secs_f64();
        let merge_time = result.stats.phase_times.merging_ms as f64 / 1000.0;

        let pred_labels = extract_labels(&result.submatrices, matrix.rows, k);
        let nmi = calculate_nmi(true_labels, &pred_labels);
        let ari = calculate_ari(true_labels, &pred_labels);

        println!("  {:<22} {:>8.4} {:>8.4} {:>10} {:>7.3}s {:>7.3}s",
            strategy_name, nmi, ari, result.submatrices.len(), merge_time, total_time);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .init();

    println!("\n{}", "=".repeat(80));
    println!("DiMergeCo Merge Strategy Ablation Study");
    println!("{}", "=".repeat(80));

    let num_threads = num_cpus::get().min(16);

    // Classic4
    let (matrix, labels, k) = load_classic4()?;
    run_ablation("Classic4", &matrix, &labels, k, 2, 2, 10, num_threads)?;
    run_ablation("Classic4", &matrix, &labels, k, 2, 2, 30, num_threads)?;

    // BCW
    let (matrix, labels, k) = load_bcw()?;
    run_ablation("BCW", &matrix, &labels, k, 2, 2, 10, num_threads)?;

    println!("\n{}", "=".repeat(80));
    println!("Ablation complete.");
    println!("{}", "=".repeat(80));

    Ok(())
}
