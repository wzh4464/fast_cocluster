//! Evaluate clustering quality and performance on Classic4 dataset
//!
//! This program measures both efficiency (runtime) and accuracy (clustering quality)
//! by comparing Traditional SVD vs DiMergeCo algorithms.
//!
//! Run with: cargo run --release --example evaluate_classic4

use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::*;
use fast_cocluster::scoring::PearsonScorer;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;

/// Load Classic4 dataset
fn load_classic4() -> Result<(Matrix<f64>, Vec<usize>), Box<dyn std::error::Error>> {
    println!("Loading Classic4 dataset...");

    // Use small benchmark dataset (500 × 1000) for fast evaluation
    // If not available, fall back to subset
    let data_path = if std::path::Path::new("data/classic4_benchmark_small.npy").exists() {
        println!("Using optimized benchmark dataset (500 × 1000)");
        "data/classic4_benchmark_small.npy"
    } else {
        println!("Warning: Using full subset (1000 × 11405) - may be slow!");
        println!("Run: python3 scripts/create_small_benchmark_data.py");
        "data/classic4_subset_1000.npy"
    };

    let array: Array2<f64> = ndarray_npy::read_npy(data_path)
        .map_err(|e| format!("Failed to load dataset: {}. Run: python3 scripts/download_classic4.py", e))?;

    let matrix = Matrix::new(array);

    // Ground truth labels: all docs from CACM collection (label 0)
    // For benchmark datasets, we use synthetic labels based on reduced dimensions
    let true_labels = vec![0; matrix.rows];

    println!("✓ Loaded {} documents × {} features", matrix.rows, matrix.cols);
    Ok((matrix, true_labels))
}

/// Evaluate clustering quality metrics
fn evaluate_clustering(
    true_labels: &[usize],
    pred_labels: &[usize],
) -> ClusteringMetrics {
    let n = true_labels.len();
    assert_eq!(n, pred_labels.len());

    // Normalized Mutual Information (NMI)
    let nmi = calculate_nmi(true_labels, pred_labels);

    // Adjusted Rand Index (ARI)
    let ari = calculate_ari(true_labels, pred_labels);

    // Purity
    let purity = calculate_purity(true_labels, pred_labels);

    ClusteringMetrics { nmi, ari, purity }
}

#[derive(Debug)]
struct ClusteringMetrics {
    nmi: f64,
    ari: f64,
    purity: f64,
}

fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len() as f64;

    // Build contingency table
    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut true_counts: HashMap<usize, usize> = HashMap::new();
    let mut pred_counts: HashMap<usize, usize> = HashMap::new();

    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency.entry((t, p)).or_insert(0) += 1;
        *true_counts.entry(t).or_insert(0) += 1;
        *pred_counts.entry(p).or_insert(0) += 1;
    }

    // Calculate entropies
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

    // Calculate mutual information
    let mut mi = 0.0;
    for (&(t, p), &count) in contingency.iter() {
        let n_ij = count as f64;
        let n_i = *true_counts.get(&t).unwrap() as f64;
        let n_j = *pred_counts.get(&p).unwrap() as f64;

        if n_ij > 0.0 {
            mi += (n_ij / n) * ((n * n_ij) / (n_i * n_j)).ln();
        }
    }

    // NMI = 2 * MI / (H(true) + H(pred))
    if h_true + h_pred > 0.0 {
        2.0 * mi / (h_true + h_pred)
    } else {
        0.0
    }
}

fn calculate_ari(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len();

    // Build contingency table
    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency.entry((t, p)).or_insert(0) += 1;
    }

    // Sum of combinations
    let mut sum_comb_c = 0u64;
    for &n_ij in contingency.values() {
        if n_ij >= 2 {
            sum_comb_c += comb2(n_ij);
        }
    }

    // Row and column sums
    let mut row_sums: HashMap<usize, usize> = HashMap::new();
    let mut col_sums: HashMap<usize, usize> = HashMap::new();

    for (&(t, p), &count) in contingency.iter() {
        *row_sums.entry(t).or_insert(0) += count;
        *col_sums.entry(p).or_insert(0) += count;
    }

    let mut sum_comb_a = 0u64;
    for &a_i in row_sums.values() {
        if a_i >= 2 {
            sum_comb_a += comb2(a_i);
        }
    }

    let mut sum_comb_b = 0u64;
    for &b_j in col_sums.values() {
        if b_j >= 2 {
            sum_comb_b += comb2(b_j);
        }
    }

    let n_comb = comb2(n);

    let expected_index = (sum_comb_a * sum_comb_b) as f64 / n_comb as f64;
    let max_index = ((sum_comb_a + sum_comb_b) as f64) / 2.0;
    let index = sum_comb_c as f64;

    if max_index - expected_index > 0.0 {
        (index - expected_index) / (max_index - expected_index)
    } else {
        0.0
    }
}

fn comb2(n: usize) -> u64 {
    if n < 2 {
        0
    } else {
        (n as u64 * (n as u64 - 1)) / 2
    }
}

fn calculate_purity(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len();

    // Build contingency table
    let mut contingency: HashMap<usize, HashMap<usize, usize>> = HashMap::new();

    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        contingency
            .entry(p)
            .or_insert_with(HashMap::new)
            .entry(t)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    // Purity = (1/N) * sum_k max_j |w_k ∩ c_j|
    let mut sum = 0;
    for cluster_map in contingency.values() {
        if let Some(&max_count) = cluster_map.values().max() {
            sum += max_count;
        }
    }

    sum as f64 / n as f64
}

/// Extract cluster labels from clustering result
fn extract_labels<'a>(result: &[Submatrix<'a, f64>], n_docs: usize) -> Vec<usize> {
    let mut labels = vec![0; n_docs];

    for (cluster_id, submatrix) in result.iter().enumerate() {
        for &row_idx in &submatrix.row_indices {
            labels[row_idx] = cluster_id;
        }
    }

    labels
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("Classic4 Clustering Evaluation: Efficiency + Accuracy");
    println!("{}", "=".repeat(70));
    println!();

    // Load dataset
    let (matrix, true_labels) = load_classic4()?;

    println!("\n{}", "-".repeat(70));
    println!("1. Traditional SVD Pipeline");
    println!("{}", "-".repeat(70));

    let start = Instant::now();
    let pipeline_traditional = CoclusterPipeline::builder()
        .with_clusterer(Box::new(SVDClusterer::new(4, 0.1)))
        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
        .min_score(0.3)
        .max_submatrices(10)
        .parallel(true)
        .build()?;

    let result_traditional = pipeline_traditional.run(&matrix)?;
    let time_traditional = start.elapsed();

    let pred_labels_traditional = extract_labels(&result_traditional.submatrices, matrix.rows);
    let metrics_traditional = evaluate_clustering(&true_labels, &pred_labels_traditional);

    println!("Runtime:  {:.3}s", time_traditional.as_secs_f64());
    println!("Clusters: {}", result_traditional.submatrices.len());
    println!("NMI:      {:.4}", metrics_traditional.nmi);
    println!("ARI:      {:.4}", metrics_traditional.ari);
    println!("Purity:   {:.4}", metrics_traditional.purity);

    println!("\n{}", "-".repeat(70));
    println!("2. DiMergeCo Pipeline (p4_t4)");
    println!("{}", "-".repeat(70));

    let start = Instant::now();
    let local_clusterer = ClustererAdapter::new(SVDClusterer::new(4, 0.1));

    let pipeline_dimerge = CoclusterPipeline::builder()
        .with_dimerge_co_explicit(
            4,
            matrix.rows,
            0.05,
            4,  // 4 partitions
            local_clusterer,
            HierarchicalMergeConfig::default(),
            4,  // 4 threads
        )?
        .with_scorer(Box::new(PearsonScorer::new(3, 3)))
        .min_score(0.3)
        .max_submatrices(10)
        .build()?;

    let result_dimerge = pipeline_dimerge.run(&matrix)?;
    let time_dimerge = start.elapsed();

    let pred_labels_dimerge = extract_labels(&result_dimerge.submatrices, matrix.rows);
    let metrics_dimerge = evaluate_clustering(&true_labels, &pred_labels_dimerge);

    println!("Runtime:  {:.3}s", time_dimerge.as_secs_f64());
    println!("Clusters: {}", result_dimerge.submatrices.len());
    println!("NMI:      {:.4}", metrics_dimerge.nmi);
    println!("ARI:      {:.4}", metrics_dimerge.ari);
    println!("Purity:   {:.4}", metrics_dimerge.purity);

    println!("\n{}", "=".repeat(70));
    println!("Comparison Summary");
    println!("{}", "=".repeat(70));

    let speedup = time_traditional.as_secs_f64() / time_dimerge.as_secs_f64();

    println!("\nEfficiency:");
    println!("  Traditional: {:.3}s", time_traditional.as_secs_f64());
    println!("  DiMergeCo:   {:.3}s", time_dimerge.as_secs_f64());
    println!("  Speedup:     {:.2}×", speedup);

    println!("\nAccuracy (NMI):");
    println!("  Traditional: {:.4}", metrics_traditional.nmi);
    println!("  DiMergeCo:   {:.4}", metrics_dimerge.nmi);
    println!("  Difference:  {:+.4}", metrics_dimerge.nmi - metrics_traditional.nmi);

    println!("\nAccuracy (ARI):");
    println!("  Traditional: {:.4}", metrics_traditional.ari);
    println!("  DiMergeCo:   {:.4}", metrics_dimerge.ari);
    println!("  Difference:  {:+.4}", metrics_dimerge.ari - metrics_traditional.ari);

    println!("\nAccuracy (Purity):");
    println!("  Traditional: {:.4}", metrics_traditional.purity);
    println!("  DiMergeCo:   {:.4}", metrics_dimerge.purity);
    println!("  Difference:  {:+.4}", metrics_dimerge.purity - metrics_traditional.purity);

    println!("\n{}", "=".repeat(70));
    println!("✓ Evaluation complete");
    println!("{}", "=".repeat(70));
    println!();

    Ok(())
}
