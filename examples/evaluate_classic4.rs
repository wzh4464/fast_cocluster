//! Evaluate DiMergeCo clustering quality and performance on Classic4 dataset
//!
//! This program measures efficiency (runtime) and accuracy (clustering quality)
//! of the DiMergeCo algorithm with different configurations.
//!
//! Run with: RUST_LOG=info cargo run --release --example evaluate_classic4

use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::{Clusterer, SVDClusterer};
use fast_cocluster::submatrix::Submatrix;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use log::info;

/// Load Classic4 benchmark dataset with ground truth labels
fn load_classic4() -> Result<(Matrix<f64>, Vec<usize>), Box<dyn std::error::Error>> {
    println!("Loading Classic4 benchmark dataset...");

    let data_path = "data/classic4_benchmark_small.npy";
    let labels_path = "data/classic4_benchmark_small_labels.npy";

    let array: Array2<f64> = ndarray_npy::read_npy(data_path)
        .map_err(|e| {
            format!(
                "Failed to load dataset: {}.\n\nPlease run:\n  \
                python3 scripts/download_classic4.py\n  \
                uv run scripts/create_small_benchmark_data.py",
                e
            )
        })?;

    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy(labels_path)
        .map_err(|e| format!("Failed to load labels: {}", e))?;
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| x as usize).collect();

    let matrix = Matrix::new(array);

    println!("✓ Loaded {} documents × {} features, {} labels",
        matrix.rows, matrix.cols, true_labels.len());
    println!("  Classes: 0=CACM, 1=CISI, 2=CRAN, 3=MED (125 each)");
    Ok((matrix, true_labels))
}

/// Calculate NMI between true and predicted labels
fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
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

/// Extract document labels using consensus clustering on membership vectors.
///
/// DiMergeCo produces multiple co-clusters from overlapping partitions.
/// This function:
/// 1. Builds a binary membership vector for each document (which co-clusters contain it)
/// 2. Runs K-means on these membership vectors to produce k final document clusters
fn extract_labels<'a>(result: &[Submatrix<'a, f64>], n_docs: usize, k: usize) -> Vec<usize> {
    if result.is_empty() {
        return vec![0; n_docs];
    }

    let n_coclusters = result.len();

    // Build membership matrix: n_docs × n_coclusters (binary)
    let mut membership = Array2::<f64>::zeros((n_docs, n_coclusters));
    for (cid, submatrix) in result.iter().enumerate() {
        for &row_idx in &submatrix.row_indices {
            if row_idx < n_docs {
                membership[[row_idx, cid]] = 1.0;
            }
        }
    }

    // K-means on membership vectors to get k document clusters
    let dataset = DatasetBase::from(membership);
    let model = KMeans::params(k)
        .max_n_iterations(200)
        .fit(&dataset)
        .expect("K-means on membership vectors failed");

    let predictions = model.predict(dataset);
    predictions.targets.to_vec()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 启用日志: RUST_LOG=info cargo run --release --example evaluate_classic4
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    println!("\n{}", "=".repeat(70));
    println!("DiMergeCo Clustering Evaluation on Classic4");
    println!("{}", "=".repeat(70));
    println!();

    // Load dataset
    let (matrix, true_labels) = load_classic4()?;

    // Baseline: SVD+KMeans on full matrix (no partitioning)
    {
        println!("\n{}", "-".repeat(70));
        println!("Baseline: SVD+KMeans on Full Matrix (no DiMergeCo)");
        println!("{}", "-".repeat(70));
        let start = Instant::now();
        let clusterer = SVDClusterer::new(4, 0.1);
        let submatrices = clusterer.cluster(&matrix)?;
        let runtime = start.elapsed().as_secs_f64();

        let mut baseline_labels = vec![0usize; matrix.rows];
        for (cid, sub) in submatrices.iter().enumerate() {
            for &r in &sub.row_indices {
                if r < matrix.rows {
                    baseline_labels[r] = cid;
                }
            }
        }
        let baseline_nmi = calculate_nmi(&true_labels, &baseline_labels);

        // Show cluster sizes
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &l in &baseline_labels { *counts.entry(l).or_insert(0) += 1; }
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by_key(|&(k, _)| *k);
        println!("  Runtime:   {:.3}s", runtime);
        println!("  Clusters:  {}", submatrices.len());
        println!("  NMI:       {:.4}", baseline_nmi);
        println!("  Label dist: {:?}", sorted);
    }

    println!("\n{}", "-".repeat(70));
    println!("Testing DiMergeCo with Different Configurations");
    println!("{}", "-".repeat(70));

    // configs: (name, num_partitions, threads, T_p iterations)
    let configs = vec![
        ("p4_t4_i1",  4, 4, 1),    // Baseline: T_p=1
        ("p4_t4_i5",  4, 4, 5),    // T_p=5
        ("p4_t4_i10", 4, 4, 10),   // T_p=10 (recommended)
        ("p4_t4_i20", 4, 4, 20),   // T_p=20 (paper default)
    ];

    let mut results = Vec::new();

    for (name, partitions, threads, iterations) in configs {
        println!("\nConfiguration: {} ({} partitions, {} threads, T_p={})",
            name, partitions, threads, iterations);
        println!("{}", "-".repeat(40));

        let start = Instant::now();
        info!("[{}] Building DiMergeCo clusterer...", name);
        let local_clusterer = ClustererAdapter::new(SVDClusterer::new(4, 0.1));

        let clusterer = DiMergeCoClusterer::new(
            4,                  // k clusters
            matrix.rows,        // n samples
            0.05,               // delta
            partitions,
            local_clusterer,
            HierarchicalMergeConfig::default(),
            threads,
            iterations,         // T_p random partitioning iterations
        )?;

        info!("[{}] DiMergeCo 构建完成，开始运行...", name);
        let result = clusterer.run(&matrix)?;
        let runtime = start.elapsed().as_secs_f64();
        info!("[{}] 完成，耗时 {:.3}s", name, runtime);

        // Diagnostic: show co-cluster sizes and coverage
        let mut covered_docs: HashSet<usize> = HashSet::new();
        for (cid, sub) in result.submatrices.iter().enumerate() {
            let doc_rows: Vec<usize> = sub.row_indices.iter()
                .copied().filter(|&r| r < matrix.rows).collect();
            covered_docs.extend(&doc_rows);
            info!("[{}]   co-cluster {}: {} doc-rows, {} cols",
                name, cid, doc_rows.len(), sub.col_indices.len());
        }
        info!("[{}]   coverage: {}/{} docs in at least one co-cluster",
            name, covered_docs.len(), matrix.rows);

        let pred_labels = extract_labels(&result.submatrices, matrix.rows, 4);

        // Diagnostic: label distribution
        let mut label_counts: HashMap<usize, usize> = HashMap::new();
        for &l in &pred_labels {
            *label_counts.entry(l).or_insert(0) += 1;
        }
        let mut sorted_counts: Vec<_> = label_counts.iter().collect();
        sorted_counts.sort_by_key(|&(k, _)| *k);
        info!("[{}]   label distribution: {:?}", name, sorted_counts);

        let nmi = calculate_nmi(&true_labels, &pred_labels);

        println!("  Runtime:  {:.3}s", runtime);
        println!("  Clusters: {}", result.submatrices.len());
        println!("  Coverage: {}/{} docs", covered_docs.len(), matrix.rows);
        println!("  NMI:      {:.4}", nmi);

        results.push((name, partitions, threads, iterations, runtime, nmi, result.submatrices.len()));
    }

    println!("\n{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "=".repeat(70));

    println!("\n{:<14} {:>6} {:>7} {:>4} {:>10} {:>10}",
        "Config", "Parts", "Threads", "T_p", "Runtime", "NMI");
    println!("{}", "-".repeat(70));
    for (name, partitions, threads, iters, runtime, nmi, _) in &results {
        println!("{:<14} {:>6} {:>7} {:>4} {:>9.3}s {:>10.4}",
            name, partitions, threads, iters, runtime, nmi);
    }

    // Find best configuration
    if let Some((best_name, _, _, _, best_time, best_nmi, _)) = results.iter()
        .max_by(|a, b| {
            // Prioritize: NMI > 0.5, then fastest
            let score_a = if a.5 > 0.5 { 1.0 / a.4 } else { 0.0 };
            let score_b = if b.5 > 0.5 { 1.0 / b.4 } else { 0.0 };
            score_a.partial_cmp(&score_b).unwrap()
        })
    {
        println!("\n Recommended Configuration: {}", best_name);
        println!("   Runtime: {:.3}s", best_time);
        println!("   NMI:     {:.4}", best_nmi);
    }

    println!("\n{}", "=".repeat(70));
    println!("✓ Evaluation complete");
    println!("{}", "=".repeat(70));
    println!();

    Ok(())
}
