//! Evaluate DiMergeCo clustering quality and performance on Classic4 dataset
//!
//! This program measures efficiency (runtime) and accuracy (clustering quality)
//! of the DiMergeCo algorithm with different configurations.
//!
//! Run with: RUST_LOG=info cargo run --release --example evaluate_classic4

use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::*;
use fast_cocluster::scoring::PearsonScorer;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use std::collections::HashMap;
use std::time::Instant;
use log::info;

/// Load Classic4 benchmark dataset (optimized size)
fn load_classic4() -> Result<(Matrix<f64>, Vec<usize>), Box<dyn std::error::Error>> {
    println!("Loading Classic4 benchmark dataset...");

    // Use small benchmark dataset (500 × 1000) - fast and efficient
    let data_path = "data/classic4_benchmark_small.npy";
    let array: Array2<f64> = ndarray_npy::read_npy(data_path)
        .map_err(|e| {
            format!(
                "Failed to load dataset: {}.\n\nPlease run:\n  \
                python3 scripts/download_classic4.py\n  \
                python3 scripts/create_small_benchmark_data.py",
                e
            )
        })?;

    let matrix = Matrix::new(array);

    // Generate ground truth labels (4 clusters, evenly distributed)
    let n_docs = matrix.rows;
    let n_clusters = 4;
    let docs_per_cluster = n_docs / n_clusters;

    let mut true_labels = Vec::new();
    for i in 0..n_docs {
        let label = i / docs_per_cluster;
        let label = label.min(n_clusters - 1); // Last cluster gets remainder
        true_labels.push(label);
    }

    println!("✓ Loaded {} documents × {} features", matrix.rows, matrix.cols);
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

    println!("\n{}", "-".repeat(70));
    println!("Testing DiMergeCo with Different Configurations");
    println!("{}", "-".repeat(70));

    let configs = vec![
        ("p2_t1", 2, 1),        // 基线：单线程
        ("p4_t4", 4, 4),        // 推荐
        ("p4_t8", 4, 8),        // 更多线程
        ("p8_t8", 8, 8),        // 更多分区+线程
    ];

    let mut results = Vec::new();

    for (name, partitions, threads) in configs {
        println!("\nConfiguration: {} ({} partitions, {} threads)", name, partitions, threads);
        println!("{}", "-".repeat(40));

        let start = Instant::now();
        info!("[{}] 开始构建 pipeline...", name);
        let local_clusterer = ClustererAdapter::new(SVDClusterer::new(4, 0.1));

        let pipeline = CoclusterPipeline::builder()
            .with_dimerge_co_explicit(
                4,                  // k clusters
                matrix.rows,        // n samples
                0.05,               // delta
                partitions,
                local_clusterer,
                HierarchicalMergeConfig::default(),
                threads,
            )?
            .with_scorer(Box::new(PearsonScorer::new(3, 3)))
            .min_score(0.3)
            .max_submatrices(10)
            .build()?;

        info!("[{}] Pipeline 构建完成，开始运行...", name);
        let result = pipeline.run(&matrix)?;
        let runtime = start.elapsed().as_secs_f64();
        info!("[{}] 完成，耗时 {:.3}s", name, runtime);

        let pred_labels = extract_labels(&result.submatrices, matrix.rows);
        let nmi = calculate_nmi(&true_labels, &pred_labels);

        println!("  Runtime:  {:.3}s", runtime);
        println!("  Clusters: {}", result.submatrices.len());
        println!("  NMI:      {:.4}", nmi);

        results.push((name, partitions, threads, runtime, nmi, result.submatrices.len()));
    }

    println!("\n{}", "=".repeat(70));
    println!("Summary");
    println!("{}", "=".repeat(70));

    println!("\n{:<10} {:>10} {:>10} {:>10} {:>10}", "Config", "Partitions", "Threads", "Runtime", "NMI");
    println!("{}", "-".repeat(70));
    for (name, partitions, threads, runtime, nmi, _) in &results {
        println!("{:<10} {:>10} {:>10} {:>9.3}s {:>10.4}", name, partitions, threads, runtime, nmi);
    }

    // Find best configuration
    if let Some((best_name, _, _, best_time, best_nmi, _)) = results.iter()
        .max_by(|a, b| {
            // Prioritize: NMI > 0.5, then fastest
            let score_a = if a.4 > 0.5 { 1.0 / a.3 } else { 0.0 };
            let score_b = if b.4 > 0.5 { 1.0 / b.3 } else { 0.0 };
            score_a.partial_cmp(&score_b).unwrap()
        })
    {
        println!("\n💡 Recommended Configuration: {}", best_name);
        println!("   Runtime: {:.3}s", best_time);
        println!("   NMI:     {:.4}", best_nmi);
    }

    println!("\n{}", "=".repeat(70));
    println!("✓ Evaluation complete");
    println!("{}", "=".repeat(70));
    println!();

    Ok(())
}
