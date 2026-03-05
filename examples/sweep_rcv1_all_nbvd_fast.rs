/// Focused parameter sweep for NBVD on RCV1-all with DiMergeCo
/// Tests key configurations based on theory
///
/// Usage: cargo run --release --example sweep_rcv1_all_nbvd_fast

use fast_cocluster::atom::nbvd::NbvdClusterer;
use fast_cocluster::atom::tri_factor_base::TriFactorConfig;
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::submatrix::Submatrix;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use std::time::Instant;

fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len() as f64;
    if n < 2.0 { return 0.0; }
    let k_true = *true_labels.iter().max().unwrap_or(&0) + 1;
    let k_pred = *pred_labels.iter().max().unwrap_or(&0) + 1;
    let mut contingency = vec![vec![0usize; k_pred]; k_true];
    for i in 0..true_labels.len() {
        if pred_labels[i] < k_pred {
            contingency[true_labels[i]][pred_labels[i]] += 1;
        }
    }
    let row_sums: Vec<f64> = contingency.iter().map(|r| r.iter().sum::<usize>() as f64).collect();
    let col_sums: Vec<f64> = (0..k_pred).map(|j| contingency.iter().map(|r| r[j]).sum::<usize>() as f64).collect();
    let mut mi = 0.0;
    for i in 0..k_true {
        for j in 0..k_pred {
            let nij = contingency[i][j] as f64;
            if nij > 0.0 && row_sums[i] > 0.0 && col_sums[j] > 0.0 {
                mi += (nij / n) * ((nij * n) / (row_sums[i] * col_sums[j])).ln();
            }
        }
    }
    let h_true: f64 = row_sums.iter().filter(|&&s| s > 0.0).map(|&s| -(s / n) * (s / n).ln()).sum();
    let h_pred: f64 = col_sums.iter().filter(|&&s| s > 0.0).map(|&s| -(s / n) * (s / n).ln()).sum();
    if h_true + h_pred == 0.0 { 0.0 } else { 2.0 * mi / (h_true + h_pred) }
}

fn calculate_ari(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len();
    if n < 2 { return 0.0; }
    let k_true = *true_labels.iter().max().unwrap_or(&0) + 1;
    let k_pred = *pred_labels.iter().max().unwrap_or(&0) + 1;
    let mut contingency = vec![vec![0i64; k_pred]; k_true];
    for i in 0..n {
        if pred_labels[i] < k_pred {
            contingency[true_labels[i]][pred_labels[i]] += 1;
        }
    }
    let comb2 = |x: i64| -> i64 { if x < 2 { 0 } else { x * (x - 1) / 2 } };
    let sum_comb_nij: i64 = contingency.iter().flat_map(|r| r.iter()).map(|&x| comb2(x)).sum();
    let sum_comb_ai: i64 = contingency.iter().map(|r| comb2(r.iter().sum::<i64>())).sum();
    let sum_comb_bj: i64 = (0..k_pred).map(|j| comb2(contingency.iter().map(|r| r[j]).sum::<i64>())).sum();
    let comb_n = comb2(n as i64) as f64;
    if comb_n < 1.0 { return 0.0; }
    let expected = (sum_comb_ai as f64) * (sum_comb_bj as f64) / comb_n;
    let max_idx = 0.5 * (sum_comb_ai as f64 + sum_comb_bj as f64);
    let denom = max_idx - expected;
    if denom.abs() < 1e-12 { 0.0 } else { (sum_comb_nij as f64 - expected) / denom }
}

fn extract_labels(submatrices: &[Submatrix<'_, f64>], n_rows: usize, k: usize) -> Vec<usize> {
    if submatrices.is_empty() { return vec![0; n_rows]; }
    let n_coclusters = submatrices.len();
    let mut membership = ndarray::Array2::<f64>::zeros((n_rows, n_coclusters));
    for (cid, sm) in submatrices.iter().enumerate() {
        for &row_idx in &sm.row_indices {
            if row_idx < n_rows { membership[[row_idx, cid]] = 1.0; }
        }
    }
    let dataset = DatasetBase::from(membership);
    let model = KMeans::params(k).max_n_iterations(300).fit(&dataset).expect("K-means failed");
    model.predict(dataset).targets.to_vec()
}

fn make_config(k: usize) -> TriFactorConfig {
    TriFactorConfig {
        n_row_clusters: k,
        n_col_clusters: k,
        max_iter: 20,
        n_init: 1,
        tol: 1e-9,
        seed: None,
    }
}

fn run_dimerge(
    matrix: &Matrix<f64>,
    true_labels: &[usize],
    k: usize,
    rows: usize,
    tp: usize,
    m_blocks: usize,
    n_blocks: usize,
    delta: f64,
) -> Option<(f64, f64, f64, usize)> {
    let start = Instant::now();
    let local = NbvdClusterer::with_config(make_config(k));
    match DiMergeCoClusterer::new(k, rows, delta, local, HierarchicalMergeConfig::default(), 16, tp, m_blocks, n_blocks) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let pred = extract_labels(&result.submatrices, rows, k);
                let nmi = calculate_nmi(true_labels, &pred);
                let ari = calculate_ari(true_labels, &pred);
                let time = start.elapsed().as_secs_f64();
                Some((nmi, ari, time, result.submatrices.len()))
            }
            Err(e) => {
                eprintln!("  DiMergeCo run failed: {}", e);
                None
            }
        },
        Err(e) => {
            eprintln!("  DiMergeCo init failed: {}", e);
            None
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    // Load RCV1-all
    println!("Loading RCV1-all...");
    let array: Array2<f64> = ndarray_npy::read_npy("data/rcv1/rcv1_all.npy").expect("load data");
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/rcv1/rcv1_all_labels.npy").expect("load labels");
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| x as usize).collect();
    let (rows, cols) = (array.nrows(), array.ncols());
    println!("RCV1-all: {} x {} (aspect ratio: {:.2})", rows, cols, rows as f64 / cols as f64);

    let matrix = Matrix::new(array);
    let k = 4;
    let delta = 0.05;

    // Theoretical preservation probability
    println!("\n=== Theoretical T_p Analysis ===");
    println!("T_p  P(preserve)");
    for tp in [5, 10, 15, 20] {
        let p = 1.0 - (-0.5 * tp as f64).exp();
        println!("{:3}  {:.4}", tp, p);
    }

    println!("\n=== Focused Parameter Sweep ===");
    println!("{}", "=".repeat(90));
    println!("{:<6} {:<8} {:<8} {:>8} {:>8} {:>10} {:>10}",
             "T_p", "m_blks", "n_blks", "NMI", "ARI", "Time(s)", "Clusters");
    println!("{}", "=".repeat(90));

    // Key configurations to test:
    // 1. Baseline: T_p=10, 8x8 (from previous run)
    // 2. Fewer blocks (larger per block): 4x4, 6x6
    // 3. Different T_p with best block size

    let configs = [
        // (T_p, m_blocks, n_blocks)
        // Fewer blocks = larger blocks = more samples per block
        (10, 4, 4),   // Larger blocks
        (10, 6, 6),   // Medium blocks
        (10, 8, 8),   // Baseline
        // Different T_p with 6x6 (likely optimal)
        (5, 6, 6),
        (15, 6, 6),
        (20, 6, 6),
        // Try 4x4 with different T_p
        (5, 4, 4),
        (15, 4, 4),
    ];

    let mut results = Vec::new();

    for &(tp, m_blks, n_blks) in &configs {
        print!("{:<6} {:<8} {:<8} ", tp, m_blks, n_blks);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        if let Some((nmi, ari, time, clusters)) = run_dimerge(&matrix, &true_labels, k, rows, tp, m_blks, n_blks, delta) {
            println!("{:>8.4} {:>8.4} {:>10.1} {:>10}", nmi, ari, time, clusters);
            results.push((tp, m_blks, n_blks, nmi, ari, time, clusters));
        } else {
            println!("{:>8} {:>8} {:>10} {:>10}", "FAIL", "-", "-", "-");
        }
    }

    println!("{}", "=".repeat(90));

    // Analysis
    if !results.is_empty() {
        // Find best
        let best = results.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
        println!("\n=== Best Configuration ===");
        println!("T_p: {}, Grid: {}x{}", best.0, best.1, best.2);
        println!("NMI: {:.4}, ARI: {:.4}, Time: {:.1}s, Clusters: {}", best.3, best.4, best.5, best.6);

        // Compare with baseline
        if let Some(baseline) = results.iter().find(|r| r.0 == 10 && r.1 == 8 && r.2 == 8) {
            let improvement = (best.3 - baseline.3) / baseline.3 * 100.0;
            println!("\nImprovement over baseline (T_p=10, 8x8): {:+.1}%", improvement);
        }

        // Analysis by T_p (fixed 6x6)
        println!("\n=== T_p Impact (6x6 grid) ===");
        for &tp in &[5, 10, 15, 20] {
            if let Some(r) = results.iter().find(|r| r.0 == tp && r.1 == 6 && r.2 == 6) {
                let p_theory = 1.0 - (-0.5 * tp as f64).exp();
                println!("T_p={:2}: NMI={:.4}, P(theory)={:.4}", tp, r.3, p_theory);
            }
        }

        // Analysis by block size (fixed T_p=10)
        println!("\n=== Block Size Impact (T_p=10) ===");
        for &(m, n) in &[(4, 4), (6, 6), (8, 8)] {
            if let Some(r) = results.iter().find(|r| r.0 == 10 && r.1 == m && r.2 == n) {
                let samples_per_block = (rows / m) * (cols / n);
                println!("{}x{}: NMI={:.4}, ~{} samples/block", m, n, r.3, samples_per_block);
            }
        }
    }

    println!("\nDone.");
}
