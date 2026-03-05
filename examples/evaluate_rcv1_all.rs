/// Full evaluation of all methods on RCV1-all (standalone vs DiMergeCo)
/// RCV1-all = all train (23k) + sampled test (27k) = 50k samples
///
/// Usage: cargo run --release --example evaluate_rcv1_all

use fast_cocluster::atom::{
    fnmf::FnmfClusterer,
    nbvd::NbvdClusterer,
    onm3f::Onm3fClusterer,
    pnmtf::PnmtfClusterer,
    tri_factor_base::TriFactorConfig,
};
use fast_cocluster::dimerge_co::parallel_coclusterer::LocalClusterer;
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::SVDClusterer;
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
    match KMeans::params(k).max_n_iterations(200).fit(&dataset) {
        Ok(model) => model.predict(dataset).targets.to_vec(),
        Err(e) => {
            eprintln!("K-means failed: {e}; falling back to trivial labels");
            vec![0; n_rows]
        }
    }
}

fn make_config(k: usize) -> TriFactorConfig {
    TriFactorConfig {
        n_row_clusters: k,
        n_col_clusters: k,
        max_iter: 20,
        n_init: 3,
        tol: 1e-9,
        seed: None,
    }
}

fn run_comparison<L: LocalClusterer>(
    name: &str,
    standalone: L,
    dimerge_local: L,
    matrix: &Matrix<f64>,
    true_labels: &[usize],
    k: usize,
    num_threads: usize,
    tp: usize,
    m_blocks: usize,
    n_blocks: usize,
) {
    let rows = matrix.data.nrows();

    // Standalone run
    let start = Instant::now();
    let standalone_result = match standalone.cluster_local(&matrix.data) {
        Ok(subs) => {
            let pred = extract_labels(&subs, rows, k);
            Some((calculate_nmi(true_labels, &pred), calculate_ari(true_labels, &pred), start.elapsed().as_secs_f64()))
        }
        Err(_) => None,
    };

    // DiMergeCo run
    let start = Instant::now();
    let dimerge_result = match DiMergeCoClusterer::new(k, rows, 0.05, dimerge_local, HierarchicalMergeConfig::default(), num_threads, tp, m_blocks, n_blocks) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let pred = extract_labels(&result.submatrices, rows, k);
                Some((calculate_nmi(true_labels, &pred), calculate_ari(true_labels, &pred), start.elapsed().as_secs_f64(), result.submatrices.len()))
            }
            Err(_) => None,
        },
        Err(_) => None,
    };

    // Print results
    print!("{:<18}", name);
    if let Some((nmi, ari, t)) = standalone_result {
        print!("  {:>6.4} {:>6.4} {:>9.1}", nmi, ari, t);
    } else {
        print!("  {:>6} {:>6} {:>9}", "FAIL", "-", "-");
    }
    if let Some((nmi, ari, t, c)) = dimerge_result {
        println!("      {:>6.4} {:>6.4} {:>9.1}  {:>7}", nmi, ari, t, c);
    } else {
        println!("      {:>6} {:>6} {:>9}  {:>7}", "FAIL", "-", "-", "-");
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    // Load RCV1-all (train + sampled test)
    println!("Loading RCV1-all...");
    let array: Array2<f64> = ndarray_npy::read_npy("data/rcv1/rcv1_all.npy").expect("load data");
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/rcv1/rcv1_all_labels.npy").expect("load labels");
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| x as usize).collect();
    let (rows, cols) = (array.nrows(), array.ncols());
    println!("RCV1-all: {} x {} ({:.2} GB)", rows, cols, (rows * cols * 8) as f64 / 1e9);

    let matrix = Matrix::new(array);
    let k = 4;
    let num_threads = 16;
    let tp = 20;  // optimized from sweep (was 10)
    let (m_blocks, n_blocks) = (6, 6);  // optimized from sweep (was 8x8)

    println!("\n{}", "=".repeat(80));
    println!("Method              Standalone                    DiMergeCo");
    println!("                    NMI     ARI    Time(s)        NMI     ARI    Time(s)  Clusters");
    println!("{}", "=".repeat(80));

    // Spectral
    run_comparison(
        "spectral",
        ClustererAdapter::new(SVDClusterer::new(k, 0.1)),
        ClustererAdapter::new(SVDClusterer::new(k, 0.1)),
        &matrix, &true_labels, k, num_threads, tp, m_blocks, n_blocks,
    );

    // NBVD
    run_comparison(
        "nbvd",
        NbvdClusterer::with_config(make_config(k)),
        NbvdClusterer::with_config(make_config(k)),
        &matrix, &true_labels, k, num_threads, tp, m_blocks, n_blocks,
    );

    // ONM3F
    run_comparison(
        "onm3f",
        Onm3fClusterer::with_config(make_config(k)),
        Onm3fClusterer::with_config(make_config(k)),
        &matrix, &true_labels, k, num_threads, tp, m_blocks, n_blocks,
    );

    // PNMTF
    run_comparison(
        "pnmtf",
        PnmtfClusterer::with_config(make_config(k), 0.1, 0.1, 0.1),
        PnmtfClusterer::with_config(make_config(k), 0.1, 0.1, 0.1),
        &matrix, &true_labels, k, num_threads, tp, m_blocks, n_blocks,
    );

    // FNMF
    run_comparison(
        "fnmf",
        FnmfClusterer::new(k, k),
        FnmfClusterer::new(k, k),
        &matrix, &true_labels, k, num_threads, tp, m_blocks, n_blocks,
    );

    println!("{}", "=".repeat(80));
    println!("\nDone.");
}
