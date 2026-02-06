/// Comprehensive DiMergeCo evaluation on RCV1 with parameter sweep
///
/// Run: cargo run --release --example evaluate_rcv1_sweep

use fast_cocluster::atom::{
    fnmf::FnmfClusterer,
    nbvd::NbvdClusterer,
    onm3f::Onm3fClusterer,
    onmtf::OnmtfClusterer,
    pnmtf::PnmtfClusterer,
    tri_factor_base::TriFactorConfig,
};
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::SVDClusterer;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use std::time::Instant;

fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    assert_eq!(true_labels.len(), pred_labels.len());
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
    assert_eq!(true_labels.len(), pred_labels.len());
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
    let expected = (sum_comb_ai as f64) * (sum_comb_bj as f64) / comb_n;
    let max_idx = 0.5 * (sum_comb_ai as f64 + sum_comb_bj as f64);
    let denom = max_idx - expected;
    if denom.abs() < 1e-12 { 0.0 } else { (sum_comb_nij as f64 - expected) / denom }
}

fn extract_labels(submatrices: &[Submatrix<'_, f64>], n_rows: usize, k: usize) -> Vec<usize> {
    use linfa::prelude::*;
    use linfa_clustering::KMeans;
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

struct EvalResult {
    method: String,
    nmi: f64,
    ari: f64,
    time_secs: f64,
    n_clusters: usize,
    params: String,
}

fn print_result(r: &EvalResult) {
    println!("{:<20} {:>7.4} {:>7.4} {:>8.1}s {:>6} {}",
        r.method, r.nmi, r.ari, r.time_secs, r.n_clusters, r.params);
}

fn run_dimerge_spectral(
    matrix: &Matrix<f64>,
    array: &Array2<f64>,
    true_labels: &[usize],
    k: usize,
    m_blocks: usize,
    n_blocks: usize,
    tp: usize,
    threshold: f64,
) -> Option<EvalResult> {
    let rows = array.nrows();
    let start = Instant::now();
    let local = ClustererAdapter::new(SVDClusterer::new(k, 0.1));
    match DiMergeCoClusterer::new(k, rows, threshold, local, HierarchicalMergeConfig::default(), 16, tp, m_blocks, n_blocks) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let runtime = start.elapsed().as_secs_f64();
                let pred = extract_labels(&result.submatrices, rows, k);
                Some(EvalResult {
                    method: "spectral".to_string(),
                    nmi: calculate_nmi(true_labels, &pred),
                    ari: calculate_ari(true_labels, &pred),
                    time_secs: runtime,
                    n_clusters: result.submatrices.len(),
                    params: format!("{}x{},tp={},th={}", m_blocks, n_blocks, tp, threshold),
                })
            }
            Err(e) => { eprintln!("spectral ERROR: {}", e); None }
        },
        Err(e) => { eprintln!("spectral ERROR: {}", e); None }
    }
}

fn run_dimerge_nbvd(
    matrix: &Matrix<f64>,
    array: &Array2<f64>,
    true_labels: &[usize],
    k: usize,
    m_blocks: usize,
    n_blocks: usize,
    tp: usize,
    threshold: f64,
    max_iter: usize,
) -> Option<EvalResult> {
    let rows = array.nrows();
    let config = TriFactorConfig {
        n_row_clusters: k, n_col_clusters: k, max_iter, n_init: 1, tol: 1e-9, seed: None,
    };
    let start = Instant::now();
    let local = NbvdClusterer::with_config(config);
    match DiMergeCoClusterer::new(k, rows, threshold, local, HierarchicalMergeConfig::default(), 16, tp, m_blocks, n_blocks) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let runtime = start.elapsed().as_secs_f64();
                let pred = extract_labels(&result.submatrices, rows, k);
                Some(EvalResult {
                    method: "nbvd".to_string(),
                    nmi: calculate_nmi(true_labels, &pred),
                    ari: calculate_ari(true_labels, &pred),
                    time_secs: runtime,
                    n_clusters: result.submatrices.len(),
                    params: format!("{}x{},tp={},iter={}", m_blocks, n_blocks, tp, max_iter),
                })
            }
            Err(e) => { eprintln!("nbvd ERROR: {}", e); None }
        },
        Err(e) => { eprintln!("nbvd ERROR: {}", e); None }
    }
}

fn run_dimerge_onm3f(
    matrix: &Matrix<f64>,
    array: &Array2<f64>,
    true_labels: &[usize],
    k: usize,
    m_blocks: usize,
    n_blocks: usize,
    tp: usize,
    threshold: f64,
    max_iter: usize,
) -> Option<EvalResult> {
    let rows = array.nrows();
    let config = TriFactorConfig {
        n_row_clusters: k, n_col_clusters: k, max_iter, n_init: 1, tol: 1e-9, seed: None,
    };
    let start = Instant::now();
    let local = Onm3fClusterer::with_config(config);
    match DiMergeCoClusterer::new(k, rows, threshold, local, HierarchicalMergeConfig::default(), 16, tp, m_blocks, n_blocks) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let runtime = start.elapsed().as_secs_f64();
                let pred = extract_labels(&result.submatrices, rows, k);
                Some(EvalResult {
                    method: "onm3f".to_string(),
                    nmi: calculate_nmi(true_labels, &pred),
                    ari: calculate_ari(true_labels, &pred),
                    time_secs: runtime,
                    n_clusters: result.submatrices.len(),
                    params: format!("{}x{},tp={},iter={}", m_blocks, n_blocks, tp, max_iter),
                })
            }
            Err(e) => { eprintln!("onm3f ERROR: {}", e); None }
        },
        Err(e) => { eprintln!("onm3f ERROR: {}", e); None }
    }
}

fn run_dimerge_pnmtf(
    matrix: &Matrix<f64>,
    array: &Array2<f64>,
    true_labels: &[usize],
    k: usize,
    m_blocks: usize,
    n_blocks: usize,
    tp: usize,
    threshold: f64,
    max_iter: usize,
    tau: f64,
    eta: f64,
    gamma: f64,
) -> Option<EvalResult> {
    let rows = array.nrows();
    let config = TriFactorConfig {
        n_row_clusters: k, n_col_clusters: k, max_iter, n_init: 1, tol: 1e-9, seed: None,
    };
    let start = Instant::now();
    let local = PnmtfClusterer::with_config(config, tau, eta, gamma);
    match DiMergeCoClusterer::new(k, rows, threshold, local, HierarchicalMergeConfig::default(), 16, tp, m_blocks, n_blocks) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let runtime = start.elapsed().as_secs_f64();
                let pred = extract_labels(&result.submatrices, rows, k);
                Some(EvalResult {
                    method: "pnmtf".to_string(),
                    nmi: calculate_nmi(true_labels, &pred),
                    ari: calculate_ari(true_labels, &pred),
                    time_secs: runtime,
                    n_clusters: result.submatrices.len(),
                    params: format!("{}x{},tp={},iter={},tau={}", m_blocks, n_blocks, tp, max_iter, tau),
                })
            }
            Err(e) => { eprintln!("pnmtf ERROR: {}", e); None }
        },
        Err(e) => { eprintln!("pnmtf ERROR: {}", e); None }
    }
}

fn run_dimerge_fnmf(
    matrix: &Matrix<f64>,
    array: &Array2<f64>,
    true_labels: &[usize],
    k: usize,
    m_blocks: usize,
    n_blocks: usize,
    tp: usize,
    threshold: f64,
    max_iter: usize,
) -> Option<EvalResult> {
    let rows = array.nrows();
    let start = Instant::now();
    let local = FnmfClusterer::new(k, max_iter);
    match DiMergeCoClusterer::new(k, rows, threshold, local, HierarchicalMergeConfig::default(), 16, tp, m_blocks, n_blocks) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let runtime = start.elapsed().as_secs_f64();
                let pred = extract_labels(&result.submatrices, rows, k);
                Some(EvalResult {
                    method: "fnmf".to_string(),
                    nmi: calculate_nmi(true_labels, &pred),
                    ari: calculate_ari(true_labels, &pred),
                    time_secs: runtime,
                    n_clusters: result.submatrices.len(),
                    params: format!("{}x{},tp={},iter={}", m_blocks, n_blocks, tp, max_iter),
                })
            }
            Err(e) => { eprintln!("fnmf ERROR: {}", e); None }
        },
        Err(e) => { eprintln!("fnmf ERROR: {}", e); None }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .init();

    println!("Loading RCV1-train...");
    let array: Array2<f64> = ndarray_npy::read_npy("data/rcv1/rcv1_train.npy").expect("Failed to load data");
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/rcv1/rcv1_train_labels.npy").expect("Failed to load labels");
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| x as usize).collect();
    let (rows, cols) = (array.nrows(), array.ncols());
    println!("Loaded: {} x {} ({:.2} GB)", rows, cols, (rows * cols * 8) as f64 / 1e9);

    let matrix = Matrix::new(array.clone());
    let k = 4;

    println!("\n{:=<80}", "");
    println!("DiMergeCo + All Methods on RCV1-train");
    println!("{:=<80}", "");
    println!("{:<20} {:>7} {:>7} {:>9} {:>6} {}", "Method", "NMI", "ARI", "Time", "Clust", "Params");
    println!("{:-<80}", "");

    // ========== Phase 1: Initial run with default params ==========
    // Based on Classic4 findings: 8x8 blocks, tp=10, threshold=0.05
    let m_blocks = 8;
    let n_blocks = 8;
    let tp = 10;
    let threshold = 0.05;

    // Spectral (fast, run first)
    if let Some(r) = run_dimerge_spectral(&matrix, &array, &true_labels, k, m_blocks, n_blocks, tp, threshold) {
        print_result(&r);
    }

    // PNMTF (from Classic4: good accuracy/speed balance)
    if let Some(r) = run_dimerge_pnmtf(&matrix, &array, &true_labels, k, m_blocks, n_blocks, tp, threshold, 10, 0.1, 0.1, 0.1) {
        print_result(&r);
    }

    // FNMF (fast ANLS method)
    if let Some(r) = run_dimerge_fnmf(&matrix, &array, &true_labels, k, m_blocks, n_blocks, tp, threshold, 20) {
        print_result(&r);
    }

    // NBVD (slower but baseline NMF)
    if let Some(r) = run_dimerge_nbvd(&matrix, &array, &true_labels, k, m_blocks, n_blocks, tp, threshold, 10) {
        print_result(&r);
    }

    // ONM3F
    if let Some(r) = run_dimerge_onm3f(&matrix, &array, &true_labels, k, m_blocks, n_blocks, tp, threshold, 10) {
        print_result(&r);
    }

    println!("\n{:=<80}", "");
    println!("Parameter Sweep for Poor Performers");
    println!("{:=<80}", "");

    // ========== Phase 2: Parameter sweep ==========
    // Sweep block sizes
    for &(mb, nb) in &[(4, 4), (8, 8), (16, 16)] {
        if let Some(r) = run_dimerge_spectral(&matrix, &array, &true_labels, k, mb, nb, tp, threshold) {
            print_result(&r);
        }
    }

    // Sweep T_p for spectral
    for &t in &[5, 10, 20] {
        if let Some(r) = run_dimerge_spectral(&matrix, &array, &true_labels, k, 8, 8, t, threshold) {
            print_result(&r);
        }
    }

    // Sweep threshold
    for &th in &[0.01, 0.05, 0.1, 0.2] {
        if let Some(r) = run_dimerge_spectral(&matrix, &array, &true_labels, k, 8, 8, 10, th) {
            print_result(&r);
        }
    }

    // Sweep max_iter for PNMTF
    for &iter in &[5, 10, 20] {
        if let Some(r) = run_dimerge_pnmtf(&matrix, &array, &true_labels, k, 8, 8, 10, 0.05, iter, 0.1, 0.1, 0.1) {
            print_result(&r);
        }
    }

    // Sweep tau for PNMTF
    for &tau in &[0.01, 0.1, 1.0] {
        if let Some(r) = run_dimerge_pnmtf(&matrix, &array, &true_labels, k, 8, 8, 10, 0.05, 10, tau, tau, tau) {
            print_result(&r);
        }
    }

    println!("\nEvaluation complete.");
}
