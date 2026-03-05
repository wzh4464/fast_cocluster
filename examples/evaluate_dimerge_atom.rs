/// Evaluate DiMergeCo with various atom co-clustering methods on RCV1
///
/// This example benchmarks DiMergeCo using different local clusterers:
/// - SpectralCC (SVD baseline)
/// - NBVD, ONM3F, ONMTF, PNMTF, FNMF (atom NMF methods)
///
/// Usage:
///   cargo run --release --example evaluate_dimerge_atom -- [classic4|c4|classic4-small|c4s|rcv1|all]

use fast_cocluster::atom::{
    fnmf::FnmfClusterer,
    nbvd::NbvdClusterer,
    onm3f::Onm3fClusterer,
    tri_factor_base::TriFactorConfig,
};
use fast_cocluster::dimerge_co::parallel_coclusterer::LocalClusterer;
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::SVDClusterer;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use std::env;
use std::time::Instant;

fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    assert_eq!(true_labels.len(), pred_labels.len(), "label length mismatch");
    let n = true_labels.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let k_true = *true_labels.iter().max().unwrap_or(&0) + 1;
    let k_pred = *pred_labels.iter().max().unwrap_or(&0) + 1;

    let mut contingency = vec![vec![0usize; k_pred]; k_true];
    for i in 0..true_labels.len() {
        if pred_labels[i] < k_pred {
            contingency[true_labels[i]][pred_labels[i]] += 1;
        }
    }

    let row_sums: Vec<f64> = contingency
        .iter()
        .map(|r| r.iter().sum::<usize>() as f64)
        .collect();
    let col_sums: Vec<f64> = (0..k_pred)
        .map(|j| contingency.iter().map(|r| r[j]).sum::<usize>() as f64)
        .collect();

    let mut mi = 0.0;
    for i in 0..k_true {
        for j in 0..k_pred {
            let nij = contingency[i][j] as f64;
            if nij > 0.0 && row_sums[i] > 0.0 && col_sums[j] > 0.0 {
                mi += (nij / n) * ((nij * n) / (row_sums[i] * col_sums[j])).ln();
            }
        }
    }

    let h_true: f64 = row_sums
        .iter()
        .filter(|&&s| s > 0.0)
        .map(|&s| -(s / n) * (s / n).ln())
        .sum();
    let h_pred: f64 = col_sums
        .iter()
        .filter(|&&s| s > 0.0)
        .map(|&s| -(s / n) * (s / n).ln())
        .sum();

    if h_true + h_pred == 0.0 {
        0.0
    } else {
        2.0 * mi / (h_true + h_pred)
    }
}

fn calculate_ari(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    assert_eq!(true_labels.len(), pred_labels.len(), "label length mismatch");
    let n = true_labels.len();
    if n < 2 {
        return 0.0;
    }
    let k_true = *true_labels.iter().max().unwrap_or(&0) + 1;
    let k_pred = *pred_labels.iter().max().unwrap_or(&0) + 1;

    let mut contingency = vec![vec![0i64; k_pred]; k_true];
    for i in 0..n {
        if pred_labels[i] < k_pred {
            contingency[true_labels[i]][pred_labels[i]] += 1;
        }
    }

    let comb2 = |x: i64| -> i64 {
        if x < 2 {
            0
        } else {
            x * (x - 1) / 2
        }
    };

    let sum_comb_nij: i64 = contingency
        .iter()
        .flat_map(|r| r.iter())
        .map(|&x| comb2(x))
        .sum();
    let sum_comb_ai: i64 = contingency
        .iter()
        .map(|r| comb2(r.iter().sum::<i64>()))
        .sum();
    let sum_comb_bj: i64 = (0..k_pred)
        .map(|j| comb2(contingency.iter().map(|r| r[j]).sum::<i64>()))
        .sum();
    let comb_n = comb2(n as i64) as f64;
    if comb_n < 1.0 {
        return 0.0;
    }

    let expected = (sum_comb_ai as f64) * (sum_comb_bj as f64) / comb_n;
    let max_idx = 0.5 * (sum_comb_ai as f64 + sum_comb_bj as f64);
    let denom = max_idx - expected;

    if denom.abs() < 1e-12 {
        0.0
    } else {
        (sum_comb_nij as f64 - expected) / denom
    }
}

fn extract_labels(submatrices: &[Submatrix<'_, f64>], n_rows: usize, k: usize) -> Vec<usize> {
    use linfa::prelude::*;
    use linfa_clustering::KMeans;

    if submatrices.is_empty() {
        return vec![0; n_rows];
    }
    let n_coclusters = submatrices.len();
    let mut membership = ndarray::Array2::<f64>::zeros((n_rows, n_coclusters));
    for (cid, sm) in submatrices.iter().enumerate() {
        for &row_idx in &sm.row_indices {
            if row_idx < n_rows {
                membership[[row_idx, cid]] = 1.0;
            }
        }
    }
    let dataset = DatasetBase::from(membership);
    let model = match KMeans::params(k)
        .max_n_iterations(200)
        .fit(&dataset)
    {
        Ok(m) => m,
        Err(e) => {
            eprintln!("K-means failed: {}; falling back to all-zeros labels", e);
            return vec![0; n_rows];
        }
    };
    model.predict(dataset).targets.to_vec()
}

fn make_config(k: usize) -> TriFactorConfig {
    TriFactorConfig {
        n_row_clusters: k,
        n_col_clusters: k,
        max_iter: 20,  // Reduced for faster evaluation
        n_init: 3,
        tol: 1e-9,
        seed: None,
    }
}

/// Run a single DiMergeCo benchmark with the given local clusterer and print results.
fn run_dimerge_benchmark(
    method_name: &str,
    local: impl LocalClusterer,
    k: usize,
    n_rows: usize,
    delta: f64,
    num_threads: usize,
    tp: usize,
    m_blocks: usize,
    n_blocks: usize,
    matrix: &Matrix<f64>,
    true_labels: &[usize],
) {
    let start = Instant::now();
    match DiMergeCoClusterer::new(k, n_rows, delta, local, HierarchicalMergeConfig::default(), num_threads, tp, m_blocks, n_blocks) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let runtime = start.elapsed().as_secs_f64();
                let pred = extract_labels(&result.submatrices, n_rows, k);
                println!("{:<12} {:>8.4} {:>8.4} {:>9.1}s {:>8}",
                    method_name, calculate_nmi(true_labels, &pred), calculate_ari(true_labels, &pred), runtime, result.submatrices.len());
            }
            Err(e) => println!("{:<12} ERROR: {}", method_name, e),
        },
        Err(e) => println!("{:<12} ERROR: {}", method_name, e),
    }
}

fn evaluate_dataset(dataset_name: &str, data_path: &str, labels_path: &str) {
    // --- Named constants ---
    const K: usize = 4;
    const NUM_THREADS: usize = 16;
    const TP: usize = 10;
    const DELTA: f64 = 0.05;
    // PNMTF penalty parameters
    const PNMTF_TAU: f64 = 0.1;
    const PNMTF_ETA: f64 = 0.1;
    const PNMTF_GAMMA: f64 = 0.1;

    println!("\n{}", "=".repeat(70));
    println!("Evaluating {} dataset", dataset_name);
    println!("{}", "=".repeat(70));

    // Load data
    let load_start = Instant::now();
    let array: Array2<f64> = match ndarray_npy::read_npy(data_path) {
        Ok(a) => a,
        Err(e) => {
            println!("Failed to load {}: {}", data_path, e);
            return;
        }
    };
    let labels_array: ndarray::Array1<i64> = match ndarray_npy::read_npy(labels_path) {
        Ok(a) => a,
        Err(e) => {
            println!("Failed to load {}: {}", labels_path, e);
            return;
        }
    };
    let true_labels: Vec<usize> = labels_array
        .iter()
        .map(|&x| {
            assert!(x >= 0, "negative label found: {}", x);
            x as usize
        })
        .collect();
    let (rows, cols) = (array.nrows(), array.ncols());
    println!(
        "Loaded in {:.1}s: {} x {} ({:.2} GB dense)",
        load_start.elapsed().as_secs_f64(),
        rows,
        cols,
        (rows * cols * 8) as f64 / 1e9
    );

    let matrix = Matrix::new(array);
    // Use more blocks for larger datasets to reduce per-block computation
    let (m_blocks, n_blocks) = if rows > 10000 || cols > 10000 {
        (8, 8)  // RCV1-scale: ~2900x5900 per block
    } else if rows > 2000 || cols > 2000 {
        (4, 4)  // Classic4-paper scale
    } else {
        (2, 2)  // Small datasets
    };

    // --- Standalone Spectral baseline for comparison ---
    println!("\n--- Standalone Baseline ---");
    {
        let start = Instant::now();
        let clusterer = ClustererAdapter::new(SVDClusterer::new(K, 0.1));
        match clusterer.cluster_local(&matrix.data) {
            Ok(subs) => {
                let runtime = start.elapsed().as_secs_f64();
                let pred = extract_labels(&subs, rows, K);
                println!("spectral(standalone): NMI={:.4}, ARI={:.4}, Time={:.1}s",
                    calculate_nmi(&true_labels, &pred), calculate_ari(&true_labels, &pred), runtime);
            }
            Err(e) => println!("spectral(standalone) ERROR: {}", e),
        }
    }

    // ─── DiMergeCo + each atom method ───────────────────────────────
    println!("\n--- DiMergeCo ({}x{} blocks, T_p={}) + Local Clusterers ---", m_blocks, n_blocks, TP);
    println!("{:<12} {:>8} {:>8} {:>10} {:>8}", "Method", "NMI", "ARI", "Time(s)", "Clusters");
    println!("{}", "-".repeat(50));

    // DiMergeCo + Spectral
    run_dimerge_benchmark("spectral",
        ClustererAdapter::new(SVDClusterer::new(K, 0.1)),
        K, rows, DELTA, NUM_THREADS, TP, m_blocks, n_blocks, &matrix, &true_labels);

    // DiMergeCo + NBVD
    run_dimerge_benchmark("nbvd",
        NbvdClusterer::with_config(make_config(K)),
        K, rows, DELTA, NUM_THREADS, TP, m_blocks, n_blocks, &matrix, &true_labels);

    // DiMergeCo + ONM3F
    run_dimerge_benchmark("onm3f",
        Onm3fClusterer::with_config(make_config(K)),
        K, rows, DELTA, NUM_THREADS, TP, m_blocks, n_blocks, &matrix, &true_labels);

    // DiMergeCo + ONMTF - SKIPPED (too slow due to O(n^2) intermediate matrices)
    // DiMergeCo + PNMTF - SKIPPED (too slow due to O(n^2) intermediate matrices)
    // When re-enabled, use PnmtfClusterer::with_config(make_config(K), PNMTF_TAU, PNMTF_ETA, PNMTF_GAMMA)
    let _ = (PNMTF_TAU, PNMTF_ETA, PNMTF_GAMMA); // suppress unused warnings

    // DiMergeCo + FNMF
    run_dimerge_benchmark("fnmf",
        FnmfClusterer::new(K, K),
        K, rows, DELTA, NUM_THREADS, TP, m_blocks, n_blocks, &matrix, &true_labels);
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .init();

    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("train");

    println!("DiMergeCo + Atom Methods Evaluation");
    println!("Mode: {}", mode);

    match mode {
        "classic4" | "c4" => {
            evaluate_dataset(
                "Classic4-paper",
                "data/classic4_paper.npy",
                "data/classic4_paper_labels.npy",
            );
        }
        "classic4-small" | "c4s" => {
            evaluate_dataset(
                "Classic4-small",
                "data/classic4_benchmark_small.npy",
                "data/classic4_benchmark_small_labels.npy",
            );
        }
        "rcv1" | "train" => {
            evaluate_dataset(
                "RCV1-train",
                "data/rcv1/rcv1_train.npy",
                "data/rcv1/rcv1_train_labels.npy",
            );
        }
        "all" => {
            evaluate_dataset(
                "Classic4-paper",
                "data/classic4_paper.npy",
                "data/classic4_paper_labels.npy",
            );
            evaluate_dataset(
                "RCV1-train",
                "data/rcv1/rcv1_train.npy",
                "data/rcv1/rcv1_train_labels.npy",
            );
        }
        _ => {
            println!("Usage: evaluate_dimerge_atom [classic4|c4|classic4-small|c4s|rcv1|all]");
        }
    }

    println!("\nEvaluation complete.");
}
