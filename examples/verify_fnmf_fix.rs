//! Verify FNMF fix on RCV1-all stratified data.
//! Runs seeds 0-2, prints NMI/ARI/rel_error for comparison with Python baseline.
//! Python baseline: NMI=0.2974, ARI=0.1526, rel_err=0.975438

use fast_cocluster::atom::fnmf::FnmfClusterer;
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
    let expected = (sum_comb_ai as f64) * (sum_comb_bj as f64) / comb_n;
    let max_idx = 0.5 * (sum_comb_ai as f64 + sum_comb_bj as f64);
    let denom = max_idx - expected;
    if denom.abs() < 1e-12 { 0.0 } else { (sum_comb_nij as f64 - expected) / denom }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading RCV1-all stratified...");
    let x: Array2<f64> = ndarray_npy::read_npy("data/rcv1/rcv1_all.npy")?;
    let labels_arr: ndarray::Array1<i64> = ndarray_npy::read_npy("data/rcv1/rcv1_all_labels.npy")?;
    let true_labels: Vec<usize> = labels_arr.iter().map(|&v| { assert!(v >= 0); v as usize }).collect();
    println!("Shape: {} x {}, k=4", x.nrows(), x.ncols());
    println!("Python baseline: NMI=0.2974, ARI=0.1526, rel_err=0.975438\n");

    println!("{:>5} {:>9} {:>9} {:>10} {:>9}", "seed", "NMI", "ARI", "rel_err", "time(s)");
    for seed in 0..3u64 {
        let start = Instant::now();
        let clusterer = FnmfClusterer {
            n_row_clusters: 4, n_col_clusters: 4,
            max_iter: 100, n_init: 1, seed: Some(seed),
        };
        let (pred, _col, rel_err) = clusterer.fit_labels(&x);
        let elapsed = start.elapsed().as_secs_f64();
        let nmi = calculate_nmi(&true_labels, &pred);
        let ari = calculate_ari(&true_labels, &pred);
        println!("{:>5} {:>9.4} {:>9.4} {:>10.6} {:>9.1}", seed, nmi, ari, rel_err, elapsed);
    }
    Ok(())
}
