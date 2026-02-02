/// Run SCC baseline + DiMergeCo on full Classic4 (6460×4667)
/// with randomized SVD, compare to previous full-SVD results.
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::{Clusterer, SVDClusterer};
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use std::time::Instant;

fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len() as f64;
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
    let model = KMeans::params(k)
        .max_n_iterations(300)
        .fit(&dataset)
        .expect("K-means failed");
    model.predict(dataset).targets.to_vec()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let array: Array2<f64> = ndarray_npy::read_npy("data/classic4_paper.npy")?;
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/classic4_paper_labels.npy")?;
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| x as usize).collect();
    let (rows, cols) = (array.nrows(), array.ncols());
    let matrix = Matrix::new(array);

    println!("Full Classic4: {} x {}", rows, cols);
    println!("Running SCC baseline (randomized SVD)...");

    let k = 4;
    let start = Instant::now();
    let clusterer = SVDClusterer::new(k, 0.1);
    let submatrices = clusterer.cluster(&matrix).expect("Baseline SCC failed");
    let runtime_baseline = start.elapsed().as_secs_f64();
    let runtime = runtime_baseline;

    let pred = extract_labels(&submatrices, rows, k);
    let nmi = calculate_nmi(&true_labels, &pred);
    let ari = calculate_ari(&true_labels, &pred);

    println!("\nResults:");
    println!("  NMI:     {:.4}", nmi);
    println!("  ARI:     {:.4}", ari);
    println!("  Time:    {:.3}s", runtime);
    println!("  Clusters: {}", submatrices.len());

    println!("\nComparison:");
    println!("  Previous (full SVD):  NMI=0.7908  ARI=0.7498  Time=1930s");
    println!("  Current  (rand SVD):  NMI={:.4}  ARI={:.4}  Time={:.1}s", nmi, ari, runtime);
    println!("  Speedup: {:.0}x", 1930.0 / runtime);

    // ─── DiMergeCo configs ────────────────────────────────────────
    println!("\n{}", "=".repeat(70));
    println!("DiMergeCo + Randomized SVD on Full Classic4");
    println!("{}", "=".repeat(70));

    let num_threads = num_cpus::get(); // use all CPUs
    // Top configs from previous full-SVD benchmark + a few larger ones
    let configs: Vec<(&str, usize, usize, usize)> = vec![
        ("2x2_tp10",  2, 2, 10),
        ("2x2_tp20",  2, 2, 20),
        ("2x2_tp30",  2, 2, 30),
        ("2x3_tp10",  2, 3, 10),
        ("2x3_tp20",  2, 3, 20),
        ("2x3_tp30",  2, 3, 30),
        ("3x2_tp10",  3, 2, 10),
        ("3x2_tp20",  3, 2, 20),
        ("3x3_tp10",  3, 3, 10),
        ("3x3_tp20",  3, 3, 20),
        ("3x3_tp30",  3, 3, 30),
        ("2x4_tp20",  2, 4, 20),
        ("2x4_tp30",  2, 4, 30),
        ("4x3_tp20",  4, 3, 20),
        ("4x4_tp30",  4, 4, 30),
    ];

    println!("{:<14} {:>6} {:>4} {:>8} {:>8} {:>10} {:>10}",
        "Config", "Blocks", "T_p", "NMI", "ARI", "Time(s)", "OldTime(s)");
    println!("{}", "-".repeat(70));

    // Old full-SVD results for comparison
    let old_results: Vec<(&str, f64, f64, f64)> = vec![
        ("2x2_tp10",  0.7418, 0.6905, 3417.91),
        ("2x2_tp20",  0.7462, 0.6815, 5558.90),
        ("2x2_tp30",  0.7491, 0.6806, 7446.95),
        ("2x3_tp10",  0.7251, 0.5892, 1572.11),
        ("2x3_tp20",  0.7338, 0.6256, 2597.85),
        ("2x3_tp30",  0.7435, 0.6465, 3175.93),
        ("3x2_tp10",  0.7423, 0.6749, 2323.14),
        ("3x2_tp20",  0.7438, 0.6638, 3680.75),
        ("3x3_tp10",  0.7244, 0.6337, 2217.64),
        ("3x3_tp20",  0.7258, 0.5980, 3207.08),
        ("3x3_tp30",  0.7289, 0.6013, 4238.49),
        ("2x4_tp20",  0.7290, 0.5834, 1905.22),
        ("2x4_tp30",  0.7294, 0.5866, 2319.44),
        ("4x3_tp20",  0.7239, 0.6220, 2785.51),
        ("4x4_tp30",  0.7018, 0.5278, 3094.28),
    ];

    for (label, mb, nb, tp) in &configs {
        let start = Instant::now();
        let local_clusterer = ClustererAdapter::new(SVDClusterer::new(k, 0.1));
        let clusterer = DiMergeCoClusterer::new(
            k, rows, 0.05, local_clusterer,
            HierarchicalMergeConfig::default(),
            num_threads, *tp, *mb, *nb,
        );
        match clusterer {
            Ok(c) => match c.run(&matrix) {
                Ok(result) => {
                    let runtime = start.elapsed().as_secs_f64();
                    let pred = extract_labels(&result.submatrices, rows, k);
                    let nmi = calculate_nmi(&true_labels, &pred);
                    let ari = calculate_ari(&true_labels, &pred);
                    let old_time = old_results.iter()
                        .find(|(n, _, _, _)| *n == *label)
                        .map(|(_, _, _, t)| format!("{:.1}", t))
                        .unwrap_or_else(|| "-".to_string());
                    println!("{:<14} {:>3}x{:<2} {:>4} {:>8.4} {:>8.4} {:>9.1}s {:>10}",
                        label, mb, nb, tp, nmi, ari, runtime, old_time);
                }
                Err(e) => println!("{:<14} ERROR: {}", label, e),
            },
            Err(e) => println!("{:<14} ERROR: {}", label, e),
        }
    }

    println!("\n{:<14} {:>6} {:>4} {:>8} {:>8} {:>10} {:>10}",
        "baseline", "-", "-",
        &format!("{:.4}", calculate_nmi(&true_labels, &extract_labels(&submatrices, rows, k))),
        &format!("{:.4}", calculate_ari(&true_labels, &extract_labels(&submatrices, rows, k))),
        &format!("{:.1}s", runtime_baseline),
        "1930.3");

    Ok(())
}
