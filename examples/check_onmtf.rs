//! ONMTF consistency check on Classic4 dataset.
//!
//! Runs Rust ONMTF on Classic4 with seeds 0-9, compares NMI/ARI against Python baselines.
//! Run: cargo run --release --example check_onmtf

use fast_cocluster::atom::onmtf::OnmtfClusterer;
use fast_cocluster::atom::TriFactorConfig;
use fast_cocluster::dimerge_co::parallel_coclusterer::LocalClusterer;
use ndarray::Array2;

fn load_classic4() -> Result<(Array2<f64>, Vec<usize>), Box<dyn std::error::Error>> {
    let data_path = "data/classic4_benchmark_small.npy";
    let labels_path = "data/classic4_benchmark_small_labels.npy";
    let array: Array2<f64> = ndarray_npy::read_npy(data_path)?;
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy(labels_path)?;
    let labels: Vec<usize> = labels_array
        .iter()
        .map(|&x| {
            assert!(x >= 0, "negative label found: {}", x);
            x as usize
        })
        .collect();
    println!(
        "Loaded Classic4: {} x {}, {} labels",
        array.nrows(),
        array.ncols(),
        labels.len()
    );
    Ok((array, labels))
}

fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    assert_eq!(true_labels.len(), pred_labels.len(), "label length mismatch");
    let n = true_labels.len() as f64;
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
    if n < 2 { return 0.0; }
    let comb_n = comb2(n as i64) as f64;
    let expected = (sum_comb_ai as f64) * (sum_comb_bj as f64) / comb_n;
    let max_idx = 0.5 * (sum_comb_ai as f64 + sum_comb_bj as f64);
    let denom = max_idx - expected;
    if denom.abs() < 1e-12 {
        0.0
    } else {
        (sum_comb_nij as f64 - expected) / denom
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x, true_labels) = load_classic4()?;
    let n_row_clusters = 4;
    let n_col_clusters = 4;
    let n_seeds = 10;

    println!("\nRunning ONMTF with {} seeds...", n_seeds);
    let mut nmis = Vec::new();
    let mut aris = Vec::new();

    for seed in 0..n_seeds {
        let config = TriFactorConfig {
            n_row_clusters,
            n_col_clusters,
            max_iter: 50,
            n_init: 1,
            tol: 1e-9,
            seed: Some(seed),
        };
        let clusterer = OnmtfClusterer::with_config(config);
        let submatrices = clusterer.cluster_local(&x)?;

        // Extract row labels from submatrices
        let mut pred_labels = vec![0usize; x.nrows()];
        for (cid, sub) in submatrices.iter().enumerate() {
            for &r in &sub.row_indices {
                pred_labels[r] = cid;
            }
        }

        let nmi = calculate_nmi(&true_labels, &pred_labels);
        let ari = calculate_ari(&true_labels, &pred_labels);
        println!("  seed {}: NMI={:.4}, ARI={:.4}", seed, nmi, ari);
        nmis.push(nmi);
        aris.push(ari);
    }

    let mean_nmi: f64 = nmis.iter().sum::<f64>() / nmis.len() as f64;
    let mean_ari: f64 = aris.iter().sum::<f64>() / aris.len() as f64;
    let std_nmi =
        (nmis.iter().map(|x| (x - mean_nmi).powi(2)).sum::<f64>() / nmis.len() as f64).sqrt();
    let std_ari =
        (aris.iter().map(|x| (x - mean_ari).powi(2)).sum::<f64>() / aris.len() as f64).sqrt();

    println!("\nONMTF Results (Rust):");
    println!("  NMI: {:.4} +/- {:.4}", mean_nmi, std_nmi);
    println!("  ARI: {:.4} +/- {:.4}", mean_ari, std_ari);

    // Load and compare with Python baseline if available
    let baseline_path = "data/atom_baselines/onmtf_classic4.json";
    if let Ok(content) = std::fs::read_to_string(baseline_path) {
        let baseline: serde_json::Value = serde_json::from_str(&content)?;
        let py_nmi = baseline["mean_nmi"].as_f64().unwrap_or(0.0);
        let py_ari = baseline["mean_ari"].as_f64().unwrap_or(0.0);
        println!("\nPython Baseline:");
        println!("  NMI: {:.4}", py_nmi);
        println!("  ARI: {:.4}", py_ari);
        println!(
            "\nDiff: NMI={:.4}, ARI={:.4}",
            (mean_nmi - py_nmi).abs(),
            (mean_ari - py_ari).abs()
        );
        if (mean_nmi - py_nmi).abs() < 0.05 && (mean_ari - py_ari).abs() < 0.05 {
            println!("PASS: Within tolerance");
        } else {
            println!("WARN: Outside tolerance (0.05)");
        }
    } else {
        println!(
            "\nNo Python baseline found at {}. Run scripts/generate_atom_baselines.py first.",
            baseline_path
        );
    }

    Ok(())
}
