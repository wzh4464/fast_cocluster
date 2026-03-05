/// Comprehensive timing benchmark for all methods on Classic4 (6460x4667).
///
/// Runs each method one at a time for proper isolation.
/// Outputs JSON results and a formatted table.
///
/// Usage:
///   cargo run --release --example benchmark_all_timings

use fast_cocluster::atom::{
    FnmfClusterer, NbvdClusterer, Onm3fClusterer, OnmtfClusterer, PnmtfClusterer,
    TriFactorConfig,
};
use fast_cocluster::dimerge_co::parallel_coclusterer::LocalClusterer;
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::SVDClusterer;
use fast_cocluster::spectral_cocluster::SpectralCocluster;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    assert_eq!(true_labels.len(), pred_labels.len());
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
    assert_eq!(true_labels.len(), pred_labels.len());
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

// ---------------------------------------------------------------------------
// Label extraction
// ---------------------------------------------------------------------------

fn extract_labels_from_submatrices(
    submatrices: &[Submatrix<'_, f64>],
    n_rows: usize,
    k: usize,
) -> Vec<usize> {
    use linfa::prelude::*;
    use linfa_clustering::KMeans;

    if submatrices.is_empty() {
        return vec![0; n_rows];
    }
    let n_coclusters = submatrices.len();
    let mut membership = Array2::<f64>::zeros((n_rows, n_coclusters));
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

// ---------------------------------------------------------------------------
// Result record
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct BenchmarkResult {
    method: String,
    time_s: Option<f64>,
    nmi: Option<f64>,
    ari: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    t_p: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocks: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn make_tri_config(k: usize) -> TriFactorConfig {
    TriFactorConfig {
        n_row_clusters: k,
        n_col_clusters: k,
        max_iter: 100,
        n_init: 1,
        tol: 1e-9,
        seed: Some(0),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .init();

    // Load Classic4 paper dataset
    println!("Loading Classic4 paper dataset...");
    let array: Array2<f64> = ndarray_npy::read_npy("data/classic4_paper.npy")?;
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/classic4_paper_labels.npy")?;
    let true_labels: Vec<usize> = labels_array
        .iter()
        .map(|&x| {
            assert!(x >= 0, "negative label: {}", x);
            x as usize
        })
        .collect();
    let (rows, cols) = (array.nrows(), array.ncols());
    println!("Loaded: {} x {} ({} labels)", rows, cols, true_labels.len());

    let k = 4;
    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Helper: run a standalone LocalClusterer on the full array
    let run_standalone =
        |name: &str, clusterer: &dyn LocalClusterer, array: &Array2<f64>| -> BenchmarkResult {
            println!("\nRunning {}...", name);
            let start = Instant::now();
            match clusterer.cluster_local(array) {
                Ok(subs) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    let pred = extract_labels_from_submatrices(&subs, rows, k);
                    let nmi = calculate_nmi(&true_labels, &pred);
                    let ari = calculate_ari(&true_labels, &pred);
                    println!("  {} done: {:.1}s  NMI={:.4}  ARI={:.4}", name, elapsed, nmi, ari);
                    BenchmarkResult {
                        method: name.to_string(),
                        time_s: Some(elapsed),
                        nmi: Some(nmi),
                        ari: Some(ari),
                        t_p: None,
                        blocks: None,
                        error: None,
                    }
                }
                Err(e) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    println!("  {} ERROR after {:.1}s: {}", name, elapsed, e);
                    BenchmarkResult {
                        method: name.to_string(),
                        time_s: Some(elapsed),
                        nmi: None,
                        ari: None,
                        t_p: None,
                        blocks: None,
                        error: Some(e.to_string()),
                    }
                }
            }
        };

    // -----------------------------------------------------------------------
    // 1. SCC (SVD + KMeans, standalone via ClustererAdapter)
    // -----------------------------------------------------------------------
    {
        let scc = ClustererAdapter::new(SVDClusterer::new(k, 0.1));
        results.push(run_standalone("SCC", &scc, &array));
    }

    // -----------------------------------------------------------------------
    // 2. SpectralCC (full bipartite Laplacian SVD)
    // -----------------------------------------------------------------------
    {
        println!("\nRunning SpectralCC...");
        let matrix = Matrix::new(array.clone());
        let spectral = SpectralCocluster::new(k, k);
        let start = Instant::now();
        match spectral.fit(&matrix) {
            Ok(subs) => {
                let elapsed = start.elapsed().as_secs_f64();
                let pred = extract_labels_from_submatrices(&subs, rows, k);
                let nmi = calculate_nmi(&true_labels, &pred);
                let ari = calculate_ari(&true_labels, &pred);
                println!("  SpectralCC done: {:.1}s  NMI={:.4}  ARI={:.4}", elapsed, nmi, ari);
                results.push(BenchmarkResult {
                    method: "SpectralCC".to_string(),
                    time_s: Some(elapsed),
                    nmi: Some(nmi),
                    ari: Some(ari),
                    t_p: None,
                    blocks: None,
                    error: None,
                });
            }
            Err(e) => {
                let elapsed = start.elapsed().as_secs_f64();
                println!("  SpectralCC ERROR after {:.1}s: {}", elapsed, e);
                results.push(BenchmarkResult {
                    method: "SpectralCC".to_string(),
                    time_s: Some(elapsed),
                    nmi: None,
                    ari: None,
                    t_p: None,
                    blocks: None,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. FNMF (standalone)
    // -----------------------------------------------------------------------
    {
        let fnmf = FnmfClusterer::new(k, 50);
        results.push(run_standalone("FNMF", &fnmf, &array));
    }

    // -----------------------------------------------------------------------
    // 4. NBVD (standalone)
    // -----------------------------------------------------------------------
    {
        let nbvd = NbvdClusterer::with_config(make_tri_config(k));
        results.push(run_standalone("NBVD", &nbvd, &array));
    }

    // -----------------------------------------------------------------------
    // 5. ONM3F (standalone)
    // -----------------------------------------------------------------------
    {
        let onm3f = Onm3fClusterer::with_config(make_tri_config(k));
        results.push(run_standalone("ONM3F", &onm3f, &array));
    }

    // -----------------------------------------------------------------------
    // 6. ONMTF (standalone)
    // -----------------------------------------------------------------------
    {
        let onmtf = OnmtfClusterer::with_config(make_tri_config(k));
        results.push(run_standalone("ONMTF", &onmtf, &array));
    }

    // -----------------------------------------------------------------------
    // 7. PNMTF (standalone -- may be very slow, Ctrl-C if needed)
    // -----------------------------------------------------------------------
    {
        let pnmtf = PnmtfClusterer::new(k, k).with_penalties(0.1, 0.1, 0.1);
        println!("\nRunning PNMTF (may take >1h, Ctrl-C to skip)...");
        results.push(run_standalone("PNMTF", &pnmtf, &array));
    }

    // -----------------------------------------------------------------------
    // 8. DiMergeCo-SCC (2x2, T_p=10)
    // -----------------------------------------------------------------------
    let matrix = Matrix::new(array);

    for &tp in &[10usize, 30] {
        let label = format!("DiMergeCo-SCC_tp{}", tp);
        println!("\nRunning {} (2x2 blocks)...", label);
        let start = Instant::now();
        let local = ClustererAdapter::new(SVDClusterer::new(k, 0.1));
        let num_threads = num_cpus::get();
        match DiMergeCoClusterer::new(
            k,
            rows,
            0.05,
            local,
            HierarchicalMergeConfig::default(),
            num_threads,
            tp,
            2,
            2,
        ) {
            Ok(c) => match c.run(&matrix) {
                Ok(result) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    let pred = extract_labels_from_submatrices(&result.submatrices, rows, k);
                    let nmi = calculate_nmi(&true_labels, &pred);
                    let ari = calculate_ari(&true_labels, &pred);
                    println!("  {} done: {:.1}s  NMI={:.4}  ARI={:.4}", label, elapsed, nmi, ari);
                    results.push(BenchmarkResult {
                        method: label,
                        time_s: Some(elapsed),
                        nmi: Some(nmi),
                        ari: Some(ari),
                        t_p: Some(tp),
                        blocks: Some("2x2".to_string()),
                        error: None,
                    });
                }
                Err(e) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    println!("  {} ERROR: {}", label, e);
                    results.push(BenchmarkResult {
                        method: label,
                        time_s: Some(elapsed),
                        nmi: None,
                        ari: None,
                        t_p: Some(tp),
                        blocks: Some("2x2".to_string()),
                        error: Some(e.to_string()),
                    });
                }
            },
            Err(e) => {
                println!("  {} ERROR: {}", label, e);
                results.push(BenchmarkResult {
                    method: label,
                    time_s: None,
                    nmi: None,
                    ari: None,
                    t_p: Some(tp),
                    blocks: Some("2x2".to_string()),
                    error: Some(e.to_string()),
                });
            }
        }
    }

    // -----------------------------------------------------------------------
    // Output formatted table
    // -----------------------------------------------------------------------
    println!("\n{}", "=".repeat(75));
    println!("Classic4 Timing Benchmark Results ({} x {})", rows, cols);
    println!("{}", "=".repeat(75));
    println!(
        "{:<25} {:>10} {:>8} {:>8}  {}",
        "Method", "Time(s)", "NMI", "ARI", "Notes"
    );
    println!("{}", "-".repeat(75));

    for r in &results {
        let time_str = r
            .time_s
            .map(|t| format!("{:.1}", t))
            .unwrap_or_else(|| "-".to_string());
        let nmi_str = r
            .nmi
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "-".to_string());
        let ari_str = r
            .ari
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "-".to_string());
        let notes = match (&r.blocks, &r.error) {
            (Some(b), None) => format!("blocks={}", b),
            (_, Some(e)) => e.clone(),
            _ => String::new(),
        };
        println!(
            "{:<25} {:>10} {:>8} {:>8}  {}",
            r.method, time_str, nmi_str, ari_str, notes
        );
    }
    println!("{}", "=".repeat(75));

    // -----------------------------------------------------------------------
    // Write JSON
    // -----------------------------------------------------------------------
    let output = serde_json::json!({
        "dataset": format!("classic4_paper ({}x{})", rows, cols),
        "hardware_info": format!("{} cores, {}", num_cpus::get(), std::env::consts::OS),
        "date": chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        "results": results,
    });

    let json_path = "baselines/results/classic4_timing_benchmark.json";
    std::fs::create_dir_all("baselines/results")?;
    std::fs::write(json_path, serde_json::to_string_pretty(&output)?)?;
    println!("\nResults written to {}", json_path);

    Ok(())
}
