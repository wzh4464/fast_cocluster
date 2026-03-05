//! Ablation study: compare merge strategies for DiMergeCo (10 seeds)
//!
//! Tests all Rust merge strategies (Adaptive, Union, Intersection, Weighted)
//! on Classic4 and BCW datasets with 10 random seeds each, measuring
//! mean +/- std for NMI, ARI, and time.
//!
//! Results saved to baselines/results/rust_merge_ablation_10seeds.json
//!
//! Run with: RUST_LOG=warn cargo run --release --example ablation_merge_strategy

use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::{Clusterer, SVDClusterer};
use fast_cocluster::submatrix::Submatrix;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::Array2;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

const NUM_SEEDS: u64 = 10;

fn load_classic4() -> Result<(Matrix<f64>, Vec<usize>, usize), Box<dyn std::error::Error>> {
    let array: Array2<f64> = ndarray_npy::read_npy("data/classic4_paper.npy")
        .map_err(|e| format!("Failed to load classic4_paper.npy: {}", e))?;
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/classic4_paper_labels.npy")
        .map_err(|e| format!("Failed to load classic4_paper_labels.npy: {}", e))?;
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| {
        assert!(x >= 0, "Negative label in classic4_paper_labels.npy");
        x as usize
    }).collect();
    let matrix = Matrix::new(array);
    println!("Loaded Classic4: {} x {}, k=4", matrix.rows, matrix.cols);
    Ok((matrix, true_labels, 4))
}

fn load_bcw() -> Result<(Matrix<f64>, Vec<usize>, usize), Box<dyn std::error::Error>> {
    let array: Array2<f64> = ndarray_npy::read_npy("data/bcw.npy")
        .map_err(|e| format!("Failed to load bcw.npy: {}", e))?;
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/bcw_labels.npy")
        .map_err(|e| format!("Failed to load bcw_labels.npy: {}", e))?;
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| {
        assert!(x >= 0, "Negative label in bcw_labels.npy");
        x as usize
    }).collect();
    let matrix = Matrix::new(array);
    println!("Loaded BCW: {} x {}, k=2", matrix.rows, matrix.cols);
    Ok((matrix, true_labels, 2))
}

fn calculate_nmi(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len() as f64;
    if n < 2.0 { return 0.0; }

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
        if p > 0.0 { h_true -= p * p.ln(); }
    }

    let mut h_pred = 0.0;
    for &count in pred_counts.values() {
        let p = count as f64 / n;
        if p > 0.0 { h_pred -= p * p.ln(); }
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

    if h_true + h_pred > 0.0 { 2.0 * mi / (h_true + h_pred) } else { 0.0 }
}

fn calculate_ari(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len();
    if n < 2 { return 0.0; }

    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut true_counts: HashMap<usize, usize> = HashMap::new();
    let mut pred_counts: HashMap<usize, usize> = HashMap::new();

    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency.entry((t, p)).or_insert(0) += 1;
        *true_counts.entry(t).or_insert(0) += 1;
        *pred_counts.entry(p).or_insert(0) += 1;
    }

    let comb2 = |x: usize| -> f64 { if x < 2 { 0.0 } else { (x * (x - 1)) as f64 / 2.0 } };

    let sum_comb_nij: f64 = contingency.values().map(|&v| comb2(v)).sum();
    let sum_comb_ai: f64 = true_counts.values().map(|&v| comb2(v)).sum();
    let sum_comb_bj: f64 = pred_counts.values().map(|&v| comb2(v)).sum();
    let comb_n = comb2(n);

    let expected = sum_comb_ai * sum_comb_bj / comb_n;
    let max_index = (sum_comb_ai + sum_comb_bj) / 2.0;

    if (max_index - expected).abs() < 1e-12 { 0.0 }
    else { (sum_comb_nij - expected) / (max_index - expected) }
}

fn extract_labels<'a>(result: &[Submatrix<'a, f64>], n_docs: usize, k: usize) -> Vec<usize> {
    if result.is_empty() { return vec![0; n_docs]; }

    let n_coclusters = result.len();
    let mut membership = Array2::<f64>::zeros((n_docs, n_coclusters));
    for (cid, submatrix) in result.iter().enumerate() {
        for &row_idx in &submatrix.row_indices {
            if row_idx < n_docs { membership[[row_idx, cid]] = 1.0; }
        }
    }

    let dataset = DatasetBase::from(membership);
    let model = KMeans::params(k)
        .max_n_iterations(200)
        .fit(&dataset)
        .expect("K-means on membership vectors failed");
    let predictions = model.predict(dataset);
    predictions.targets.to_vec()
}

fn merge_strategies() -> Vec<(String, HierarchicalMergeConfig)> {
    vec![
        ("Adaptive".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Adaptive,
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Union".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Union,
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Intersection(0.3)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Intersection { overlap_threshold: 0.3 },
            merge_threshold: 0.3,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Intersection(0.5)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Intersection { overlap_threshold: 0.5 },
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Weighted(0.7/0.3)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Weighted { left_weight: 0.7, right_weight: 0.3 },
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
        ("Weighted(0.5/0.5)".to_string(), HierarchicalMergeConfig {
            merge_strategy: MergeStrategy::Weighted { left_weight: 0.5, right_weight: 0.5 },
            merge_threshold: 0.5,
            rescore_merged: true,
            parallel_level: 10,
        }),
    ]
}

fn mean(vals: &[f64]) -> f64 {
    if vals.is_empty() { return 0.0; }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_dev(vals: &[f64]) -> f64 {
    if vals.len() < 2 { return 0.0; }
    let m = mean(vals);
    let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    var.sqrt()
}

struct SeedResult {
    nmi: f64,
    ari: f64,
    time_s: f64,
    coclusters: usize,
}

fn run_strategy_seeds(
    matrix: &Matrix<f64>,
    true_labels: &[usize],
    k: usize,
    m_blocks: usize,
    n_blocks: usize,
    t_p: usize,
    num_threads: usize,
    config: &HierarchicalMergeConfig,
) -> Result<Vec<SeedResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::with_capacity(NUM_SEEDS as usize);

    for seed in 0..NUM_SEEDS {
        let start = Instant::now();
        let local_clusterer = ClustererAdapter::new(SVDClusterer::new(k, 0.1));

        let clusterer = DiMergeCoClusterer::new(
            k, matrix.rows, 0.05, local_clusterer, config.clone(),
            num_threads, t_p, m_blocks, n_blocks,
        )?.with_base_seed(seed);

        let result = clusterer.run(matrix)?;
        let time_s = start.elapsed().as_secs_f64();

        let pred_labels = extract_labels(&result.submatrices, matrix.rows, k);
        let nmi = calculate_nmi(true_labels, &pred_labels);
        let ari = calculate_ari(true_labels, &pred_labels);

        results.push(SeedResult {
            nmi,
            ari,
            time_s,
            coclusters: result.submatrices.len(),
        });
    }

    Ok(results)
}

fn run_ablation(
    name: &str,
    matrix: &Matrix<f64>,
    true_labels: &[usize],
    k: usize,
    m_blocks: usize,
    n_blocks: usize,
    t_p: usize,
    num_threads: usize,
) -> Result<Value, Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(80));
    println!("ABLATION -- {} ({}x{}, T_p={}, threads={}, {} seeds)",
        name, m_blocks, n_blocks, t_p, num_threads, NUM_SEEDS);
    println!("{}", "=".repeat(80));

    // Baseline: SVD on full matrix (deterministic, single run)
    let baseline_json = {
        let start = Instant::now();
        let clusterer = SVDClusterer::new(k, 0.1);
        let submatrices = clusterer.cluster(matrix)?;
        let runtime = start.elapsed().as_secs_f64();

        let mut labels = vec![0usize; matrix.rows];
        for (cid, sub) in submatrices.iter().enumerate() {
            for &r in &sub.row_indices {
                if r < matrix.rows { labels[r] = cid; }
            }
        }
        let nmi = calculate_nmi(true_labels, &labels);
        let ari = calculate_ari(true_labels, &labels);
        println!("\n  Baseline SCC:  NMI={:.4}  ARI={:.4}  Time={:.3}s", nmi, ari, runtime);

        json!({ "nmi": (nmi * 10000.0).round() / 10000.0,
                "ari": (ari * 10000.0).round() / 10000.0,
                "time_s": (runtime * 1000.0).round() / 1000.0 })
    };

    println!("\n  {:<22} {:>12} {:>12} {:>12} {:>12}",
        "Strategy", "NMI", "ARI", "CoClusters", "Time(s)");
    println!("  {}", "-".repeat(72));

    let mut strategies_json = serde_json::Map::new();

    for (strategy_name, config) in merge_strategies() {
        let seed_results = run_strategy_seeds(
            matrix, true_labels, k, m_blocks, n_blocks, t_p, num_threads, &config,
        )?;

        let nmis: Vec<f64> = seed_results.iter().map(|r| r.nmi).collect();
        let aris: Vec<f64> = seed_results.iter().map(|r| r.ari).collect();
        let times: Vec<f64> = seed_results.iter().map(|r| r.time_s).collect();
        let coclusters: Vec<usize> = seed_results.iter().map(|r| r.coclusters).collect();

        let nmi_mean = mean(&nmis);
        let nmi_std = std_dev(&nmis);
        let ari_mean = mean(&aris);
        let ari_std = std_dev(&aris);
        let time_mean = mean(&times);
        let time_std = std_dev(&times);
        let cc_mean = mean(&coclusters.iter().map(|&c| c as f64).collect::<Vec<_>>());

        println!("  {:<22} {:>5.4}+/-{:<5.4} {:>5.4}+/-{:<5.4} {:>10.1} {:>5.3}+/-{:<5.3}",
            strategy_name, nmi_mean, nmi_std, ari_mean, ari_std, cc_mean, time_mean, time_std);

        let r4 = |v: f64| (v * 10000.0).round() / 10000.0;
        let r3 = |v: f64| (v * 1000.0).round() / 1000.0;

        strategies_json.insert(strategy_name, json!({
            "nmi_mean": r4(nmi_mean),
            "nmi_std": r4(nmi_std),
            "ari_mean": r4(ari_mean),
            "ari_std": r4(ari_std),
            "time_mean_s": r3(time_mean),
            "time_std_s": r3(time_std),
            "coclusters_mean": (cc_mean * 10.0).round() / 10.0,
            "n_seeds": NUM_SEEDS,
            "raw_nmi": nmis.iter().map(|&v| r4(v)).collect::<Vec<_>>(),
            "raw_ari": aris.iter().map(|&v| r4(v)).collect::<Vec<_>>(),
            "raw_time_s": times.iter().map(|&v| r3(v)).collect::<Vec<_>>(),
        }));
    }

    Ok(json!({
        "dataset": name,
        "m_blocks": m_blocks,
        "n_blocks": n_blocks,
        "t_p": t_p,
        "threads": num_threads,
        "n_seeds": NUM_SEEDS,
        "baseline_scc": baseline_json,
        "strategies": strategies_json,
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .format_timestamp_millis()
        .init();

    println!("\n{}", "=".repeat(80));
    println!("DiMergeCo Merge Strategy Ablation Study ({} seeds)", NUM_SEEDS);
    println!("{}", "=".repeat(80));

    let num_threads = num_cpus::get().min(16);
    let mut all_results = serde_json::Map::new();

    // Classic4
    {
        let (matrix, labels, k) = load_classic4()?;
        let r = run_ablation("Classic4", &matrix, &labels, k, 2, 2, 10, num_threads)?;
        all_results.insert("classic4_tp10".to_string(), r);
    }

    // BCW
    {
        let (matrix, labels, k) = load_bcw()?;
        let r = run_ablation("BCW", &matrix, &labels, k, 2, 2, 10, num_threads)?;
        all_results.insert("bcw_tp10".to_string(), r);
    }

    let output = json!({
        "timestamp": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        "tool": "Rust DiMergeCo ablation_merge_strategy (10 seeds)",
        "datasets": all_results,
    });

    let output_path = "baselines/results/rust_merge_ablation_10seeds.json";
    std::fs::write(output_path, serde_json::to_string_pretty(&output)?)?;
    println!("\nResults saved to {}", output_path);

    println!("\n{}", "=".repeat(80));
    println!("Ablation complete.");
    println!("{}", "=".repeat(80));

    Ok(())
}
