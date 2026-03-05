//! Quick FNMF test on RCV1-train
use fast_cocluster::atom::fnmf::FnmfClusterer;
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
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

fn extract_labels(submatrices: &[fast_cocluster::submatrix::Submatrix<'_, f64>], n_rows: usize, k: usize) -> Vec<usize> {
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
    let dataset = linfa::DatasetBase::from(membership);
    let model = KMeans::params(k).max_n_iterations(300).fit(&dataset).expect("K-means failed");
    model.predict(dataset).targets.to_vec()
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    
    // Load RCV1-train
    let array: Array2<f64> = ndarray_npy::read_npy("data/rcv1/rcv1_train.npy").expect("load data");
    let labels_array: ndarray::Array1<i64> = ndarray_npy::read_npy("data/rcv1/rcv1_train_labels.npy").expect("load labels");
    let true_labels: Vec<usize> = labels_array.iter().map(|&x| x as usize).collect();
    let (rows, cols) = (array.nrows(), array.ncols());
    println!("RCV1-train: {} x {}", rows, cols);
    
    let matrix = Matrix::new(array.clone());
    let k = 4;
    
    // Test DiMergeCo + FNMF
    println!("\n--- DiMergeCo + FNMF ---");
    let start = Instant::now();
    let local = FnmfClusterer::new(k, 50);
    match DiMergeCoClusterer::new(k, rows, 0.05, local, HierarchicalMergeConfig::default(), 16, 10, 8, 8) {
        Ok(c) => match c.run(&matrix) {
            Ok(result) => {
                let runtime = start.elapsed().as_secs_f64();
                let pred = extract_labels(&result.submatrices, rows, k);
                let nmi = calculate_nmi(&true_labels, &pred);
                println!("NMI={:.4}, Clusters={}, Time={:.1}s", nmi, result.submatrices.len(), runtime);
            }
            Err(e) => println!("ERROR: {}", e),
        },
        Err(e) => println!("ERROR: {}", e),
    }
    
    // Also test standalone FNMF on the full matrix for comparison
    println!("\n--- Standalone FNMF (full matrix) ---");
    let start = Instant::now();
    let clusterer = FnmfClusterer::new(k, 50);
    use fast_cocluster::dimerge_co::parallel_coclusterer::LocalClusterer;
    match clusterer.cluster_local(&array) {
        Ok(subs) => {
            let runtime = start.elapsed().as_secs_f64();
            let pred = extract_labels(&subs, rows, k);
            let nmi = calculate_nmi(&true_labels, &pred);
            println!("NMI={:.4}, Clusters={}, Time={:.1}s", nmi, subs.len(), runtime);
        }
        Err(e) => println!("ERROR: {}", e),
    }
    
    println!("\nPython baseline: NMI=0.3170, Time=12.1s");
}
