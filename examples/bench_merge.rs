// Quick benchmark for merge optimization
// Compare time before/after parallel merge

use fast_cocluster::atom::nbvd::NbvdClusterer;
use fast_cocluster::atom::tri_factor_base::TriFactorConfig;
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use ndarray::Array2;
use std::time::Instant;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Loading RCV1-all...");
    let array: Array2<f64> = ndarray_npy::read_npy("data/rcv1/rcv1_all.npy").expect("load data");
    let (rows, cols) = (array.nrows(), array.ncols());
    println!("RCV1-all: {} x {}", rows, cols);

    let matrix = Matrix::new(array);
    let k = 4;
    let config = TriFactorConfig {
        n_row_clusters: k,
        n_col_clusters: k,
        max_iter: 20,
        inner_iter: 10,
        n_init: 1,
        tol: 1e-9,
        seed: None,
        timeout_secs: None,
    };

    // Test with optimized config: T_p=20, 6x6
    println!("\n=== Benchmark: T_p=20, 6x6 (with parallel merge optimization) ===");
    
    let start = Instant::now();
    let local = NbvdClusterer::with_config(config);
    let clusterer = DiMergeCoClusterer::new(
        k, rows, 0.05, local,
        HierarchicalMergeConfig::default(),
        16,  // threads
        20,  // T_p
        6, 6 // blocks
    ).expect("init");
    
    let result = clusterer.run(&matrix).expect("run");
    let total_time = start.elapsed().as_secs_f64();
    
    println!("\nResults:");
    println!("  Total time: {:.1}s", total_time);
    println!("  Clusters: {}", result.submatrices.len());
    println!("\nDone.");
}
