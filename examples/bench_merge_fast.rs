/// Quick benchmark for merge optimization
/// Uses smaller config for faster comparison

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
        n_init: 3,
        tol: 1e-9,
        seed: None,
    };

    // Fast config: T_p=5, 4x4 (5 * 16 = 80 partitions, much faster)
    println!("\n=== Benchmark: T_p=5, 4x4 (with parallel merge optimization) ===");
    println!("Expected partitions: 5 * 16 = 80");

    let start = Instant::now();
    let local = NbvdClusterer::with_config(config);
    let clusterer = DiMergeCoClusterer::new(
        k, rows, 0.05, local,
        HierarchicalMergeConfig::default(),
        16,  // threads
        5,   // T_p (small for quick test)
        4, 4 // blocks
    ).expect("init");

    let result = clusterer.run(&matrix).expect("run");
    let total_time = start.elapsed().as_secs_f64();

    println!("\nResults:");
    println!("  Total time: {:.1}s", total_time);
    println!("  Clusters: {}", result.submatrices.len());

    // Reference from sweep (T_p=5, 4x4 = 2129.1s before optimization)
    println!("\nReference (pre-optimization): ~2129s");
    println!("Speedup: {:.2}x", 2129.0 / total_time);

    println!("\nDone.");
}
