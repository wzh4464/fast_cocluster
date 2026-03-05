//! MovieLens 25M Co-clustering Benchmark
//! Compares Standalone vs DiMergeCo performance on user-movie rating matrix
//!
//! Usage: cargo run --release --example evaluate_movielens

use fast_cocluster::atom::{
    nbvd::NbvdClusterer,
    onm3f::Onm3fClusterer,
    pnmtf::PnmtfClusterer,
    tri_factor_base::TriFactorConfig,
};
use fast_cocluster::dimerge_co::parallel_coclusterer::LocalClusterer;
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::SVDClusterer;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::time::Instant;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("======================================================================");
    println!("MovieLens 25M Co-clustering Benchmark");
    println!("Comparing Standalone vs DiMergeCo");
    println!("======================================================================\n");

    // Test on different dataset sizes
    let datasets = [
        ("small", "data/movielens_small.npy", 10),   // 5K × 3K
        ("medium", "data/movielens_medium.npy", 15), // 10K × 8K
        ("large", "data/movielens_large.npy", 20),   // 20K × 15K
    ];

    for (name, path, k) in datasets {
        println!("\n{}", "=".repeat(70));
        println!("Dataset: {} (k={})", name, k);
        println!("{}", "=".repeat(70));

        let matrix = match load_matrix(path) {
            Ok(m) => m,
            Err(e) => {
                println!("  Skip: {}", e);
                continue;
            }
        };

        let (rows, cols) = (matrix.nrows(), matrix.ncols());
        let nnz = matrix.iter().filter(|&&x| x > 0.0).count();
        let density = nnz as f64 / (rows * cols) as f64;
        println!("  Shape: {} × {}", rows, cols);
        println!("  Non-zeros: {} ({:.2}% density)", nnz, density * 100.0);
        println!("  Memory: {:.1} MB", (rows * cols * 8) as f64 / 1e6);

        run_benchmark(&matrix, k);
    }

    println!("\n======================================================================");
    println!("✓ Benchmark complete");
    println!("======================================================================");
}

fn load_matrix(path: &str) -> Result<Array2<f64>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let matrix: Array2<f32> =
        Array2::read_npy(file).map_err(|e| format!("Cannot read {}: {}", path, e))?;
    Ok(matrix.mapv(|x| x as f64))
}

fn make_config(k: usize) -> TriFactorConfig {
    TriFactorConfig {
        n_row_clusters: k,
        n_col_clusters: k,
        max_iter: 20,
        n_init: 1,
        tol: 1e-9,
        seed: None,
    }
}

fn run_benchmark(array: &Array2<f64>, k: usize) {
    let rows = array.nrows();
    let matrix = Matrix::new(array.clone());
    let num_threads = 16;

    // DiMergeCo config - adaptive based on matrix size
    let tp = 20;
    let (m_blocks, n_blocks) = if rows > 10000 { (6, 6) } else { (4, 4) };

    println!("\n  {:<12} {:>12} {:>12} {:>10} {:>8}",
             "Method", "Standalone", "DiMergeCo", "Speedup", "Clusters");
    println!("  {:-<12} {:-<12} {:-<12} {:-<10} {:-<8}", "", "", "", "", "");

    // Spectral (SVD-based)
    {
        let (standalone_time, dimerge_time, dimerge_clusters) = run_method(
            "spectral",
            || ClustererAdapter::new(SVDClusterer::new(k, 0.1)),
            || ClustererAdapter::new(SVDClusterer::new(k, 0.1)),
            array,
            &matrix,
            rows,
            k,
            num_threads,
            tp,
            m_blocks,
            n_blocks,
        );
        print_result("Spectral", standalone_time, dimerge_time, dimerge_clusters);
    }

    // NBVD
    {
        let (standalone_time, dimerge_time, dimerge_clusters) = run_method(
            "nbvd",
            || NbvdClusterer::with_config(make_config(k)),
            || NbvdClusterer::with_config(make_config(k)),
            array,
            &matrix,
            rows,
            k,
            num_threads,
            tp,
            m_blocks,
            n_blocks,
        );
        print_result("NBVD", standalone_time, dimerge_time, dimerge_clusters);
    }

    // ONM3F
    {
        let (standalone_time, dimerge_time, dimerge_clusters) = run_method(
            "onm3f",
            || Onm3fClusterer::with_config(make_config(k)),
            || Onm3fClusterer::with_config(make_config(k)),
            array,
            &matrix,
            rows,
            k,
            num_threads,
            tp,
            m_blocks,
            n_blocks,
        );
        print_result("ONM3F", standalone_time, dimerge_time, dimerge_clusters);
    }

    // PNMTF
    {
        let (standalone_time, dimerge_time, dimerge_clusters) = run_method(
            "pnmtf",
            || PnmtfClusterer::with_config(make_config(k), 0.1, 0.1, 0.1),
            || PnmtfClusterer::with_config(make_config(k), 0.1, 0.1, 0.1),
            array,
            &matrix,
            rows,
            k,
            num_threads,
            tp,
            m_blocks,
            n_blocks,
        );
        print_result("PNMTF", standalone_time, dimerge_time, dimerge_clusters);
    }
}

fn run_method<L, F1, F2>(
    _name: &str,
    standalone_factory: F1,
    dimerge_factory: F2,
    array: &Array2<f64>,
    matrix: &Matrix<f64>,
    rows: usize,
    k: usize,
    num_threads: usize,
    tp: usize,
    m_blocks: usize,
    n_blocks: usize,
) -> (Option<f64>, Option<f64>, Option<usize>)
where
    L: LocalClusterer + 'static,
    F1: FnOnce() -> L,
    F2: FnOnce() -> L,
{
    // Standalone
    let start = Instant::now();
    let clusterer = standalone_factory();
    let standalone_time = match clusterer.cluster_local(array) {
        Ok(_) => Some(start.elapsed().as_secs_f64()),
        Err(_) => None,
    };

    // DiMergeCo
    let start = Instant::now();
    let local = dimerge_factory();
    let (dimerge_time, dimerge_clusters) = match DiMergeCoClusterer::new(
        k,
        rows,
        0.05,
        local,
        HierarchicalMergeConfig::default(),
        num_threads,
        tp,
        m_blocks,
        n_blocks,
    ) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => (
                Some(start.elapsed().as_secs_f64()),
                Some(result.submatrices.len()),
            ),
            Err(_) => (None, None),
        },
        Err(_) => (None, None),
    };

    (standalone_time, dimerge_time, dimerge_clusters)
}

fn print_result(name: &str, standalone: Option<f64>, dimerge: Option<f64>, clusters: Option<usize>) {
    let standalone_str = standalone
        .map(|t| format!("{:.2}s", t))
        .unwrap_or_else(|| "FAIL".to_string());
    let dimerge_str = dimerge
        .map(|t| format!("{:.2}s", t))
        .unwrap_or_else(|| "FAIL".to_string());
    let speedup_str = match (standalone, dimerge) {
        (Some(s), Some(d)) if d > 0.0 => format!("{:.2}x", s / d),
        _ => "-".to_string(),
    };
    let cluster_str = clusters
        .map(|c| format!("{}", c))
        .unwrap_or_else(|| "-".to_string());

    println!(
        "  {:<12} {:>12} {:>12} {:>10} {:>8}",
        name, standalone_str, dimerge_str, speedup_str, cluster_str
    );
}
