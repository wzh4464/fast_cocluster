//! DiMergeCo Advantage Benchmark
//! Shows DiMergeCo parallelizes inherently sequential NMF algorithms.
//!
//! OPENBLAS_NUM_THREADS=1: BLAS stays single-threaded
//! Standalone = pure sequential (cannot use multi-core)
//! DiMergeCo  = partition-level parallelism via Rayon
//!
//! Multi-scale: tests 10K×5K, 20K×10K, 50K×20K to show scaling advantage
//!
//! Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RUST_LOG=info \
//!        cargo run --release --example benchmark_dimerge_advantage

use fast_cocluster::atom::{
    fnmf::FnmfClusterer,
    nbvd::NbvdClusterer,
    onm3f::Onm3fClusterer,
    onmtf::OnmtfClusterer,
    pnmtf::PnmtfClusterer,
    tri_factor_base::TriFactorConfig,
};
use fast_cocluster::dimerge_co::parallel_coclusterer::LocalClusterer;
use fast_cocluster::dimerge_co::*;
use fast_cocluster::matrix::Matrix;
use fast_cocluster::pipeline::SVDClusterer;
use fast_cocluster::submatrix::Submatrix;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::io::Write as IoWrite;
use std::time::Instant;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("======================================================================");
    println!("DiMergeCo Advantage Benchmark");
    println!("BLAS: single-threaded | DiMergeCo Rayon: multi-threaded");
    println!("======================================================================\n");

    // Load full matrix, then slice to different sizes
    println!("Loading MovieLens Full...");
    let start = Instant::now();
    let file = File::open("data/movielens_full_dense.npy")
        .expect("Cannot open movielens_full_dense.npy");
    let matrix_f32: Array2<f32> = Array2::read_npy(file).expect("Cannot read npy");
    let full_array = matrix_f32.mapv(|x| x as f64);
    println!("  Loaded in {:.1}s ({} x {})\n", start.elapsed().as_secs_f64(),
             full_array.nrows(), full_array.ncols());

    let k = 10;
    let max_threads = 72;

    // Matrix sizes to test (rows, cols)
    let sizes: Vec<(usize, usize, &str)> = vec![
        (10_000, 5_000,  "10K×5K"),
        (20_000, 10_000, "20K×10K"),
        (50_000, 20_000, "50K×20K"),
    ];

    // ============================================================
    // PART 1: Multi-Scale Speed Comparison (All Methods)
    // ============================================================
    println!("######################################################################");
    println!("PART 1: Multi-Scale Speed Comparison");
    println!("  Standalone: sequential NMF (BLAS single-threaded, no parallelism)");
    println!("  DiMergeCo:  {} Rayon threads (partition-level parallelism)", max_threads);
    println!("######################################################################\n");

    for &(nrows, ncols, label) in &sizes {
        let array = full_array.slice(ndarray::s![..nrows, ..ncols]).to_owned();
        let matrix = Matrix::new(array.clone());
        let nnz = array.iter().filter(|&&x| x > 0.0).count();
        let density = nnz as f64 / (nrows * ncols) as f64;

        // Choose grid and T_p based on matrix size
        let (tp, m_blocks, n_blocks) = grid_config(nrows, ncols);
        let total_partitions = tp * m_blocks * n_blocks;

        println!("======================================================================");
        println!("Matrix: {} ({} x {})", label, nrows, ncols);
        println!("  Non-zeros: {} ({:.2}% density)", nnz, density * 100.0);
        println!("  Memory: {:.2} GB", (nrows * ncols * 8) as f64 / 1e9);
        println!("  DiMergeCo: T_p={}, {}x{} grid, {} total partition runs",
                 tp, m_blocks, n_blocks, total_partitions);
        println!("======================================================================");

        println!("{:<12} {:>12} {:>12} {:>10} {:>10} {:>10}",
                 "Method", "Standalone", "DiMergeCo", "Speedup", "Clusters", "Cov%");
        println!("{}", "-".repeat(68));

        // NBVD
        run_comparison("NBVD",
            || NbvdClusterer::with_config(make_config(k)),
            || NbvdClusterer::with_config(make_config(k)),
            &array, &matrix, nrows, k, max_threads, tp, m_blocks, n_blocks);

        // ONM3F
        run_comparison("ONM3F",
            || Onm3fClusterer::with_config(make_config(k)),
            || Onm3fClusterer::with_config(make_config(k)),
            &array, &matrix, nrows, k, max_threads, tp, m_blocks, n_blocks);

        // ONMTF
        run_comparison("ONMTF",
            || OnmtfClusterer::with_config(make_config(k)),
            || OnmtfClusterer::with_config(make_config(k)),
            &array, &matrix, nrows, k, max_threads, tp, m_blocks, n_blocks);

        // PNMTF
        run_comparison("PNMTF",
            || PnmtfClusterer::with_config(make_config(k), 0.1, 0.1, 0.1),
            || PnmtfClusterer::with_config(make_config(k), 0.1, 0.1, 0.1),
            &array, &matrix, nrows, k, max_threads, tp, m_blocks, n_blocks);

        // FNMF
        run_comparison("FNMF",
            || FnmfClusterer::new(k, k),
            || FnmfClusterer::new(k, k),
            &array, &matrix, nrows, k, max_threads, tp, m_blocks, n_blocks);

        // Spectral
        run_comparison("Spectral",
            || ClustererAdapter::new(SVDClusterer::new(k, 0.1)),
            || ClustererAdapter::new(SVDClusterer::new(k, 0.1)),
            &array, &matrix, nrows, k, max_threads, tp, m_blocks, n_blocks);

        println!("{}\n", "-".repeat(68));
    }

    // ============================================================
    // PART 2: Thread Scaling (NBVD, 50K×20K)
    // ============================================================
    println!("######################################################################");
    println!("PART 2: Thread Scaling (NBVD, 50K x 20K)");
    println!("######################################################################\n");

    let (nrows, ncols) = (50_000, 20_000);
    let array = full_array.slice(ndarray::s![..nrows, ..ncols]).to_owned();
    let matrix = Matrix::new(array.clone());
    let (tp, m_blocks, n_blocks) = grid_config(nrows, ncols);

    // Standalone baseline
    println!("Standalone baseline (single-threaded BLAS)...");
    let start = Instant::now();
    let clusterer = NbvdClusterer::with_config(make_config(k));
    let standalone_time = match clusterer.cluster_local(&array) {
        Ok(_) => start.elapsed().as_secs_f64(),
        Err(e) => { println!("  FAIL: {}", e); return; }
    };
    println!("  Standalone: {:.1}s\n", standalone_time);

    let thread_counts = [1, 2, 4, 8, 16, 36, 72];
    println!("{:>8} {:>12} {:>14} {:>10}",
             "Threads", "DiMergeCo", "vs-Standalone", "Parallel-Eff");
    println!("{}", "-".repeat(48));

    let mut single_thread_time: Option<f64> = None;

    for &threads in &thread_counts {
        print!("{:>8} ", threads);
        std::io::stdout().flush().unwrap();

        let start = Instant::now();
        let local = NbvdClusterer::with_config(make_config(k));
        match DiMergeCoClusterer::new(
            k, nrows, 0.05, local, HierarchicalMergeConfig::default(),
            threads, tp, m_blocks, n_blocks,
        ) {
            Ok(c) => match c.run(&matrix) {
                Ok(_) => {
                    let t = start.elapsed().as_secs_f64();
                    let vs_standalone = standalone_time / t;
                    if single_thread_time.is_none() {
                        single_thread_time = Some(t);
                    }
                    let parallel_eff = match single_thread_time {
                        Some(t1) => (t1 / t) / threads as f64 * 100.0,
                        None => 100.0,
                    };
                    println!("{:>10.1}s {:>12.2}x {:>8.1}%",
                             t, vs_standalone, parallel_eff);
                }
                Err(e) => println!("FAIL: {}", e),
            },
            Err(e) => println!("FAIL: {}", e),
        }
    }
    println!("{}", "-".repeat(48));

    // ============================================================
    // PART 3: Config Sensitivity (NBVD, 50K×20K)
    // ============================================================
    println!("\n######################################################################");
    println!("PART 3: Config Sensitivity (NBVD, 50K x 20K, {} threads)", max_threads);
    println!("######################################################################\n");

    println!("Standalone baseline: {:.1}s\n", standalone_time);

    let configs: Vec<(usize, usize, usize)> = vec![
        (1, 2, 2),    //  4 partitions, 4 total
        (1, 4, 4),    // 16 partitions, 16 total
        (1, 8, 8),    // 64 partitions, 64 total
        (3, 2, 2),    //  4 partitions, 12 total
        (3, 4, 4),    // 16 partitions, 48 total
        (3, 8, 8),    // 64 partitions, 192 total
        (5, 4, 4),    // 16 partitions, 80 total
        (5, 8, 8),    // 64 partitions, 320 total
        (10, 4, 4),   // 16 partitions, 160 total
        (10, 8, 8),   // 64 partitions, 640 total
    ];

    println!("{:>5} {:>6} {:>8} {:>12} {:>10} {:>8}",
             "T_p", "Grid", "Runs", "DiMergeCo", "Speedup", "Cov%");
    println!("{}", "-".repeat(54));

    for &(tp_val, mb, nb) in &configs {
        let total_runs = tp_val * mb * nb;
        print!("{:>5} {:>3}x{:<3} {:>6} ", tp_val, mb, nb, total_runs);
        std::io::stdout().flush().unwrap();

        let start = Instant::now();
        let local = NbvdClusterer::with_config(make_config(k));
        match DiMergeCoClusterer::new(
            k, nrows, 0.05, local, HierarchicalMergeConfig::default(),
            max_threads, tp_val, mb, nb,
        ) {
            Ok(c) => match c.run(&matrix) {
                Ok(result) => {
                    let t = start.elapsed().as_secs_f64();
                    let speedup = standalone_time / t;
                    let cov = compute_coverage(&result.submatrices, nrows);
                    println!("{:>10.1}s {:>9.2}x {:>6.1}%", t, speedup, cov * 100.0);
                }
                Err(e) => println!("FAIL: {}", e),
            },
            Err(e) => println!("FAIL: {}", e),
        }
    }
    println!("{}", "-".repeat(54));

    println!("\n======================================================================");
    println!("Benchmark complete!");
    println!("  Speedup > 1.0x = DiMergeCo faster than sequential Standalone");
    println!("  Key: DiMergeCo parallelizes inherently sequential NMF algorithms");
    println!("======================================================================");
}

/// Choose grid config based on matrix size
fn grid_config(nrows: usize, ncols: usize) -> (usize, usize, usize) {
    let n = nrows.max(ncols);
    if n <= 10_000 {
        (3, 2, 2)     // small: 4 partitions, 12 total
    } else if n <= 25_000 {
        (3, 4, 4)     // medium: 16 partitions, 48 total
    } else {
        (5, 4, 4)     // large: 16 partitions, 80 total
    }
}

fn make_config(k: usize) -> TriFactorConfig {
    TriFactorConfig {
        n_row_clusters: k,
        n_col_clusters: k,
        max_iter: 10,
        inner_iter: 10,
        n_init: 1,
        tol: 1e-6,
        seed: Some(42),
        timeout_secs: None,
    }
}

fn run_comparison<L, F1, F2>(
    name: &str,
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
) where
    L: LocalClusterer + 'static,
    F1: FnOnce() -> L,
    F2: FnOnce() -> L,
{
    print!("{:<12} ", name);
    std::io::stdout().flush().unwrap();

    // Standalone
    let start = Instant::now();
    let clusterer = standalone_factory();
    let (standalone_time, _standalone_cov) = match clusterer.cluster_local(array) {
        Ok(subs) => {
            let cov = compute_coverage(&subs, rows);
            (start.elapsed().as_secs_f64(), cov)
        }
        Err(e) => {
            println!("STANDALONE FAIL: {}", e);
            return;
        }
    };

    // DiMergeCo
    let start = Instant::now();
    let local = dimerge_factory();
    match DiMergeCoClusterer::new(
        k, rows, 0.05, local, HierarchicalMergeConfig::default(),
        num_threads, tp, m_blocks, n_blocks,
    ) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let t = start.elapsed().as_secs_f64();
                let speedup = standalone_time / t;
                let cov = compute_coverage(&result.submatrices, rows);
                println!("{:>10.1}s {:>10.1}s {:>9.2}x {:>10} {:>8.1}%",
                         standalone_time, t, speedup, result.submatrices.len(), cov * 100.0);
            }
            Err(e) => println!("{:>10.1}s {:>10} DIMERGE FAIL: {}", standalone_time, "---", e),
        },
        Err(e) => println!("{:>10.1}s {:>10} DIMERGE FAIL: {}", standalone_time, "---", e),
    }
}

fn compute_coverage(submatrices: &[Submatrix<'_, f64>], total_rows: usize) -> f64 {
    let mut covered = vec![false; total_rows];
    for sm in submatrices {
        for &idx in &sm.row_indices {
            if idx < total_rows { covered[idx] = true; }
        }
    }
    covered.iter().filter(|&&x| x).count() as f64 / total_rows as f64
}
