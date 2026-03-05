//! Full MovieLens Benchmark (162K × 59K)
//! Tests all methods: Standalone vs DiMergeCo
//! Focus: Speed comparison while maintaining quality
//!
//! Usage: RAYON_NUM_THREADS=72 cargo run --release --example benchmark_full_movielens

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
use std::time::Instant;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("======================================================================");
    println!("MovieLens Full Benchmark (162K × 59K)");
    println!("All Methods: Standalone vs DiMergeCo");
    println!("======================================================================\n");

    // Load dense matrix
    println!("Loading MovieLens Full (this may take a while)...");
    let start = Instant::now();
    let file = File::open("data/movielens_full_dense.npy").expect("Cannot open movielens_full_dense.npy");
    let matrix_f32: Array2<f32> = Array2::read_npy(file).expect("Cannot read npy");
    let array = matrix_f32.mapv(|x| x as f64);
    println!("  Loaded in {:.1}s", start.elapsed().as_secs_f64());

    let (rows, cols) = (array.nrows(), array.ncols());
    let nnz = array.iter().filter(|&&x| x > 0.0).count();
    let density = nnz as f64 / (rows * cols) as f64;
    println!("  Shape: {} × {}", rows, cols);
    println!("  Non-zeros: {} ({:.4}% density)", nnz, density * 100.0);
    println!("  Memory: {:.2} GB\n", (rows * cols * 8) as f64 / 1e9);

    let matrix = Matrix::new(array.clone());

    // Parameters
    let k = 20;  // number of clusters
    let num_threads = 72;  // half of 144 cores
    let tp = 15;  // T_p iterations
    let (m_blocks, n_blocks) = (8, 8);  // 64 partitions per iteration

    println!("Configuration:");
    println!("  Clusters (k): {}", k);
    println!("  Threads: {}", num_threads);
    println!("  DiMergeCo: T_p={}, {}×{} grid\n", tp, m_blocks, n_blocks);

    println!("{:<12} {:>12} {:>12} {:>10} {:>10} {:>12}",
             "Method", "Standalone", "DiMergeCo", "Speedup", "Clusters", "Quality");
    println!("{}", "-".repeat(70));

    // Spectral (SVD-based) - fastest baseline
    run_comparison(
        "Spectral",
        || ClustererAdapter::new(SVDClusterer::new(k, 0.1)),
        || ClustererAdapter::new(SVDClusterer::new(k, 0.1)),
        &array, &matrix, rows, k, num_threads, tp, m_blocks, n_blocks,
    );

    // NBVD
    run_comparison(
        "NBVD",
        || NbvdClusterer::with_config(make_config(k)),
        || NbvdClusterer::with_config(make_config(k)),
        &array, &matrix, rows, k, num_threads, tp, m_blocks, n_blocks,
    );

    // ONM3F
    run_comparison(
        "ONM3F",
        || Onm3fClusterer::with_config(make_config(k)),
        || Onm3fClusterer::with_config(make_config(k)),
        &array, &matrix, rows, k, num_threads, tp, m_blocks, n_blocks,
    );

    // ONMTF
    run_comparison(
        "ONMTF",
        || OnmtfClusterer::with_config(make_config(k)),
        || OnmtfClusterer::with_config(make_config(k)),
        &array, &matrix, rows, k, num_threads, tp, m_blocks, n_blocks,
    );

    // PNMTF
    run_comparison(
        "PNMTF",
        || PnmtfClusterer::with_config(make_config(k), 0.1, 0.1, 0.1),
        || PnmtfClusterer::with_config(make_config(k), 0.1, 0.1, 0.1),
        &array, &matrix, rows, k, num_threads, tp, m_blocks, n_blocks,
    );

    // FNMF
    run_comparison(
        "FNMF",
        || FnmfClusterer::with_clusters(k, k, 10),
        || FnmfClusterer::with_clusters(k, k, 10),
        &array, &matrix, rows, k, num_threads, tp, m_blocks, n_blocks,
    );

    println!("{}", "-".repeat(70));
    println!("\n✓ Benchmark complete");
    println!("  Speedup > 1.0 means DiMergeCo is faster");
    println!("  Quality = coverage ratio (rows covered / total rows)");
}

fn make_config(k: usize) -> TriFactorConfig {
    TriFactorConfig {
        n_row_clusters: k,
        n_col_clusters: k,
        max_iter: 10,  // reduced from default (300) since full MovieLens is very large; 10 iters suffice for convergence on subblocks
        n_init: 3,
        tol: 1e-6,
        seed: Some(42),
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
    // Standalone
    print!("{:<12} ", name);
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let start = Instant::now();
    let clusterer = standalone_factory();
    let (standalone_time, standalone_coverage) = match clusterer.cluster_local(array) {
        Ok(subs) => {
            let cov = compute_coverage(&subs, rows);
            (start.elapsed().as_secs_f64(), cov)
        }
        Err(e) => {
            println!("{:>12} (FAIL: {})", "---", e);
            return;
        }
    };

    // DiMergeCo
    let start = Instant::now();
    let local = dimerge_factory();
    let (dimerge_time, dimerge_clusters, dimerge_coverage) = match DiMergeCoClusterer::new(
        k, rows, 0.05, local, HierarchicalMergeConfig::default(),
        num_threads, tp, m_blocks, n_blocks,
    ) {
        Ok(c) => match c.run(matrix) {
            Ok(result) => {
                let cov = compute_coverage(&result.submatrices, rows);
                (start.elapsed().as_secs_f64(), result.submatrices.len(), cov)
            }
            Err(e) => {
                println!("{:>10.1}s {:>12} (FAIL: {})", standalone_time, "---", e);
                return;
            }
        },
        Err(e) => {
            println!("{:>10.1}s {:>12} (FAIL: {})", standalone_time, "---", e);
            return;
        }
    };

    let speedup = standalone_time / dimerge_time;
    let quality_ratio = dimerge_coverage / standalone_coverage.max(0.001);

    println!(
        "{:>10.1}s {:>10.1}s {:>9.2}x {:>10} {:>10.1}%",
        standalone_time, dimerge_time, speedup, dimerge_clusters, quality_ratio * 100.0
    );
}

fn compute_coverage(submatrices: &[Submatrix<'_, f64>], total_rows: usize) -> f64 {
    let mut covered = vec![false; total_rows];
    for sm in submatrices {
        for &idx in &sm.row_indices {
            if idx < total_rows {
                covered[idx] = true;
            }
        }
    }
    covered.iter().filter(|&&x| x).count() as f64 / total_rows as f64
}
