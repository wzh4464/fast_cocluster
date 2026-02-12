use std::error::Error;

use ndarray::{Array2, Axis};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::submatrix::Submatrix;

/// Configuration shared by all tri-factorization methods
#[derive(Debug, Clone)]
pub struct TriFactorConfig {
    pub n_row_clusters: usize,
    pub n_col_clusters: usize,
    pub max_iter: usize,
    pub n_init: usize,
    pub tol: f64,
    pub seed: Option<u64>,
}

impl Default for TriFactorConfig {
    fn default() -> Self {
        Self {
            n_row_clusters: 2,
            n_col_clusters: 2,
            max_iter: 100,
            n_init: 1,
            tol: 1e-9,
            seed: None,
        }
    }
}

/// Result from tri-factorization: X ≈ F * S * G^T
pub struct TriFactorResult {
    pub f: Array2<f64>,
    pub s: Array2<f64>,
    pub g: Array2<f64>,
    pub row_labels: Vec<usize>,
    pub col_labels: Vec<usize>,
}

/// Trait for algorithm-specific update rules in tri-factorization
pub trait TriFactorUpdater: Send + Sync {
    fn update_f(&self, x: &Array2<f64>, f: &mut Array2<f64>, s: &Array2<f64>, g: &Array2<f64>);
    fn update_g(&self, x: &Array2<f64>, f: &Array2<f64>, s: &Array2<f64>, g: &mut Array2<f64>);
    fn update_s(&self, x: &Array2<f64>, f: &Array2<f64>, s: &mut Array2<f64>, g: &Array2<f64>);
    fn compute_criterion(
        &self,
        x: &Array2<f64>,
        f: &Array2<f64>,
        s: &Array2<f64>,
        g: &Array2<f64>,
    ) -> f64;
    fn normalize(
        &self,
        x: &Array2<f64>,
        f: &mut Array2<f64>,
        s: &mut Array2<f64>,
        g: &mut Array2<f64>,
    );
}

/// Run tri-factorization with n_init random restarts.
/// Keeps the result with the best (lowest) criterion value.
///
/// Convergence loop matches the Python NMTFcoclust pattern:
/// - Outer loop: up to max_iter rounds, checks criterion convergence after each
/// - Inner loop: max_iter update steps (F, G, S) without normalization
/// - After inner loop: normalize, compute criterion, check convergence
pub fn run_tri_factorization(
    config: &TriFactorConfig,
    updater: &dyn TriFactorUpdater,
    x: &Array2<f64>,
) -> TriFactorResult {
    let n = x.nrows();
    let m = x.ncols();
    let k = config.n_row_clusters;
    let l = config.n_col_clusters;

    let mut best_criterion = f64::INFINITY;
    let mut best_f = Array2::zeros((n, k));
    let mut best_s = Array2::zeros((k, l));
    let mut best_g = Array2::zeros((m, l));

    let base_seed = config.seed.unwrap_or(42);

    for init_idx in 0..config.n_init {
        let seed = base_seed.wrapping_add(init_idx as u64);
        let mut rng = StdRng::seed_from_u64(seed);

        let mut f = Array2::random_using((n, k), Uniform::new(0.0, 1.0), &mut rng);
        let mut s = Array2::random_using((k, l), Uniform::new(0.0, 1.0), &mut rng);
        let mut g = Array2::random_using((m, l), Uniform::new(0.0, 1.0), &mut rng);

        let mut prev_criterion = f64::INFINITY;

        // Single convergence loop: update F, G, S once per iteration
        // Then normalize and check convergence
        for _iter in 0..config.max_iter {
            // Update F, G, S
            updater.update_f(x, &mut f, &s, &g);
            updater.update_g(x, &f, &s, &mut g);
            updater.update_s(x, &f, &mut s, &g);

            // Normalize
            updater.normalize(x, &mut f, &mut s, &mut g);

            // Check convergence
            let criterion = updater.compute_criterion(x, &f, &s, &g);
            if (prev_criterion - criterion).abs() <= config.tol {
                break;
            }
            prev_criterion = criterion;
        }

        let final_criterion = updater.compute_criterion(x, &f, &s, &g);
        if final_criterion < best_criterion {
            best_criterion = final_criterion;
            best_f = f;
            best_s = s;
            best_g = g;
        }
    }

    let row_labels = argmax_axis1(&best_f);
    let col_labels = argmax_axis1(&best_g);

    TriFactorResult {
        f: best_f,
        s: best_s,
        g: best_g,
        row_labels,
        col_labels,
    }
}

/// Compute argmax along axis 1 for each row
fn argmax_axis1(a: &Array2<f64>) -> Vec<usize> {
    a.axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

/// Build Submatrix instances from row/col cluster labels
pub fn build_submatrices_from_labels<'a>(
    matrix: &'a Array2<f64>,
    row_labels: &[usize],
    col_labels: &[usize],
    n_row_clusters: usize,
    n_col_clusters: usize,
) -> Vec<Submatrix<'a, f64>> {
    let mut submatrices = Vec::new();

    for rc in 0..n_row_clusters {
        for cc in 0..n_col_clusters {
            let rows: Vec<usize> = row_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == rc)
                .map(|(idx, _)| idx)
                .collect();
            let cols: Vec<usize> = col_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == cc)
                .map(|(idx, _)| idx)
                .collect();

            if rows.is_empty() || cols.is_empty() {
                continue;
            }

            if let Some(sub) = Submatrix::from_indices(matrix, &rows, &cols) {
                submatrices.push(sub);
            }
        }
    }

    submatrices
}
