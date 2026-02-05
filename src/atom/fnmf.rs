use std::error::Error;

use ndarray::{Array2, Axis};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::dimerge_co::parallel_coclusterer::LocalClusterer;
use crate::submatrix::Submatrix;

use super::nnls::nnlsm_blockpivot;
use super::tri_factor_base::build_submatrices_from_labels;

/// FNMF (Kim & Park 2011) — ANLS with block principal pivoting
///
/// Binary factorization: X ≈ W * H^T (not tri-factor).
/// Uses alternating NNLS to solve for H given W, then W given H.
pub struct FnmfClusterer {
    pub n_row_clusters: usize,
    pub n_col_clusters: usize,
    pub max_iter: usize,
    pub n_init: usize,
    pub seed: Option<u64>,
}

impl FnmfClusterer {
    pub fn new(n_row_clusters: usize, n_col_clusters: usize) -> Self {
        Self {
            n_row_clusters,
            n_col_clusters,
            max_iter: 50,
            n_init: 1,
            seed: None,
        }
    }

    /// Run a single ANLS-BPP factorization
    fn fit_single(&self, x: &Array2<f64>, seed: u64) -> (Array2<f64>, Array2<f64>, f64) {
        let m = x.nrows();
        let n = x.ncols();
        // For co-clustering, use max(n_row_clusters, n_col_clusters) as rank
        let k = self.n_row_clusters.max(self.n_col_clusters);

        let mut rng = StdRng::seed_from_u64(seed);
        let mut w = Array2::random_using((m, k), Uniform::new(0.0, 1.0), &mut rng);
        let mut h = Array2::random_using((n, k), Uniform::new(0.0, 1.0), &mut rng);

        for _iter in 0..self.max_iter {
            // Solve for H: min ||W * H^T - X||  → min ||W * Z - X|| where Z = H^T
            // NNLS: min ||W * Z - X|| s.t. Z >= 0, solved column by column of X
            // nnlsm_blockpivot(W, X) → Z (k x n), then H = Z^T
            let result_h = nnlsm_blockpivot(&w, x, Some(&h.t().to_owned()));
            h = result_h.x.t().to_owned();

            // Solve for W: min ||H * W^T - X^T||
            let result_w = nnlsm_blockpivot(&h, &x.t().to_owned(), Some(&w.t().to_owned()));
            w = result_w.x.t().to_owned();
        }

        // Normalize column pairs
        let (w_norm, h_norm) = normalize_column_pair(&w, &h);

        // Compute relative error
        let approx = w_norm.dot(&h_norm.t());
        let diff = x - &approx;
        let error = diff.mapv(|v| v * v).sum().sqrt();
        let x_norm = x.mapv(|v| v * v).sum().sqrt();
        let rel_error = if x_norm > 0.0 { error / x_norm } else { error };

        (w_norm, h_norm, rel_error)
    }
}

/// Normalize column pairs so that each column of W has unit norm
fn normalize_column_pair(w: &Array2<f64>, h: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let k = w.ncols();
    let mut w_out = w.clone();
    let mut h_out = h.clone();

    for j in 0..k {
        let norm = w_out.column(j).mapv(|v| v * v).sum().sqrt();
        if norm > 1e-30 {
            w_out.column_mut(j).mapv_inplace(|v| v / norm);
            h_out.column_mut(j).mapv_inplace(|v| v * norm);
        }
    }

    (w_out, h_out)
}

/// Compute argmax along axis 1
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

impl LocalClusterer for FnmfClusterer {
    fn cluster_local<'a>(
        &self,
        matrix: &'a Array2<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
        let base_seed = self.seed.unwrap_or(42);
        let mut best_w = None;
        let mut best_h = None;
        let mut best_error = f64::INFINITY;

        for init_idx in 0..self.n_init {
            let seed = base_seed.wrapping_add(init_idx as u64);
            let (w, h, error) = self.fit_single(matrix, seed);
            if error < best_error {
                best_error = error;
                best_w = Some(w);
                best_h = Some(h);
            }
        }

        let w = best_w.unwrap();
        let h = best_h.unwrap();

        // Row labels from W, col labels from H
        // For co-clustering, we need separate row and col cluster counts
        // Use first n_row_clusters columns of W for row labels
        // and first n_col_clusters columns of H for col labels
        let row_labels = argmax_axis1(&w.slice(ndarray::s![.., ..self.n_row_clusters]).to_owned());
        let col_labels = argmax_axis1(&h.slice(ndarray::s![.., ..self.n_col_clusters]).to_owned());

        Ok(build_submatrices_from_labels(
            matrix,
            &row_labels,
            &col_labels,
            self.n_row_clusters,
            self.n_col_clusters,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::{check_block_labels, make_block_diagonal};

    #[test]
    fn test_fnmf_block_diagonal() {
        let x = make_block_diagonal();
        let clusterer = FnmfClusterer {
            n_row_clusters: 2,
            n_col_clusters: 2,
            max_iter: 50,
            n_init: 3,
            seed: Some(42),
        };
        let (w, h, error) = clusterer.fit_single(&x, 42);
        let row_labels = argmax_axis1(&w);
        let col_labels = argmax_axis1(&h);
        assert!(
            check_block_labels(&row_labels, 10),
            "FNMF row labels should separate the two blocks: {:?}",
            row_labels
        );
        assert!(
            check_block_labels(&col_labels, 10),
            "FNMF col labels should separate the two blocks: {:?}",
            col_labels
        );
        assert!(error < 0.5, "FNMF relative error should be low: {}", error);
    }
}
