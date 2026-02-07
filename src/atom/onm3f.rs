use std::error::Error;

use ndarray::Array2;

use crate::dimerge_co::parallel_coclusterer::LocalClusterer;
use crate::submatrix::Submatrix;

use super::normalization::normalize_standard;
use super::tri_factor_base::{
    build_submatrices_from_labels, run_tri_factorization, TriFactorConfig, TriFactorUpdater,
};
use super::update_rules::{reconstruction_error, sqrt_multiplicative_update};

/// ONM3F (Ding 2006) — Sqrt damping + orthogonality-encouraging denominators
pub struct Onm3fUpdater;

impl TriFactorUpdater for Onm3fUpdater {
    fn update_f(&self, x: &Array2<f64>, f: &mut Array2<f64>, s: &Array2<f64>, g: &Array2<f64>) {
        // F ← F * ((X*G*S^T) / (F*F^T*X*G*S^T))^0.5
        // Optimized: F*F^T*numer = F*(F^T*numer) to avoid m×m intermediate
        let xgs = x.dot(g).dot(&s.t()); // m×k
        let ft_xgs = f.t().dot(&xgs); // k×k
        let denom = f.dot(&ft_xgs); // m×k (no m×m intermediate!)
        *f = sqrt_multiplicative_update(f, &xgs, &denom, 1e-10);
    }

    fn update_g(&self, x: &Array2<f64>, f: &Array2<f64>, s: &Array2<f64>, g: &mut Array2<f64>) {
        // G ← G * ((X^T*F*S) / (G*G^T*X^T*F*S))^0.5
        // Optimized: G*G^T*numer = G*(G^T*numer) to avoid n×n intermediate
        let xtfs = x.t().dot(f).dot(s); // n×k
        let gt_xtfs = g.t().dot(&xtfs); // k×k
        let denom = g.dot(&gt_xtfs); // n×k (no n×n intermediate!)
        *g = sqrt_multiplicative_update(g, &xtfs, &denom, 1e-10);
    }

    fn update_s(&self, x: &Array2<f64>, f: &Array2<f64>, s: &mut Array2<f64>, g: &Array2<f64>) {
        // S ← S * ((F^T*X*G) / (F^T*F*S*G^T*G))^0.5
        let ftf = f.t().dot(f); // k×k
        let gtg = g.t().dot(g); // k×k
        let numer = f.t().dot(x).dot(g); // k×k
        let denom = ftf.dot(&*s).dot(&gtg); // k×k
        *s = sqrt_multiplicative_update(s, &numer, &denom, 1e-10);
    }

    fn compute_criterion(
        &self,
        x: &Array2<f64>,
        f: &Array2<f64>,
        s: &Array2<f64>,
        g: &Array2<f64>,
    ) -> f64 {
        reconstruction_error(x, f, s, g)
    }

    fn normalize(
        &self,
        x: &Array2<f64>,
        f: &mut Array2<f64>,
        s: &mut Array2<f64>,
        g: &mut Array2<f64>,
    ) {
        normalize_standard(x, f, s, g);
    }
}

/// ONM3F co-clusterer implementing LocalClusterer
pub struct Onm3fClusterer {
    pub config: TriFactorConfig,
}

impl Onm3fClusterer {
    pub fn new(n_row_clusters: usize, n_col_clusters: usize) -> Self {
        Self {
            config: TriFactorConfig {
                n_row_clusters,
                n_col_clusters,
                ..Default::default()
            },
        }
    }

    pub fn with_config(config: TriFactorConfig) -> Self {
        Self { config }
    }
}

impl LocalClusterer for Onm3fClusterer {
    fn cluster_local<'a>(
        &self,
        matrix: &'a Array2<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
        let result = run_tri_factorization(&self.config, &Onm3fUpdater, matrix);
        Ok(build_submatrices_from_labels(
            matrix,
            &result.row_labels,
            &result.col_labels,
            self.config.n_row_clusters,
            self.config.n_col_clusters,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::{check_block_labels, make_block_diagonal};
    use crate::atom::tri_factor_base::run_tri_factorization;

    #[test]
    fn test_onm3f_block_diagonal() {
        let x = make_block_diagonal();
        let config = TriFactorConfig {
            n_row_clusters: 2,
            n_col_clusters: 2,
            max_iter: 20,
            n_init: 3,
            tol: 1e-9,
            seed: Some(42),
        };
        let result = run_tri_factorization(&config, &Onm3fUpdater, &x);
        assert!(
            check_block_labels(&result.row_labels, 10),
            "ONM3F row labels should separate the two blocks: {:?}",
            result.row_labels
        );
        assert!(
            check_block_labels(&result.col_labels, 10),
            "ONM3F col labels should separate the two blocks: {:?}",
            result.col_labels
        );
    }
}
