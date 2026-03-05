use std::error::Error;

use ndarray::Array2;

use crate::dimerge_co::parallel_coclusterer::LocalClusterer;
use crate::submatrix::Submatrix;

use super::normalization::normalize_standard;
use super::tri_factor_base::{
    build_submatrices_from_labels, run_tri_factorization, TriFactorConfig, TriFactorUpdater,
};
use super::update_rules::{multiplicative_update, reconstruction_error};

/// NBVD (Long 2005) — Basic multiplicative updates for X ≈ F*S*G^T
pub struct NbvdUpdater;

impl TriFactorUpdater for NbvdUpdater {
    fn update_f(&self, x: &Array2<f64>, f: &mut Array2<f64>, s: &Array2<f64>, g: &Array2<f64>) {
        // F ← F * (X*G*S^T) / (F*S*G^T*G*S^T)
        // Optimized: avoid m×n intermediate by computing (G^T*G) first
        // Denom = F * S * (G^T*G) * S^T  (all k×k intermediates except F×result)
        let gtg = g.t().dot(g); // k×k
        let numer = x.dot(g).dot(&s.t()); // m×k
        let denom = f.dot(s).dot(&gtg).dot(&s.t()); // m×k (no m×n intermediate!)
        *f = multiplicative_update(f, &numer, &denom, 1e-10);
    }

    fn update_g(&self, x: &Array2<f64>, f: &Array2<f64>, s: &Array2<f64>, g: &mut Array2<f64>) {
        // G ← G * (X^T*F*S) / (G*S^T*F^T*F*S)
        // Optimized: avoid n×m intermediate by computing (F^T*F) first
        // Denom = G * S^T * (F^T*F) * S  (all k×k intermediates except G×result)
        let ftf = f.t().dot(f); // k×k
        let numer = x.t().dot(f).dot(s); // n×k
        let denom = g.dot(&s.t()).dot(&ftf).dot(s); // n×k (no n×m intermediate!)
        *g = multiplicative_update(g, &numer, &denom, 1e-10);
    }

    fn update_s(&self, x: &Array2<f64>, f: &Array2<f64>, s: &mut Array2<f64>, g: &Array2<f64>) {
        // S ← S * (F^T*X*G) / (F^T*F*S*G^T*G)
        // Already efficient: all intermediates are k×k or k×m/n
        let ftf = f.t().dot(f); // k×k
        let gtg = g.t().dot(g); // k×k
        let numer = f.t().dot(x).dot(g); // k×k
        let denom = ftf.dot(&*s).dot(&gtg); // k×k
        *s = multiplicative_update(s, &numer, &denom, 1e-10);
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

/// NBVD co-clusterer implementing LocalClusterer
pub struct NbvdClusterer {
    pub config: TriFactorConfig,
}

impl NbvdClusterer {
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

impl LocalClusterer for NbvdClusterer {
    fn cluster_local<'a>(
        &self,
        matrix: &'a Array2<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
        let result = run_tri_factorization(&self.config, &NbvdUpdater, matrix);
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
    fn test_nbvd_block_diagonal() {
        let x = make_block_diagonal();
        let config = TriFactorConfig {
            n_row_clusters: 2,
            n_col_clusters: 2,
            max_iter: 20,
            n_init: 3,
            tol: 1e-9,
            seed: Some(42),
        };
        let result = run_tri_factorization(&config, &NbvdUpdater, &x);
        assert!(
            check_block_labels(&result.row_labels, 10),
            "NBVD row labels should separate the two blocks: {:?}",
            result.row_labels
        );
        assert!(
            check_block_labels(&result.col_labels, 10),
            "NBVD col labels should separate the two blocks: {:?}",
            result.col_labels
        );
    }

    #[test]
    fn test_nbvd_clusterer_produces_submatrices() {
        let x = make_block_diagonal();
        let clusterer = NbvdClusterer {
            config: TriFactorConfig {
                n_row_clusters: 2,
                n_col_clusters: 2,
                max_iter: 20,
                n_init: 3,
                tol: 1e-9,
                seed: Some(42),
            },
        };
        let subs = clusterer.cluster_local(&x).unwrap();
        assert!(!subs.is_empty(), "Should produce at least one submatrix");
        // With 2x2 co-clustering on block diagonal, expect 2 or 4 submatrices
        assert!(subs.len() <= 4);
    }
}
