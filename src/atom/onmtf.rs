use std::error::Error;

use ndarray::Array2;

use crate::dimerge_co::parallel_coclusterer::LocalClusterer;
use crate::submatrix::Submatrix;

use super::normalization::normalize_standard;
use super::tri_factor_base::{
    build_submatrices_from_labels, run_tri_factorization, TriFactorConfig, TriFactorUpdater,
};
use super::update_rules::{multiplicative_update, reconstruction_error};

/// ONMTF (Yoo 2010) — Different denominator coupling
pub struct OnmtfUpdater;

impl TriFactorUpdater for OnmtfUpdater {
    fn update_f(&self, x: &Array2<f64>, f: &mut Array2<f64>, s: &Array2<f64>, g: &Array2<f64>) {
        // F ← F * (X*G*S^T) / (F*S*G^T*X^T*F)
        let numer = x.dot(g).dot(&s.t());
        let denom = f.dot(s).dot(&g.t()).dot(&x.t()).dot(&*f);
        *f = multiplicative_update(f, &numer, &denom, 1e-16);
    }

    fn update_g(&self, x: &Array2<f64>, f: &Array2<f64>, s: &Array2<f64>, g: &mut Array2<f64>) {
        // G ← G * (X^T*F*S) / (G*S^T*F^T*X*G)
        let numer = x.t().dot(f).dot(s);
        let denom = g.dot(&s.t()).dot(&f.t()).dot(x).dot(&*g);
        *g = multiplicative_update(g, &numer, &denom, 1e-16);
    }

    fn update_s(&self, x: &Array2<f64>, f: &Array2<f64>, s: &mut Array2<f64>, g: &Array2<f64>) {
        // S ← S * (F^T*X*G) / (F^T*F*S*G^T*G)
        let numer = f.t().dot(x).dot(g);
        let denom = f.t().dot(f).dot(&*s).dot(&g.t()).dot(g);
        *s = multiplicative_update(s, &numer, &denom, 1e-16);
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

/// ONMTF co-clusterer implementing LocalClusterer
pub struct OnmtfClusterer {
    pub config: TriFactorConfig,
}

impl OnmtfClusterer {
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

impl LocalClusterer for OnmtfClusterer {
    fn cluster_local<'a>(
        &self,
        matrix: &'a Array2<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
        let result = run_tri_factorization(&self.config, &OnmtfUpdater, matrix);
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
    fn test_onmtf_block_diagonal() {
        let x = make_block_diagonal();
        let config = TriFactorConfig {
            n_row_clusters: 2,
            n_col_clusters: 2,
            max_iter: 20,
            n_init: 3,
            tol: 1e-9,
            seed: Some(42),
        };
        let result = run_tri_factorization(&config, &OnmtfUpdater, &x);
        assert!(
            check_block_labels(&result.row_labels, 10),
            "ONMTF row labels should separate the two blocks: {:?}",
            result.row_labels
        );
        assert!(
            check_block_labels(&result.col_labels, 10),
            "ONMTF col labels should separate the two blocks: {:?}",
            result.col_labels
        );
    }
}
