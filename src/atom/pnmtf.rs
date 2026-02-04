use std::error::Error;

use ndarray::Array2;

use crate::dimerge_co::parallel_coclusterer::LocalClusterer;
use crate::submatrix::Submatrix;

use super::normalization::normalize_pnmtf;
use super::tri_factor_base::{
    build_submatrices_from_labels, run_tri_factorization, TriFactorConfig, TriFactorUpdater,
};
use super::update_rules::{nan_to_num, reconstruction_error, sqrt_multiplicative_update};

/// PNMTF (Wang 2017) — Penalty regularization with tau, eta, gamma
pub struct PnmtfUpdater {
    pub tau: f64,
    pub eta: f64,
    pub gamma: f64,
}

impl PnmtfUpdater {
    /// Build the penalty matrix P = ones(k,k) - I_k
    fn penalty_matrix(k: usize) -> Array2<f64> {
        let mut p = Array2::ones((k, k));
        for i in 0..k {
            p[[i, i]] = 0.0;
        }
        p
    }
}

impl TriFactorUpdater for PnmtfUpdater {
    fn update_f(&self, x: &Array2<f64>, f: &mut Array2<f64>, s: &Array2<f64>, g: &Array2<f64>) {
        // F ← F * ((X*G*S^T) / (F*S*G^T*G*S^T + tau*F*P_g))^0.5
        let k = f.ncols();
        let p_g = Self::penalty_matrix(k);
        let numer = x.dot(g).dot(&s.t());
        let denom = f.dot(s).dot(&g.t()).dot(g).dot(&s.t()) + &(self.tau * f.dot(&p_g));
        *f = sqrt_multiplicative_update(f, &numer, &denom, 1e-16);
    }

    fn update_g(&self, x: &Array2<f64>, f: &Array2<f64>, s: &Array2<f64>, g: &mut Array2<f64>) {
        // G ← G * ((X^T*F*S) / (G*S^T*F^T*F*S + eta*G*P_s))^0.5
        let l = g.ncols();
        let p_s = Self::penalty_matrix(l);
        let numer = x.t().dot(f).dot(s);
        let denom = g.dot(&s.t()).dot(&f.t()).dot(f).dot(s) + &(self.eta * g.dot(&p_s));
        *g = sqrt_multiplicative_update(g, &numer, &denom, 1e-16);
    }

    fn update_s(&self, x: &Array2<f64>, f: &Array2<f64>, s: &mut Array2<f64>, g: &Array2<f64>) {
        // S ← S * ((F^T*X*G) / (F^T*F*S*G^T*G + gamma*S))^0.5
        let numer = f.t().dot(x).dot(g);
        let denom = f.t().dot(f).dot(&*s).dot(&g.t()).dot(g) + &(self.gamma * &*s);
        *s = sqrt_multiplicative_update(s, &numer, &denom, 1e-16);
    }

    fn compute_criterion(
        &self,
        x: &Array2<f64>,
        f: &Array2<f64>,
        s: &Array2<f64>,
        g: &Array2<f64>,
    ) -> f64 {
        let k = f.ncols();
        let l = g.ncols();
        let p_g = Self::penalty_matrix(k);
        let p_s = Self::penalty_matrix(l);

        let recon = reconstruction_error(x, f, s, g);
        // tr(F * P_g * F^T) = sum of elementwise (F * P_g * F^T)
        let penalty_f = f.dot(&p_g).dot(&f.t()).diag().sum();
        let penalty_g = g.dot(&p_s).dot(&g.t()).diag().sum();
        let penalty_s = s.t().dot(s).diag().sum();

        0.5 * recon
            + 0.5 * self.tau * penalty_f
            + 0.5 * self.eta * penalty_g
            + 0.5 * self.gamma * penalty_s
    }

    fn normalize(
        &self,
        _x: &Array2<f64>,
        f: &mut Array2<f64>,
        s: &mut Array2<f64>,
        g: &mut Array2<f64>,
    ) {
        normalize_pnmtf(f, s, g);
    }
}

/// PNMTF co-clusterer implementing LocalClusterer
pub struct PnmtfClusterer {
    pub config: TriFactorConfig,
    pub tau: f64,
    pub eta: f64,
    pub gamma: f64,
}

impl PnmtfClusterer {
    pub fn new(n_row_clusters: usize, n_col_clusters: usize) -> Self {
        Self {
            config: TriFactorConfig {
                n_row_clusters,
                n_col_clusters,
                ..Default::default()
            },
            tau: 0.0,
            eta: 0.0,
            gamma: 0.0,
        }
    }

    pub fn with_penalties(mut self, tau: f64, eta: f64, gamma: f64) -> Self {
        self.tau = tau;
        self.eta = eta;
        self.gamma = gamma;
        self
    }

    pub fn with_config(config: TriFactorConfig, tau: f64, eta: f64, gamma: f64) -> Self {
        Self {
            config,
            tau,
            eta,
            gamma,
        }
    }
}

impl LocalClusterer for PnmtfClusterer {
    fn cluster_local<'a>(
        &self,
        matrix: &'a Array2<f64>,
    ) -> Result<Vec<Submatrix<'a, f64>>, Box<dyn Error>> {
        let updater = PnmtfUpdater {
            tau: self.tau,
            eta: self.eta,
            gamma: self.gamma,
        };
        let result = run_tri_factorization(&self.config, &updater, matrix);
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
    fn test_pnmtf_block_diagonal() {
        let x = make_block_diagonal();
        let config = TriFactorConfig {
            n_row_clusters: 2,
            n_col_clusters: 2,
            max_iter: 20,
            n_init: 3,
            tol: 1e-9,
            seed: Some(42),
        };
        let updater = PnmtfUpdater {
            tau: 0.1,
            eta: 0.1,
            gamma: 0.1,
        };
        let result = run_tri_factorization(&config, &updater, &x);
        assert!(
            check_block_labels(&result.row_labels, 10),
            "PNMTF row labels should separate the two blocks: {:?}",
            result.row_labels
        );
        assert!(
            check_block_labels(&result.col_labels, 10),
            "PNMTF col labels should separate the two blocks: {:?}",
            result.col_labels
        );
    }
}
