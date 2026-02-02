/**
 * File: /src/cocluster.rs
 * Created Date: Thursday, June 13th 2024
 * Author: Zihan
 * -----
 * Last Modified: Monday, 26th May 2025 11:59:26 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 18-06-2024		Zihan	Refactor Coclusterer and Add Utility Functions
**/
// src/cocluster.rs
use ndarray::{s, Array2};
use ndarray_linalg::SVD as NdarraySVD;
extern crate nalgebra as na;
use crate::util::are_equivalent_classifications;
use std::fmt;

use linfa::prelude::*;
use linfa_clustering::KMeans as LinfaKMeans;

/// A co-clustering structure that performs k-means clustering on the rows and columns of a matrix.
pub struct Coclusterer {
    /// The matrix to be co-clustered.
    matrix: Array2<f64>,
    /// The number of rows in the matrix.
    row: usize,
    /// The number of columns in the matrix.
    col: usize,
    /// The number of co-clusters.
    k: usize,
    /// The tolerance for score.
    tol: f64,
}

/// Clones a 2D ndarray array view into a nalgebra DMatrix.
///
/// # Arguments
///
/// * `array_view` - The 2D array view to clone.
///
/// # Returns
///
/// * A DMatrix containing the cloned data.

impl Coclusterer {
    /// Creates a new `Coclusterer`.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The matrix to be co-clustered.
    /// * `m` - The number of row clusters.
    /// * `n` - The number of column clusters.
    /// * `tol` - The tolerance for score.
    ///
    /// # Returns
    ///
    /// * A new `Coclusterer`.
    pub fn new(matrix: Array2<f64>, k: usize, tol: f64) -> Coclusterer {
        let row = matrix.shape()[0];
        let col = matrix.shape()[1];
        Coclusterer {
            matrix,
            row,
            col,
            k,
            tol,
        }
    }

    /// Performs co-clustering on the matrix and returns a list of submatrices that meet the tolerance criteria.
    ///
    /// # Returns
    ///
    /// * A vector that gives the groupiong of the rows and columns.
    pub(crate) fn cocluster(&mut self) -> Result<Vec<usize>, &'static str> {
        log::info!(
            "Coclusterer::cocluster: 矩阵 {}×{}, k={}",
            self.row, self.col, self.k
        );

        let svd_start = std::time::Instant::now();
        let epsilon = 1e-10;

        // 计算行和与列和
        let du: Vec<f64> = (0..self.row)
            .map(|r| self.matrix.row(r).sum())
            .collect();
        let dv: Vec<f64> = (0..self.col)
            .map(|c| self.matrix.column(c).sum())
            .collect();

        // D_u^{-1/2}, D_v^{-1/2}，零值归零防止 Inf/NaN
        let du_inv_sqrt: Vec<f64> = du.iter()
            .map(|&x| if x.abs() < epsilon { 0.0 } else { x.powf(-0.5) })
            .collect();
        let dv_inv_sqrt: Vec<f64> = dv.iter()
            .map(|&x| if x.abs() < epsilon { 0.0 } else { x.powf(-0.5) })
            .collect();

        let zero_rows = du.iter().filter(|&&x| x.abs() < epsilon).count();
        let zero_cols = dv.iter().filter(|&&x| x.abs() < epsilon).count();
        if zero_rows > 0 || zero_cols > 0 {
            log::warn!(
                "  归一化: {} 零行, {} 零列（已用 epsilon 保护）",
                zero_rows, zero_cols
            );
        }

        // 归一化矩阵（直接使用 ndarray）
        let mut normalized = Array2::<f64>::zeros((self.row, self.col));
        for r in 0..self.row {
            for c in 0..self.col {
                normalized[[r, c]] = self.matrix[[r, c]] * du_inv_sqrt[r] * dv_inv_sqrt[c];
            }
        }

        let k = self.k;
        log::info!("  SVD 开始 ({}×{} 归一化矩阵, ndarray-linalg/OpenBLAS)...", self.row, self.col);
        let (u_mat, v_mat) = perform_svd(&normalized, k)?;
        log::info!("  SVD 完成 [{} ms]", svd_start.elapsed().as_millis());

        // 拼接 u, v → f 矩阵 (row+col, k)
        let n_samples = self.row + self.col;
        let n_features = k;
        let mut f_array = Array2::<f64>::zeros((n_samples, n_features));
        for r in 0..self.row {
            for c in 0..k {
                f_array[[r, c]] = u_mat[[r, c]];
            }
        }
        for r in 0..self.col {
            for c in 0..k {
                f_array[[self.row + r, c]] = v_mat[[r, c]];
            }
        }

        // Identify samples with all-zero embeddings (from stripped zero rows/cols in SVD).
        // These get random K-means assignments and would pollute clusters.
        let zero_embedding: Vec<bool> = (0..n_samples)
            .map(|r| (0..n_features).all(|c| f_array[[r, c]].abs() < 1e-15))
            .collect();

        // Normalize each row of f_array to unit L2 norm before K-means.
        // This is the standard spectral co-clustering normalization (Dhillon 2001)
        // that prevents row-dominated vs column-dominated clusters in joint K-means.
        for r in 0..n_samples {
            let norm: f64 = (0..n_features).map(|c| f_array[[r, c]].powi(2)).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for c in 0..n_features {
                    f_array[[r, c]] /= norm;
                }
            }
        }

        // K-means 聚类
        log::info!("  K-means 开始 ({} samples, {} features, k={})...", n_samples, n_features, k);
        let kmeans_start = std::time::Instant::now();
        let dataset = DatasetBase::from(f_array);
        let model = LinfaKMeans::params(k)
            .max_n_iterations(100)
            .fit(&dataset)
            .map_err(|_| "K-means clustering failed")?;

        let predictions = model.predict(dataset);
        let mut assignments: Vec<usize> = predictions.targets.to_vec();
        log::info!("  K-means 完成 [{} ms]", kmeans_start.elapsed().as_millis());

        // Mark zero-embedding samples as unassigned
        let mut unassigned_rows = 0;
        let mut unassigned_cols = 0;
        for (idx, &is_zero) in zero_embedding.iter().enumerate() {
            if is_zero {
                assignments[idx] = usize::MAX;
                if idx < self.row {
                    unassigned_rows += 1;
                } else {
                    unassigned_cols += 1;
                }
            }
        }
        if unassigned_rows > 0 || unassigned_cols > 0 {
            log::info!(
                "  排除零嵌入样本: {} 行, {} 列 (不参与聚类)",
                unassigned_rows, unassigned_cols
            );
        }

        Ok(assignments)
    }
}

/// 使用 ndarray-linalg SVD（通过 OpenBLAS/LAPACK）
///
/// Zero rows and columns (from normalization of zero-sum rows/columns) are stripped
/// before SVD to avoid LAPACK convergence issues on highly degenerate matrices, then
/// reconstructed with zero embeddings in the output U and V matrices.
/// If LAPACK dgesdd fails, falls back to nalgebra SVD.
fn perform_svd(
    matrix: &Array2<f64>,
    k: usize,
) -> Result<(Array2<f64>, Array2<f64>), &'static str> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    // Identify non-zero rows and columns; zero rows/columns cause LAPACK dgesdd convergence failures
    let non_zero_rows: Vec<usize> = (0..nrows)
        .filter(|&r| matrix.row(r).iter().any(|&v| v.abs() > 1e-15))
        .collect();
    let non_zero_cols: Vec<usize> = (0..ncols)
        .filter(|&c| matrix.column(c).iter().any(|&v| v.abs() > 1e-15))
        .collect();

    if non_zero_cols.is_empty() || non_zero_rows.is_empty() {
        log::warn!("  SVD: all rows or columns are zero, returning zero embeddings");
        return Ok((
            Array2::<f64>::zeros((nrows, k)),
            Array2::<f64>::zeros((ncols, k)),
        ));
    }

    let rows_removed = nrows - non_zero_rows.len();
    let cols_removed = ncols - non_zero_cols.len();
    if rows_removed > 0 || cols_removed > 0 {
        log::info!(
            "  SVD: stripped {} zero rows ({} → {}), {} zero columns ({} → {}) before decomposition",
            rows_removed, nrows, non_zero_rows.len(),
            cols_removed, ncols, non_zero_cols.len()
        );
    }

    // Build reduced matrix (only non-zero rows and columns)
    let reduced = if rows_removed > 0 || cols_removed > 0 {
        let mut reduced = Array2::<f64>::zeros((non_zero_rows.len(), non_zero_cols.len()));
        for (new_r, &orig_r) in non_zero_rows.iter().enumerate() {
            for (new_c, &orig_c) in non_zero_cols.iter().enumerate() {
                reduced[[new_r, new_c]] = matrix[[orig_r, orig_c]];
            }
        }
        reduced
    } else {
        matrix.clone()
    };

    // Use randomized SVD when k is much smaller than matrix dimensions (huge speedup).
    // Fall back to full SVD for small matrices or when k is close to min(m, n).
    let min_dim = reduced.nrows().min(reduced.ncols());
    let use_randomized = k < min_dim / 2 && min_dim > 50;

    let (u_truncated, v_reduced) = if use_randomized {
        log::info!(
            "  SVD: using randomized truncated SVD (k={} << min_dim={})",
            k, min_dim
        );
        randomized_svd(&reduced, k, 10, 3)?
    } else {
        log::info!(
            "  SVD: using full LAPACK SVD (k={}, min_dim={})",
            k, min_dim
        );
        // Try LAPACK dgesdd first, fall back to nalgebra if it fails
        let svd_result = reduced.svd(true, true);
        let (u_reduced, vt_reduced) = match svd_result {
            Ok((u_opt, _sigma, vt_opt)) => {
                let u = u_opt.ok_or("Error: U matrix is None")?;
                let vt = vt_opt.ok_or("Error: Vt matrix is None")?;
                (u, vt)
            }
            Err(e) => {
                log::warn!("  SVD LAPACK dgesdd failed ({:?}), falling back to nalgebra SVD...", e);
                nalgebra_svd_fallback(&reduced)?
            }
        };

        // Truncate to k singular vectors
        let k = k.min(u_reduced.ncols()).min(vt_reduced.nrows());
        let u_truncated = u_reduced.slice(s![.., ..k]).to_owned();
        let v_truncated = vt_reduced.t().slice(s![.., ..k]).to_owned();
        (u_truncated, v_truncated)
    };

    // Reconstruct full U and V matrices with zero embeddings for stripped rows/columns
    let u_full = if rows_removed > 0 {
        let mut u_full = Array2::<f64>::zeros((nrows, k));
        for (new_r, &orig_r) in non_zero_rows.iter().enumerate() {
            u_full.row_mut(orig_r).assign(&u_truncated.row(new_r));
        }
        u_full
    } else {
        u_truncated
    };

    let v_full = if cols_removed > 0 {
        let mut v_full = Array2::<f64>::zeros((ncols, k));
        for (new_c, &orig_c) in non_zero_cols.iter().enumerate() {
            v_full.row_mut(orig_c).assign(&v_reduced.row(new_c));
        }
        v_full
    } else {
        v_reduced
    };

    Ok((u_full, v_full))
}

/// Fallback SVD using nalgebra when LAPACK dgesdd fails to converge.
fn nalgebra_svd_fallback(
    matrix: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), &'static str> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    // Convert ndarray → nalgebra DMatrix
    let na_matrix = na::DMatrix::from_fn(nrows, ncols, |r, c| matrix[[r, c]]);

    let svd = na::linalg::SVD::new(na_matrix, true, true);

    let u_na = svd.u.ok_or("nalgebra SVD: U matrix is None")?;
    let vt_na = svd.v_t.ok_or("nalgebra SVD: Vt matrix is None")?;

    // Convert nalgebra → ndarray
    let mut u = Array2::<f64>::zeros((nrows, u_na.ncols()));
    for r in 0..nrows {
        for c in 0..u_na.ncols() {
            u[[r, c]] = u_na[(r, c)];
        }
    }

    let mut vt = Array2::<f64>::zeros((vt_na.nrows(), ncols));
    for r in 0..vt_na.nrows() {
        for c in 0..ncols {
            vt[[r, c]] = vt_na[(r, c)];
        }
    }

    log::info!("  SVD: nalgebra fallback succeeded ({}×{} → U {}×{}, Vt {}×{})",
        nrows, ncols, u.nrows(), u.ncols(), vt.nrows(), vt.ncols());

    Ok((u, vt))
}

/// Randomized truncated SVD using Halko et al. 2011 (Algorithm 5.1).
///
/// Computes an approximate rank-k SVD in O(m*n*k) time instead of O(m*n*min(m,n)).
/// Returns (U[:, :k], V[:, :k]) — left and right singular vectors.
fn randomized_svd(
    matrix: &Array2<f64>,
    k: usize,
    n_oversampling: usize,
    n_power_iter: usize,
) -> Result<(Array2<f64>, Array2<f64>), &'static str> {
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;

    let (m, n) = (matrix.nrows(), matrix.ncols());
    let l = (k + n_oversampling).min(m).min(n);

    log::info!(
        "  Randomized SVD: {}×{} matrix, k={}, l={} (oversampling={}), power_iter={}",
        m, n, k, l, n_oversampling, n_power_iter
    );

    // Step 1: Random Gaussian projection matrix Omega (n x l)
    let omega: Array2<f64> = Array2::random((n, l), StandardNormal);

    // Step 2: Y = A * Omega (m x l)
    let mut y = matrix.dot(&omega);

    // Step 3: Power iterations for better approximation: Y = A * (A^T * (A * ... ))
    // with QR stabilization at each step
    for i in 0..n_power_iter {
        // QR factorization of Y for numerical stability
        y = qr_q(&y)?;
        // Y = A^T * Q  (n x l)
        let z = matrix.t().dot(&y);
        // QR factorization of Z for numerical stability
        let z_q = qr_q(&z)?;
        // Y = A * Q_z  (m x l)
        y = matrix.dot(&z_q);
        log::debug!("  Randomized SVD: power iteration {}/{}", i + 1, n_power_iter);
    }

    // Step 4: QR factorization of Y to get orthonormal basis Q
    let q = qr_q(&y)?;

    // Step 5: Project to low-dimensional space: B = Q^T * A (l x n)
    let b = q.t().dot(matrix);

    // Step 6: Thin SVD of B (l x n) via nalgebra.
    // ndarray-linalg's svd() allocates full n×n Vt which is too large when n >> l.
    // nalgebra's svd() produces thin factors: U(l×l), S(l), V(n×l).
    log::info!("  Randomized SVD: computing thin SVD of {}×{} B matrix via nalgebra", b.nrows(), b.ncols());
    let (bl, bn) = (b.nrows(), b.ncols());
    let na_b = na::DMatrix::from_fn(bl, bn, |r, c| b[[r, c]]);
    let na_svd = na_b.svd(true, true);
    let na_u = na_svd.u.ok_or("Randomized SVD: nalgebra U is None")?;
    let na_v = na_svd.v_t.ok_or("Randomized SVD: nalgebra Vt is None")?.transpose();

    // Convert nalgebra → ndarray: U_B is l×l, V_B is n×min(l,n)
    let u_b = Array2::from_shape_fn((na_u.nrows(), na_u.ncols()), |(r, c)| na_u[(r, c)]);
    let v_b = Array2::from_shape_fn((na_v.nrows(), na_v.ncols()), |(r, c)| na_v[(r, c)]);

    // Step 7: Recover U = Q * U_B[:, :k], V = V_B[:, :k]
    let k_actual = k.min(u_b.ncols()).min(v_b.ncols());
    let u_b_k = u_b.slice(s![.., ..k_actual]).to_owned();
    let u = q.dot(&u_b_k);
    let v = v_b.slice(s![.., ..k_actual]).to_owned();

    log::info!(
        "  Randomized SVD: done, U {}×{}, V {}×{}",
        u.nrows(), u.ncols(), v.nrows(), v.ncols()
    );

    Ok((u, v))
}

/// Compute the Q factor of a thin QR decomposition using nalgebra.
fn qr_q(matrix: &Array2<f64>) -> Result<Array2<f64>, &'static str> {
    let (m, n) = (matrix.nrows(), matrix.ncols());

    // Convert ndarray → nalgebra DMatrix
    let na_matrix = na::DMatrix::from_fn(m, n, |r, c| matrix[[r, c]]);

    let qr = na_matrix.qr();
    let q_na = qr.q();

    // The thin Q has shape (m x min(m,n)); we only need the first n columns
    let q_cols = n.min(m);
    let mut q = Array2::<f64>::zeros((m, q_cols));
    for r in 0..m {
        for c in 0..q_cols {
            q[[r, c]] = q_na[(r, c)];
        }
    }

    Ok(q)
}

#[cfg(test)]
mod tests {
    use na::{Matrix3, QR};
    use ndarray::{array, Array2};
    use ndarray_rand::RandomExt;
    use rand::{random, Rng};

    use crate::submatrix::Submatrix;

    use super::*;

    #[test]
    /// Test for updating the score of a submatrix.
    fn test_update_score() {
        // submatrix is smaller than 3 *. 3
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut submatrix_opt = crate::submatrix::Submatrix::new(&data, vec![0, 1], vec![0, 1]);

        if let Some(mut sm) = submatrix_opt {
            sm.update_score();
            assert_eq!(sm.score, None::<f64>);
        }

        // submatrix is larger than 3*3

        /*
        A = U * S * V^T, where S = diag(2, 1, 0)

        test for submatrix:
        |A.score - 0.5| < 1e-6

         */

        // U, V random orthogonal matrix
        let u_matrix = random_orthogonal_matrix();
        let v_matrix = random_orthogonal_matrix();
        // S diagonal matrix (2, 1, 0)
        let s_matrix = Matrix3::new(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0., 0., 0.);
        // A = U * S * V
        let a_matrix = u_matrix * s_matrix * v_matrix.transpose();
        let a_vec = a_matrix.as_slice().to_vec();

        let matrix_data = Array2::<f64>::from_shape_vec((3, 3), a_vec).unwrap();
        let mut submatrix_opt_large = crate::submatrix::Submatrix::new(&matrix_data, vec![0, 1, 2], vec![0, 1, 2]);

        if let Some(ref mut sm_large) = submatrix_opt_large {
            sm_large.update_score();
        }

        let score = submatrix_opt_large.expect("Submatrix should be Some").score.expect("Score should be Some after update");

        println!("score: {}", &score);

        assert!((score - 2.0).abs() < 1e-5);
    }

    /// test cocluster
    #[test]
    fn test_cocluster() {
        // test with
        // B = np.array(
        // [
        //     [0, 0, 1, 1, 0],
        //     [0, 1, 1, 0, 0],
        //     [0, 0, 2, 2, 0],
        //     [0, 1, 1, 0, 0],
        //     [1, 0, 0, 0, 1],
        //     [0, 0, 3, 3, 0],
        // ]
        // )
        let b_matrix = Array2::from_shape_vec(
            (6, 5),
            vec![
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0,
                1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 3.0, 0.0,
            ],
        )
        .unwrap();

        let mut success_count = 0;
        let mut failure_count = 0;

        // 循环调用 cocluster_test_helper 100 次
        for _ in 0..10000 {
            match cocluster_test_helper(b_matrix.clone()) {
                Ok(_) => success_count += 1,
                Err(e) => {
                    failure_count += 1;
                    println!("Test failed with error: {}", e);
                }
            }
        }

        // 计算并打印成功率
        let total = success_count + failure_count;
        let success_rate = (success_count as f64 / total as f64) * 100.0;
        println!("Success rate: {:.2}%", success_rate);

        // 可选：在成功率低于某个阈值时，测试失败
        assert!(
            success_rate >= 97.0,
            "Success rate is below acceptable threshold: {:.2}%",
            success_rate
        );
    }
    fn cocluster_test_helper(
        b_matrix: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
    ) -> Result<(), String> {
        let mut coclusterer = Coclusterer::new(b_matrix, 3, 1e-1);
        let assignment_vec = coclusterer
            .cocluster()
            .map_err(|e| format!("Cocluster error: {:?}", e))?;
        let expected_vec = vec![2, 0, 2, 0, 1, 2, 1, 0, 2, 2, 1];

        if are_equivalent_classifications(assignment_vec.clone(), expected_vec) {
            Ok(())
        } else {
            Err(format!("assignment_vec: {:?}", assignment_vec))
        }
    }

    /// Generates a random orthogonal matrix.
    ///
    /// # Returns
    ///
    /// * A random 3x3 orthogonal matrix.
    fn random_orthogonal_matrix() -> Matrix3<f64> {
        let mut rng = rand::rng();

        // 随机生成一个 3x3 矩阵
        let mat: Matrix3<f64> = Matrix3::from_fn(|_, _| rng.random::<f64>());

        // 进行 QR 分解
        let qr = QR::new(mat);
        let q = qr.q();

        q
    }
}
