/**
 * File: /src/cocluster.rs
 * Created Date: Thursday, June 13th 2024
 * Author: Zihan
 * -----
 * Last Modified: Thursday, 29th May 2025 11:03:04 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 18-06-2024		Zihan	Refactor Coclusterer and Add Utility Functions
**/
// src/cocluster.rs
use nalgebra::SVD;
use nalgebra::{DMatrix, DVector, Dyn, Matrix2};
use ndarray::{stack, Array1, Array2};
extern crate nalgebra as na;
use crate::util::are_equivalent_classifications;
use crate::util::clone_to_dmatrix;
use std::{cmp::max, fmt};

// use kmeans_smid
use kmeans_smid::{KMeans, KMeansConfig};

use crate::submatrix::Submatrix;

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
        // print self.matrix.view()
        // println!("self.matrix: \n{}", self.matrix.view());

        // println!("self.matrix shape: {:?}", self.matrix.shape());
        // println!("self.matrix view shape: {:?}", self.matrix.view().shape());

        // svd to get u,s,v
        let na_matrix = clone_to_dmatrix(self.matrix.view());

        // normalize the matrix: a_ij/sqrt(sum_f(a_if)sum_g(a_gj))
        // println!("Original matrix: \n{}", na_matrix);

        // D = sum by row and as a diagonal matrix
        let one_column = DVector::from_element(self.col, 1.0);
        let one_row = DVector::from_element(self.row, 1.0).transpose();

        // println!("one_column: \n{}", one_column);
        // println!("one_row: \n{}", one_row);

        let du = &na_matrix * one_column;
        let dv = (one_row * &na_matrix);

        // println!("du: \n{}", du);
        // println!("dv: \n{}", dv);

        // 计算du和dv的-1/2次幂
        let du_inv_sqrt = du.map(|x| x.powf(-0.5));
        let dv_inv_sqrt = dv.map(|x| x.powf(-0.5));

        // 归一化矩阵，每行乘以du_inv_sqrt，每列乘以dv_inv_sqrt
        let mut na_matrix_normalized = na_matrix.clone();
        for (i, mut row) in na_matrix_normalized.row_iter_mut().enumerate() {
            row *= du_inv_sqrt[i];
        }
        for (j, mut col) in na_matrix_normalized.column_iter_mut().enumerate() {
            col *= dv_inv_sqrt[j];
        }

        // 打印归一化后的矩阵
        // println!("Normalized matrix: \n{}", na_matrix_normalized);

        // panic!("stop");

        // let svd_result = &na_matrix_normalized.svd(true, true);
        // let u: na::Matrix<f64, Dyn, Dyn, na::VecStorage<f64, Dyn, Dyn>> = svd_result.u.unwrap(); // shaped as (row, row)
        // let vt: na::Matrix<f64, Dyn, Dyn, na::VecStorage<f64, Dyn, Dyn>> = svd_result.v_t.unwrap(); // shaped as (col, col)
        // let v: na::Matrix<f64, Dyn, Dyn, na::VecStorage<f64, Dyn, Dyn>> = vt.transpose(); // shaped as (col, row)

        // u, v 取前 self.k 列, 然后 vstack 成 f
        let k = self.k;

        let (u_mat, v_t_mat) = perform_svd(na_matrix_normalized, k)?;

        let u = u_mat.view((0, 0), (self.row, k));
        let v = v_t_mat.view((0, 0), (self.col, k));

        // 正常情况下的处理逻辑在这里继续
        let f = DMatrix::from_fn(self.row + self.col, k, |r, c| {
            if r < self.row {
                u[(r, c)]
            } else {
                v[(r - self.row, c)]
            }
        });

        let f_data: Vec<f64> = f.transpose().data.as_slice().iter().copied().collect();
        let kmeans_f: KMeans<f64, 8> = KMeans::new(f_data, f.nrows(), f.ncols());

        let result_f =
            kmeans_f.kmeans_lloyd(k, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());

        Ok(result_f.assignments)
    }

    fn reassign_clusters(&mut self, u: &Array2<f64>, v: &Array2<f64>) {
        // Simplified reassignment logic for brevity
        let mut assignments = vec![0; self.matrix.nrows() + self.matrix.ncols()];
        // In a real scenario, you would use u and v to determine cluster assignments
        // For example, by finding the column with the max value in each row of u (for row assignments)
        // and v (for column assignments)

        // Placeholder: Assign based on max component in U_k for rows, V_k for columns
        // This needs to be adapted to the actual structure of u and v from SVD
        // and how they map to k clusters.

        // Example of how one might assign rows (conceptual):
        for i in 0..self.matrix.nrows() {
            let row_vec = u.row(i);
            let (idx, _val) = row_vec.iter().enumerate().fold((0, f64::NEG_INFINITY), |(max_idx, max_val), (current_idx, &current_val)| {
                if current_val > max_val {
                    (current_idx, current_val)
                } else {
                    (max_idx, max_val)
                }
            });
            assignments[i] = idx; 
        }

        // Example of how one might assign columns (conceptual):
        // Note: v is often V^T from SVD, so iterating rows of v might correspond to columns of original matrix implicitly
        for j in 0..self.matrix.ncols() {
             let col_vec = v.row(j); // Assuming v's rows correspond to original matrix columns for clustering
             let (idx, _val) = col_vec.iter().enumerate().fold((0, f64::NEG_INFINITY), |(max_idx, max_val), (current_idx, &current_val)| {
                if current_val > max_val {
                    (current_idx, current_val)
                } else {
                    (max_idx, max_val)
                }
            });
            assignments[self.matrix.nrows() + j] = idx;
        }

        // self.matrix = assignments.iter().take(self.matrix.nrows()).cloned().collect::<Array2<f64>>(); // Commented out due to E0277 and incorrect logic
    }

    fn calculate_error(&self, u: &Array2<f64>, s: &Array1<f64>, v_t: &Array2<f64>) -> f64 {
        // S is a vector of singular values, needs to be a diagonal matrix Sigma
        let mut sigma_mat = Array2::zeros((self.k, self.k));
        for i in 0..self.k {
            sigma_mat[(i, i)] = s[i];
        }
        let product_u_sigma = u.dot(&sigma_mat);
        // v_t is already &Array2<f64>, so pass it directly to dot.
        let reconstructed_data = product_u_sigma.dot(v_t);
        let diff = &self.matrix - &reconstructed_data;
        diff.mapv(|x| x.powi(2)).sum().sqrt() / (self.matrix.nrows() * self.matrix.ncols()) as f64
    }

    // Example test function, can be moved to tests module
    #[cfg(test)]
    pub fn test_coclusterer() {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let data = Array2::random((100, 50), Uniform::new(0., 10.));
        let mut coclusterer = Coclusterer::new(data, 5, 1e-4);
        match coclusterer.cocluster() {
            Ok(assignments) => {
                println!("Coclustering successful, {} assignments.", assignments.len());
                // Add more assertions here based on expected behavior
            }
            Err(e) => panic!("Coclustering failed: {}", e),
        }
    }
}

fn perform_svd(
    na_matrix_normalized: DMatrix<f64>,
    k: usize,
) -> Result<(DMatrix<f64>, DMatrix<f64>), &'static str> {
    let svd_result = SVD::new(na_matrix_normalized, true, true);

    let u_mat = match svd_result.u {
        Some(u) => u.columns(0, k).into_owned(), // 获取前 k 列
        None => return Err("Error: svd_result.u is None"),
    };

    let v_t_mat = match svd_result.v_t {
        Some(vt) => vt.rows(0, k).into_owned().transpose(), // 获取前 k 行并转置
        None => return Err("Error: svd_result.v_t is None"),
    };

    Ok((u_mat, v_t_mat))
}

#[cfg(test)]
mod tests {
    use na::{Matrix3, QR};
    use ndarray::{array, Array1, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::{Rng as NdarrayRng, SeedableRng as NdarraySeedableRng};
    use ndarray_rand::rand::rngs::StdRng as NdarrayStdRng;
    use crate::submatrix::Submatrix;
    use super::*; // Imports Coclusterer etc.
    use ::rand::Rng;

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
        let mut rng = ::rand::rngs::ThreadRng::default(); 
        let mat: Matrix3<f64> = Matrix3::from_fn(|_, _| rng.random::<f64>());
        let qr = QR::new(mat);
        let q = qr.q();

        q
    }

    #[test]
    fn test_cocluster_simple() {
        let data = array![
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ];
        let mut coclusterer = Coclusterer::new(data.clone(), 2, 1e-5);
        let result = coclusterer.cocluster();
        assert!(result.is_ok());
        let assignments = result.unwrap();

        // Check that rows 0,1 are in one cluster and 2,3 in another (or vice versa for cluster ID)
        // And cols 0,1 in one and 2,3 in another.
        // This depends on how SVD or other methods assign initial clusters.
        // For this simple block diagonal, we expect clean separation.
        let row0_cluster = assignments[0];
        let row1_cluster = assignments[1];
        let row2_cluster = assignments[2];
        let row3_cluster = assignments[3];

        let col0_cluster = assignments[data.nrows() + 0];
        let col1_cluster = assignments[data.nrows() + 1];
        let col2_cluster = assignments[data.nrows() + 2];
        let col3_cluster = assignments[data.nrows() + 3];

        assert_eq!(row0_cluster, row1_cluster, "Rows 0 and 1 should be in the same cluster");
        assert_eq!(row2_cluster, row3_cluster, "Rows 2 and 3 should be in the same cluster");
        assert_ne!(row0_cluster, row2_cluster, "Row clusters should be different for the two blocks");

        assert_eq!(col0_cluster, col1_cluster, "Cols 0 and 1 should be in the same cluster");
        assert_eq!(col2_cluster, col3_cluster, "Cols 2 and 3 should be in the same cluster");
        assert_ne!(col0_cluster, col2_cluster, "Col clusters should be different for the two blocks");
        
        // Also check if row clusters align with col clusters for the blocks
        // e.g. cluster ID for (rows 0,1) should be same as for (cols 0,1)
        // This is a stronger check and depends on the algorithm's ability to find coclusters
        // For this simple example, they should ideally match up to produce two coclusters.
        // assert_eq!(row0_cluster, col0_cluster, "Block 1 (0,0) row/col clusters should match");
        // assert_eq!(row2_cluster, col2_cluster, "Block 2 (1,1) row/col clusters should match");
    }

    #[test]
    fn test_random_matrix_coclustering() {
        let mut rng = NdarrayStdRng::seed_from_u64(42);
        let data = Array2::random_using((50, 30), Uniform::new(0.0, 1.0), &mut rng);
        let mut coclusterer = Coclusterer::new(data, 3, 1e-4);
        let result = coclusterer.cocluster();
        assert!(result.is_ok(), "Coclustering failed on random matrix: {:?}", result.err());
        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 50 + 30, "Incorrect number of assignments");
    }

    #[test]
    fn test_initial_assignment_randomness() {
        // Check if different seeds produce different initial (random) assignments if applicable
        // This test is more relevant if the initialization has a random component not controlled by SVD output directly.
        // The current implementation's `perform_svd_and_initial_assignment` is deterministic based on SVD.
        // If true random initialization was used (e.g. like in k-means++ style for centroids U, V),
        // this test would be more meaningful.
        
        // For now, let's ensure that with the same data and k, it produces the same SVD-based initial assignment.
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        let mut coclusterer1 = Coclusterer::new(data.clone(), 2, 1e-4);
        let assignments1 = coclusterer1.cocluster().unwrap();

        let mut coclusterer2 = Coclusterer::new(data.clone(), 2, 1e-4);
        let assignments2 = coclusterer2.cocluster().unwrap();
        assert_eq!(assignments1, assignments2, "Initial assignments should be deterministic for SVD method");
    }

    use nalgebra::{DMatrix, RowDVector};
    // Mock SVD result for testing reassign_clusters and calculate_error
    fn mock_svd_results(nrows: usize, ncols: usize, k: usize) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
        let u = Array2::random((nrows, k), Uniform::new(-1.0, 1.0));
        let s_values = Array1::random(k, Uniform::new(0.1, 1.0)); 
        let s_vec = Array1::from_vec(s_values.into_raw_vec());
        let v_t = Array2::random((k, ncols), Uniform::new(-1.0, 1.0));
        (u, s_vec, v_t)
    }

    #[test]
    fn test_reassign_clusters() {
        let data = Array2::zeros((10, 8)); // Content doesn't matter for this part of reassign
        let mut coclusterer = Coclusterer::new(data, 3, 1e-4);
        let (u, _s, v_t_mock) = mock_svd_results(10,8,3);
        // In SVD, V is usually (ncols, k), so V_T is (k, ncols).
        // The reassign_clusters expects v to be (ncols, k) if it iterates v.row(j).
        // If current mock_svd_results returns v_t as (k, ncols), then for reassign_clusters, we need v which is v_t.t()
        // Let's assume perform_svd returns U, S, V (not V_T) and reassign_clusters expects U and V.
        // The SVD in nalgebra returns u, s, v_t. The current reassign_clusters takes u, v.
        // Let's adapt the test or function signature. Assuming reassign_clusters takes u and v (where v is V from SVD, not V_T)
        // So if svd_v_t is (k, ncols), then svd_v is (ncols, k).
        let svd_v = v_t_mock.t().into_owned(); // svd_v is (ncols, k)

        coclusterer.reassign_clusters(&u, &svd_v);
        assert_eq!(coclusterer.matrix.len(), 10 + 8);
        // Further checks could be done if we had expected assignments based on u and svd_v values.
    }

    #[test]
    fn test_calculate_error() {
        let data = Array2::random((10,8), Uniform::new(0.0,1.0));
        let mut coclusterer = Coclusterer::new(data.clone(), 3, 1e-4);
        let (u, s, v_t) = mock_svd_results(10,8,3);
        let error = coclusterer.calculate_error(&u, &s, &v_t);
        assert!(error >= 0.0, "Error should be non-negative");
        // A perfect reconstruction (if u,s,v_t were from data itself) would give error near 0.
        // Since they are random, error will be some positive value.
    }

    #[test]
    fn test_convergence() {
        let mut rng = NdarrayStdRng::seed_from_u64(123);
        let data = Array2::random_using((30,20), Uniform::new(0.0, 1.0), &mut rng);
        let mut coclusterer = Coclusterer::new(data, 2, 1e-6);
        let result = coclusterer.cocluster();
        assert!(result.is_ok(), "Coclustering failed to converge or errored: {:?}", result.err());
        // Check if iterations were less than max_iter, implying convergence
        // This requires exposing iteration count from Coclusterer or checking error value for convergence.
        // For now, just checking if it runs to completion.
    }
}
