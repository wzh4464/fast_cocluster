/**
 * File: /src/cocluster.rs
 * Created Date: Thursday, June 13th 2024
 * Author: Zihan
 * -----
 * Last Modified: Tuesday, 18th June 2024 1:08:27 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/

// src/cocluster.rs

use na::{DMatrix, Dyn};
use ndarray::Array2;
extern crate nalgebra as na;
use std::fmt;

// use kmeans_smid
use kmeans_smid::{KMeans, KMeansConfig};

use crate::submatrix::Submatrix;

/// A co-clustering structure that performs k-means clustering on the rows and columns of a matrix.
pub struct Coclusterer {
    /// The matrix to be co-clustered.
    matrix: Array2<f32>,
    /// The number of rows in the matrix.
    row:    usize,
    /// The number of columns in the matrix.
    col:    usize,
    /// The number of row clusters.
    m:      usize,
    /// The number of column clusters.
    n:      usize,
    /// The tolerance for score.
    tol:    f32,
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
pub fn clone_to_dmatrix<T>(array_view: ndarray::ArrayView2<T>) -> DMatrix<T>
where
    T: Clone,
    T: na::Scalar,
{
    let nrows = array_view.nrows();
    let ncols = array_view.ncols();
    let elements = array_view.iter().cloned().collect::<Vec<T>>();
    DMatrix::from_vec(nrows, ncols, elements)
}

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
    pub fn new(matrix: Array2<f32>, m: usize, n: usize, tol: f32) -> Coclusterer {
        let row = matrix.shape()[0];
        let col = matrix.shape()[1];
        Coclusterer {
            matrix,
            m,
            n,
            row,
            col,
            tol,
        }
    }

    /// Performs co-clustering on the matrix and returns a list of submatrices that meet the tolerance criteria.
    ///
    /// # Returns
    ///
    /// * A vector of `Submatrix` objects that meet the tolerance criteria.
    pub(crate) fn cocluster(&mut self) -> Vec<Submatrix<f32>> {
        // svd to get u,s,v
        let na_matrix = clone_to_dmatrix(self.matrix.view());
        let svd_result = na_matrix.svd(true, true);
        let u: na::Matrix<f32, Dyn, Dyn, na::VecStorage<f32, Dyn, Dyn>> = svd_result.u.unwrap(); // shaped as (row, row)
        let vt: na::Matrix<f32, Dyn, Dyn, na::VecStorage<f32, Dyn, Dyn>> = svd_result.v_t.unwrap(); // shaped as (col, col)
        let v: na::Matrix<f32, Dyn, Dyn, na::VecStorage<f32, Dyn, Dyn>> = vt.transpose(); // shaped as (col, row)

        let u_data = u.data.as_vec().clone();
        let kmeans_u: KMeans<f32, 8> = KMeans::new(u_data, self.row, self.matrix.shape()[0]);
        // there is an assert for samples.len() == sample_cnt * sample_dims
        let result_u = kmeans_u.kmeans_lloyd(
            self.m,
            100,
            KMeans::init_kmeanplusplus,
            &KMeansConfig::default(),
        );

        // 对 v 应用 K-means
        let v_data = v.data.as_vec().clone();
        let kmeans_v: KMeans<f32, 8> = KMeans::new(v_data, self.col, self.matrix.shape()[1]);
        let result_v = kmeans_v.kmeans_lloyd(
            self.n,
            100,
            KMeans::init_kmeanplusplus,
            &KMeansConfig::default(),
        );

        // generate submatrix list, keep score < tol
        let mut submatrix_list: Vec<Submatrix<f32>> = Vec::new();
        // use pipe
        for i in 0..self.m {
            for j in 0..self.n {
                let mut row_indices: Vec<usize> = Vec::new();
                let mut col_indices: Vec<usize> = Vec::new();
                for k in 0..self.row {
                    if result_u.assignments[k] == i {
                        row_indices.push(k);
                    }
                }
                for k in 0..self.col {
                    if result_v.assignments[k] == j {
                        col_indices.push(k);
                    }
                }
                let submatrix = Submatrix::new(&self.matrix, row_indices, col_indices);
                match submatrix {
                    Some(submatrix) => {
                        match submatrix.score {
                            Some(score) => {
                                if score < self.tol {
                                    submatrix_list.push(submatrix);
                                }
                            }
                            None => {}
                        }
                    }
                    None => {}
                }
            }
        }

        return submatrix_list;
    }

}

#[cfg(test)]
mod tests {
    use na::{Matrix3, QR};
    use ndarray::Array2;
    use rand::{random, Rng};

    use crate::submatrix::Submatrix;

    use super::*;

    #[test]
    /// Test for updating the score of a submatrix.
    fn test_update_score() {
        // submatrix is smaller than 3 *. 3
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut submatrix = Submatrix::new(&data, vec![0, 1], vec![0, 1]);
        
        if let Some(mut submatrix) = submatrix {
            submatrix.update_score();
            assert_eq!(submatrix.score, None);
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

        let matrix = Array2::<f32>::from_shape_vec((3, 3), a_vec).unwrap();
        let mut submatrix = Submatrix::new(
            &matrix,
            vec![0, 1, 2],
            vec![0, 1, 2],
        );

        if let Some(ref mut submatrix) = submatrix {
            submatrix.update_score();
        }
        
        let score = submatrix.unwrap().score.unwrap();

        println!("score: {}", &score);

        assert!((score - 2.0).abs() < 1e-6);
    }

    /// Generates a random orthogonal matrix.
    ///
    /// # Returns
    ///
    /// * A random 3x3 orthogonal matrix.
    fn random_orthogonal_matrix() -> Matrix3<f32> {
        let mut rng = rand::thread_rng();

        // 随机生成一个 3x3 矩阵
        let mat: Matrix3<f32> = Matrix3::from_fn(|_, _| rng.gen::<f32>());

        // 进行 QR 分解
        let qr = QR::new(mat);
        let q = qr.q();

        q
    }
}
