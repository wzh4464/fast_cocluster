/**
 * File: /src/cocluster.rs
 * Created Date: Thursday, June 13th 2024
 * Author: Zihan
 * -----
 * Last Modified: Tuesday, 18th June 2024 11:26:25 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 18-06-2024		Zihan	Refactor Coclusterer and Add Utility Functions
**/
// src/cocluster.rs
use nalgebra::{DMatrix, DVector, Dyn, Matrix2};
use ndarray::{stack, Array2};
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
    matrix: Array2<f32>,
    /// The number of rows in the matrix.
    row: usize,
    /// The number of columns in the matrix.
    col: usize,
    /// The number of row clusters.
    m: usize,
    /// The number of column clusters.
    n: usize,
    /// The tolerance for score.
    tol: f32,
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

        let svd_result = na_matrix_normalized.svd(true, true);
        // let u: na::Matrix<f32, Dyn, Dyn, na::VecStorage<f32, Dyn, Dyn>> = svd_result.u.unwrap(); // shaped as (row, row)
        // let vt: na::Matrix<f32, Dyn, Dyn, na::VecStorage<f32, Dyn, Dyn>> = svd_result.v_t.unwrap(); // shaped as (col, col)
        // let v: na::Matrix<f32, Dyn, Dyn, na::VecStorage<f32, Dyn, Dyn>> = vt.transpose(); // shaped as (col, row)

        // u, v 取前 self.k 列, 然后 vstack 成 f
        let k = max(self.m, self.n);

        // let u_mat = svd_result.u.unwrap(); // 创建一个更长生命周期的变量
        // let v_t = svd_result.v_t.unwrap().transpose(); // 创建一个更长生命周期的变量
        let (u_mat, v_t_mat) = match (svd_result.u, svd_result.v_t) {
            (Some(u_mat), Some(v_t_mat)) => (u_mat, v_t_mat.transpose()),
            _ => {
                // println!("Error: svd_result.u or svd_result.v_t is None");
                return Err("Error: svd_result.u or svd_result.v_t is None");
            }
        };

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

        // println!("f: \n{}", f);
        let f_data: Vec<f32> = f.transpose().data.as_slice().iter().copied().collect();
        // println!("f_data: \n{:?}", f_data);
        let kmeans_f: KMeans<f32, 8> = KMeans::new(f_data, f.nrows(), f.ncols());

        let result_f =
            kmeans_f.kmeans_lloyd(k, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());

        // print!("result_f: \n");
        // for i in result_f.assignments.iter() {
        //     print!("{:?} ", i);
        // }
        // println!();

        return Ok(result_f.assignments);
    }
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
        let mut submatrix = Submatrix::new(&matrix, vec![0, 1, 2], vec![0, 1, 2]);

        if let Some(ref mut submatrix) = submatrix {
            submatrix.update_score();
        }

        let score = submatrix.unwrap().score.unwrap();

        println!("score: {}", &score);

        assert!((score - 2.0).abs() < 1e-6);
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
            success_rate >= 90.0,
            "Success rate is below acceptable threshold: {:.2}%",
            success_rate
        );
    }
    fn cocluster_test_helper(
        b_matrix: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
    ) -> Result<(), String> {
        let mut coclusterer = Coclusterer::new(b_matrix, 3, 3, 1e-1);
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
