/*
 * File: /cocluster.rs
 * Created Date: Tuesday November 21st 2023
 * Author: Zihan
 * -----
 * Last Modified: Tuesday, 21st November 2023 1:52:06 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

// 在 cocluster.rs 文件中

use ndarray::Array2;
extern crate nalgebra as na;

// use kmeans
use kmeans::{KMeans, KMeansConfig};

pub struct Coclusterer {
    // 字段定义
    // need a matrix to init, float
    matrix: Array2<f32>,
    // shape of matrix
    row: usize,
    col: usize,
    // m,n to save cluster number for rows and columns
    m: usize,
    n: usize,
    // torlerance
    tol: f32,
}

impl Coclusterer {
    // 方法实现
    // 构造函数
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

    // k-means for rows and columns
    pub(crate) fn cocluster(&mut self) -> Vec<Submatrix> {
        // svd to get u,s,v
        let na_matrix: na::Matrix<f32, na::Dynamic, na::Dynamic, na::VecStorage<f32, na::Dynamic, na::Dynamic>> = na::DMatrix::from_row_slice(
            self.matrix.shape()[0],
            self.matrix.shape()[1],
            self.matrix.as_slice().unwrap(),
        );
        let svd_result = na_matrix.svd(true, true);
        let u: na::Matrix<f32, na::Dynamic, na::Dynamic, na::VecStorage<f32, na::Dynamic, na::Dynamic>> = svd_result.u.unwrap(); // shaped as (row, row)
        let vt: na::Matrix<f32, na::Dynamic, na::Dynamic, na::VecStorage<f32, na::Dynamic, na::Dynamic>> = svd_result.v_t.unwrap(); // shaped as (col, col)
        let v: na::Matrix<f32, na::Dynamic, na::Dynamic, na::VecStorage<f32, na::Dynamic, na::Dynamic>> = vt.transpose(); // shaped as (col, row)

        let u_data = u.data.as_vec().clone();
        let kmeans_u = KMeans::new(u_data, self.row, self.m);
        let result_u = kmeans_u.kmeans_lloyd(self.m, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());

        // 对 v 应用 K-means
        let v_data = v.data.as_vec().clone();
        let kmeans_v = KMeans::new(v_data, self.col, self.n);
        let result_v = kmeans_v.kmeans_lloyd(self.n, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());

        // generate submatrix list, keep score < tol
        let mut submatrix_list: Vec<Submatrix> = Vec::new();
        // use pipe
        for i in 0..self.m {
            for j in 0..self.n {
                let mut row_index: Vec<usize> = Vec::new();
                let mut col_index: Vec<usize> = Vec::new();
                for k in 0..self.row {
                    if result_u.assignments[k] == i {
                        row_index.push(k);
                    }
                }
                for k in 0..self.col {
                    if result_v.assignments[k] == j {
                        col_index.push(k);
                    }
                }
                let submatrix = Submatrix::new(self.matrix.clone(), row_index, col_index);
                if submatrix.score < self.tol {
                    submatrix_list.push(submatrix);
                }
            }
        }

        return submatrix_list;
    }
}

pub struct Submatrix {
    matrix: Array2<f32>,
    // vector of row index
    row_index: Vec<usize>,
    // vector of col index
    col_index: Vec<usize>,
    // boolean vector of row index
    score: f32,
}

impl Submatrix {
    pub fn new(matrix: Array2<f32>, row_index: Vec<usize>, col_index: Vec<usize>) -> Submatrix {
        let score = 0.0;
        let mut new_obj = Submatrix {
            matrix,
            row_index,
            col_index,
            score,
        };
        new_obj.update_score();
        new_obj
    }

    pub fn update_score(&mut self) {
        // calculate svd and get first two singular values
        let omatrix = na::DMatrix::from_row_slice(
            self.matrix.shape()[0],
            self.matrix.shape()[1],
            self.matrix.as_slice().unwrap(),
        );
        let submatrix = omatrix.select_rows(self.row_index.as_slice()).select_columns(self.col_index.as_slice());
        let svd = submatrix.svd(true, true);
        let s1 = svd.singular_values[0];
        let s2 = svd.singular_values[1];
        // calculate score
        self.score = s2 / s1;
    }
}
