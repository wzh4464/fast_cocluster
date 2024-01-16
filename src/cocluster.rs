/**
 * File: /src/cocluster.rs
 * Created Date: Thursday December 28th 2023
 * Author: Zihan
 * -----
 * Last Modified: Tuesday, 16th January 2024 4:22:01 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/


use na::Dyn;
use ndarray::Array2;
extern crate nalgebra as na;
use std::fmt;

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
        let na_matrix: na::Matrix<
            f32,
            Dyn,
            Dyn,
            na::VecStorage<f32, Dyn, Dyn>,
        > = na::DMatrix::from_row_slice(
            self.matrix.shape()[0],
            self.matrix.shape()[1],
            self.matrix.as_slice().unwrap(),
        );
        let svd_result = na_matrix.svd(true, true);
        let u: na::Matrix<
            f32,
            Dyn,
            Dyn,
            na::VecStorage<f32, Dyn, Dyn>,
        > = svd_result.u.unwrap(); // shaped as (row, row)
        let vt: na::Matrix<
            f32,
            Dyn,
            Dyn,
            na::VecStorage<f32, Dyn, Dyn>,
        > = svd_result.v_t.unwrap(); // shaped as (col, col)
        let v: na::Matrix<
            f32,
            Dyn,
            Dyn,
            na::VecStorage<f32, Dyn, Dyn>,
        > = vt.transpose(); // shaped as (col, row)

        let u_data = u.data.as_vec().clone();
        let kmeans_u = KMeans::new(u_data, self.row, self.matrix.shape()[0]);
        // there is an assert for samples.len() == sample_cnt * sample_dims
        let result_u = kmeans_u.kmeans_lloyd(
            self.m,
            100,
            KMeans::init_kmeanplusplus,
            &KMeansConfig::default(),
        );

        // 对 v 应用 K-means
        let v_data = v.data.as_vec().clone();
        let kmeans_v = KMeans::new(v_data, self.col, self.matrix.shape()[1]);
        let result_v = kmeans_v.kmeans_lloyd(
            self.n,
            100,
            KMeans::init_kmeanplusplus,
            &KMeansConfig::default(),
        );

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
        // if submatrix is smaller than 3*3, score gives an inf
        let row_len = self.row_index.len();
        let col_len = self.col_index.len();
        if row_len < 3 || col_len < 3 {
            self.score = f32::INFINITY;
            return;
        }

        // calculate svd and get first two singular values
        let omatrix = na::DMatrix::from_row_slice(
            self.matrix.shape()[0],
            self.matrix.shape()[1],
            self.matrix.as_slice().unwrap(),
        );
        let submatrix = omatrix
            .select_rows(self.row_index.as_slice())
            .select_columns(self.col_index.as_slice());
        let svd = submatrix.svd(true, true);
        let s1 = svd.singular_values[0];
        let s2 = svd.singular_values[1];
        // calculate score
        self.score = s2 / s1;
    }
}

impl fmt::Display for Submatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Submatrix:\nMatrix:\n{:?}\nRow Index: {:?}\nColumn Index: {:?}\nScore: {}",
            self.matrix, self.row_index, self.col_index, self.score
        )
    }
}

impl fmt::Debug for Submatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Debug 实现通常类似于 Display，但可能包含更多详细信息
        // 这里只是简单复用了 Display 实现
        write!(f, "{}", self)
    }
}
