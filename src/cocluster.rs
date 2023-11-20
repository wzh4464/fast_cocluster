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
        Coclusterer {
            matrix: matrix,
            m: m,
            n: n,
            tol: tol,
            row: matrix.shape()[0],
            col: matrix.shape()[1],
        }
    }

    // k-means for rows and columns
    #[allow(dead_code)]
    pub fn cocluster(&mut self) -> Cocluster {
        // svd to get u,s,v
        let na_matrix = na::DMatrix::from_row_slice(
            self.matrix.shape()[0],
            self.matrix.shape()[1],
            &self.matrix.into_raw_vec(),
        );
        let svd_result = na_matrix.svd(true, true);
        let u = svd_result.u.unwrap(); // shaped as (row, row)
        let vt = svd_result.v_t.unwrap(); // shaped as (col, col)
        let v = vt.transpose(); // shaped as (col, col)

        // do k-means for rows and columns

        return Cocluster {
            // all 1
            row_cluster: vec![1; self.row],
            col_cluster: vec![1; self.col],
            matrix: self.matrix.clone(),
            row: self.row,
            col: self.col,
            m: self.m,
            n: self.n,
        };
    }
}

struct Cocluster {
    row_cluster: Vec<usize>,
    col_cluster: Vec<usize>,
    matrix: Array2<f32>,
    row: usize,
    col: usize,
    m: usize,
    n: usize,
}

impl Cocluster {
    pub fn new(matrix: Array2<f32>, m: usize, n: usize) -> Cocluster {
        let row = matrix.shape()[0];
        let col = matrix.shape()[1];
        Cocluster {
            matrix,
            m,
            n,
            row,
            col,
            row_cluster: vec![0; row],
            col_cluster: vec![0; col],
        }
    }
}

struct Submatrix {
    matrix: Array2<f32>,
    // vector of row index
    row_index: Vec<usize>,
    // vector of col index
    col_index: Vec<usize>,
    // boolean vector of row index
    row_index_bool: Vec<bool>,
    // boolean vector of col index
    col_index_bool: Vec<bool>,
    score: f32,
}

impl Submatrix {
    pub fn new(matrix: Array2<f32>, row_index: Vec<usize>, col_index: Vec<usize>) -> Submatrix {
        let row_index_bool = vec![false; matrix.shape()[0]];
        let col_index_bool = vec![false; matrix.shape()[1]];
        let mut score = 0.0;
        let mut new_obj = Submatrix {
            matrix,
            row_index,
            col_index,
            row_index_bool,
            col_index_bool,
            score,
        };
        new_obj.update_score();
        new_obj
    }

    pub fn update_score(&mut self) {
        // calculate svd and get first two singular values
        let submatrix = na::DMatrix::from_row_slice(
            self.matrix.shape()[0],
            self.matrix.shape()[1],
            &self.matrix.into_raw_vec(),
        );
        let svd = submatrix.svd(true, true);
        let s1 = svd.singular_values[0];
        let s2 = svd.singular_values[1];
        // calculate score
        self.score = s2 / s1;
    }
}
