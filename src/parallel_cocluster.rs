/**
 * File: /src/parallel_cocluster.rs
 * Created Date: Monday January 22nd 2024
 * Author: Zihan
 * -----
 * Last Modified: Monday, 22nd January 2024 1:11:30 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/
// Array2
use ndarray::Array2;

// Coclusterer
use crate::cocluster;
use cocluster::Coclusterer;

// matrix
use crate::matrix;
use matrix::Matrix;

struct ParallelCoclusterer {
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
    coclusterer: Coclusterer,
}

// impl ParallelCoclusterer {
//     fn new(matrix : Matrix, m: usize, n: usize, tol: f32) -> ParallelCoclusterer {
//         let row = matrix.rows;
//         let col = matrix.cols;
//         let coclusterer = Coclusterer::new(matrix, m, n, tol);
//         ParallelCoclusterer { matrix, row, col, m, n, tol, coclusterer }
//     }
// }