/*
 * File: /main.rs
 * Created Date: Tuesday November 21st 2023
 * Author: Zihan
 * -----
 * Last Modified: Tuesday, 21st November 2023 1:09:55 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

mod cocluster;

use cocluster::Coclusterer;

extern crate ndarray;
extern crate ndarray_parallel;
extern crate rayon;

use ndarray::{Array, Array2, Axis};
use ndarray_parallel::par_azip;

fn main() {
    // 创建两个 3x3 的矩阵
    let a = Array::from_shape_vec((3, 3), (0..9).collect()).unwrap();
    let b = Array::from_shape_vec((3, 3), (0..9).collect()).unwrap();

    // 并行计算矩阵乘法
    let result = matrix_multiply_parallel(&a, &b);
    println!("Result:\n{}", result);
}

// 并行矩阵乘法函数
fn matrix_multiply_parallel(a: &Array2<i32>, b: &Array2<i32>) -> Array2<i32> {
    let n = a.len_of(Axis(0));
    let p = b.len_of(Axis(1));

    // 初始化结果矩阵
    let result = Array2::zeros((n, p));

    // 使用 par_azip! 宏进行并行操作
    par_azip!((index (i, j), result in &mut result) {
        result[index] = (0..n).map(|k| a[(i, k)] * b[(k, j)]).sum();
    });

    result
}
