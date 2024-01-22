#![allow(unused)]

/**
 * File: /src/main.rs
 * Created Date: Friday January 12th 2024
 * Author: Zihan
 * -----
 * Last Modified: Monday, 22nd January 2024 4:17:55 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/
mod cocluster;
pub mod matrix;
use cocluster::Coclusterer;
mod submatrix;
use submatrix::Submatrix;

mod union_dc;
use union_dc::UnionDC;

use ndarray::{s, Array, Array2, Ix2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use std::collections::VecDeque;
#[cfg(debug_assertions)]
use std::time::Instant;

mod parallel_cocluster;

// test Coclusterer

fn select_non_contiguous_slices(
    matrix: &Array2<i32>,
    row_indices: &[usize],
    col_indices: &[usize],
) -> Array2<i32> {
    let mut selected_data = Vec::new();
    for &row_idx in row_indices {
        for &col_idx in col_indices {
            selected_data.push(matrix[[row_idx, col_idx]]);
        }
    }
    Array::from_shape_vec((row_indices.len(), col_indices.len()), selected_data).unwrap()
}

fn main() {
    let a = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
    let submatrix = Submatrix::from_indices(&a, &[0, 2], &[1, 2]); 
    let submatrix = submatrix.unwrap();

    // 访问子矩阵的元素
    println!("{:?}", &submatrix.get(1, 0).unwrap()); // 使用 get 方法访问元素
    println!("{:?}", &submatrix[(1, 0)]); // 使用索引访问元素

    println!("{}", submatrix);
}

fn example_for_ndarray() {
    let A_mat = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
    let B_mat = A_mat.slice(s![..=1, ..=1]);
    println!("{:?}", B_mat);
}

fn example_for_uniondc() {
    // generate a queue with integers [3,5,1,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]
    let mut queue = VecDeque::from([3, 5, 1, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    // define a function to add two integers
    let func = |a: i32, b: i32| a + b;
    // define a function to check if the queue has only one element
    // let end_condition = |queue: &VecDeque<i32>| queue.len() < 2;
    let end_condition = |queue: &VecDeque<i32>| queue.len() < 2;

    // call UnionDC
    UnionDC::union_dc(&mut queue, func, &end_condition);

    // print result
    println!("{:?}", queue);
}

fn test_cocluster() {
    // init a constant matrix : Array2<f32>
    // rand 200 * 200
    let test_matrix: Array2<f32> = Array2::random((200, 200), Uniform::new(0.0, 1.0));

    // println!("{:?}", test_matrix);

    let mut coclusterer = Coclusterer::new(test_matrix, 3, 3, 1e-1);
    #[cfg(debug_assertions)]
    let start_time = Instant::now();

    println!("{:?}", coclusterer.cocluster());
    #[cfg(debug_assertions)]
    {
        // count time end
        let end_time = Instant::now();
        println!("Time cost: {:?}", end_time.duration_since(start_time));
    }
}
