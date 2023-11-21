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


use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// test Coclusterer

fn main() {
    // init a constant matrix : Array2<f32>
    // rand 200 * 200
    let test_matrix: Array2<f32> = Array2::random((200, 200), Uniform::new(0.0, 1.0));

    println!("{:?}", test_matrix);

    let mut coclusterer = Coclusterer::new(test_matrix, 3, 3, 1e-4);

    coclusterer.cocluster();
}
