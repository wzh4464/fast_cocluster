/*
 * File: /main.rs
 * Created Date: Tuesday November 21st 2023
 * Author: Zihan
 * -----
 * Last Modified: Wednesday, 22nd November 2023 11:34:08 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

mod cocluster;
use cocluster::Coclusterer;

mod union_dc;
use union_dc::UnionDC;

use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use std::collections::VecDeque;
#[cfg(debug_assertions)]
use std::time::Instant;

// test Coclusterer

fn main() {
    example_for_uniondc();
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
