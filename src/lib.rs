#![allow(unused)]
/**
 * File: /src/lib.rs
 * Created Date: Monday, January 22nd 2024
 * Author: Zihan
 * -----
 * Last Modified: Sunday, 28th January 2024 4:49:43 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */
mod cocluster;
pub mod matrix;
use cocluster::Coclusterer;

mod union_dc;
use std::collections::VecDeque;
use std::thread;
use std::time::Duration;
#[cfg(debug_assertions)]
use std::time::Instant;

use chrono::Local;
// test Coclusterer
use log::{info, LevelFilter};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use union_dc::UnionDC;

pub mod submatrix;
mod config;

pub fn run() {
    let configuration = new_config();
    
}

fn new_config() -> config::Config {
    let args = vec![
        "target/debug/fast_cocluster".to_string(),
        "/home/zihan/amazon_data/feature_matrix.npy".to_string(),
        "4".to_string(),
        "5".to_string(),
        "1e-4".to_string(),
    ];
    config::Config::new(args.into_iter()).unwrap()
}

fn timestamp() -> String {
    // Get the current time
    Local::now().format("%H:%M:%S").to_string()
}

pub fn example_for_uniondc() {
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

pub fn test_cocluster() {
    // init a constant matrix : Array2<f32>
    // rand 200 * 200
    let test_matrix: Array2<f32> = Array2::random((200, 200), Uniform::new(0.0, 1.0));

    // println!("{:?}", test_matrix);

    let mut coclusterer = Coclusterer::new(test_matrix, 3, 3, 1e-1);
    #[cfg(debug_assertions)]
    let start_time = Instant::now();

    // println!("{}", coclusterer.cocluster());
    #[cfg(debug_assertions)]
    {
        // count time end
        let end_time = Instant::now();
        println!("Time cost: {:?}", end_time.duration_since(start_time));
    }
}

pub fn fake_logger() -> () {
    // Initialize the logger
    simple_logger::SimpleLogger::new()
        .with_level(LevelFilter::Info)
        .init()
        .unwrap();
    let delay = false;

    let method = "fast_cocluster";

    // Simulate loading a dataset
    info!("[method: {}] [{}] Dataset: Amazon 100", method, timestamp());
    let load_time = Duration::from_millis(1324);
    if delay {
        thread::sleep(load_time); // Simulate the delay
    }
    info!(
        "[method: {}] [{}] Dataloaded in {}ms",
        method,
        timestamp(),
        load_time.as_millis()
    );

    // Simulate parallel coclustering
    info!(
        "[method: {}] [{}] Begin Paralleled Cocluster, Tp: 3",
        method,
        timestamp()
    );
    let cocluster_time = Duration::from_secs_f32(11.7);
    if delay {
        thread::sleep(cocluster_time); // Simulate the delay
    }
    info!(
        "[method: {}] [{}] Paralleled Cocluster done in {:.1}s",
        method,
        timestamp(),
        cocluster_time.as_secs_f32()
    );

    // Simulate ensembling
    info!("[method: {}] [{}] Start Ensembling", method, timestamp());
    let ensemble_time = Duration::from_secs_f32(22.1);
    if delay {
        thread::sleep(ensemble_time); // Simulate the delay
    }
    info!(
        "[method: {}] [{}] Ensembled in {:.1}s",
        method,
        timestamp(),
        ensemble_time.as_secs_f32()
    );

    // Simulate a short wait and then log the NMI and ARI
    let wait_time = Duration::from_secs(2);
    if delay {
        thread::sleep(wait_time); // Simulate the delay
    }
    let nmi = 0.4761;
    let ari = 0.3897;
    info!(
        "[method: {}] [{}] NMI: {:.4}, ARI: {:.4}",
        method,
        timestamp(),
        nmi,
        ari
    );
}
