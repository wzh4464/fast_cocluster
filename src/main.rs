#![allow(unused)]

use std::env::args;

/**
 * File: /src/main.rs
 * Created Date: Friday, January 12th 2024
 * Author: Zihan
 * -----
 * Last Modified: Friday, 26th January 2024 9:37:40 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */

mod config;

fn main() {
    let config = config::Config::new(args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        std::process::exit(1);
    });
}
