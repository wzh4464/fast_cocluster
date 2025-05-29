use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
/**
 * File: /src/config.rs
 * Created Date: Friday, January 26th 2024
 * Author: Zihan
 * -----
 * Last Modified: Thursday, 29th May 2025 10:59:30 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 * 29-01-2024		Zihan	change the config tests
 */
use std::fs::File;
pub struct Config {
    // 字段定义
    // need a matrix to init, float
    matrix: Array2<f64>,
    // shape of matrix
    row: usize,
    col: usize,
    // m,n to save cluster number for rows and columns
    m: usize,
    n: usize,
    // torlerance
    tol: f64,
}

pub struct DiMergeCoConfig {
    pub t_m: usize,
    pub t_n: usize,
    pub t_max: usize,
    pub p_thresh: f64,
    pub overlap_threshold: f64,
    pub score_weights: (f64, f64, f64), // (coherence, density, size) - Added from HierarchicalMerger
    pub parallel: bool,
    pub local_k: usize,
    pub local_tol: f64,
}

impl Config {
    /// constructor
    ///
    /// # Examples
    /// ```bash
    /// $ cargo run -- "data/matrix.npy" 2 2 0.1
    pub fn new(
        mut args: impl Iterator<Item = String>,
    ) -> Result<Config, Box<dyn std::error::Error>> {
        // read args
        // args:
        // 0: program name
        // 1: matrix path
        // 2: m
        // 3: n
        // 4: tol
        // read from npy file
        args.next();
        let reader = File::open(args.next().unwrap())?;
        let matrix = Array2::<f64>::read_npy(reader)?;
        let m = args.next().unwrap().parse::<usize>()?;
        let n = args.next().unwrap().parse::<usize>()?;
        let tol = args.next().unwrap().parse::<f64>()?;
        let row = matrix.shape()[0];
        let col = matrix.shape()[1];

        Ok(Config {
            matrix,
            m,
            n,
            tol,
            row,
            col,
        })
    }

    pub fn get_matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn get_n(&self) -> usize {
        self.n
    }

    pub fn get_tol(&self) -> f64 {
        self.tol
    }

    pub fn get_row(&self) -> usize {
        self.row
    }

    pub fn get_col(&self) -> usize {
        self.col
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_config() {
        // make npy file at target/debug/fast_cocluster/test.npy
        let file = File::create("target/debug/test.npy").unwrap();
        ndarray_npy::WriteNpyExt::write_npy(&ndarray::Array2::<f64>::zeros((24000, 768)), file)
            .unwrap();
        let args = vec![
            "target/debug/fast_cocluster".to_string(),
            "target/debug/test.npy".to_string(),
            "4".to_string(),
            "5".to_string(),
            "1e-4".to_string(),
        ];
        let config = Config::new(args.into_iter()).unwrap();
        assert_eq!(config.m, 4);
        assert_eq!(config.n, 5);
        assert_eq!(config.tol, 1e-4);
        assert_eq!(config.row, 24000);
        assert_eq!(config.col, 768);

        // get methods
        assert_eq!(config.get_m(), 4);
        assert_eq!(config.get_n(), 5);
        assert_eq!(config.get_tol(), 1e-4);
        assert_eq!(config.get_row(), 24000);
        assert_eq!(config.get_col(), 768);
    }
}
