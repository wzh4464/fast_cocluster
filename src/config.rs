use std::fs::File;

/**
 * File: /src/config.rs
 * Created Date: Friday, January 26th 2024
 * Author: Zihan
 * -----
 * Last Modified: Friday, 26th January 2024 10:00:00 pm
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
 */
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
pub struct Config {
    // 字段定义
    // need a matrix to init, float
    matrix: Array2<f32>,
    // shape of matrix
    row:    usize,
    col:    usize,
    // m,n to save cluster number for rows and columns
    m:      usize,
    n:      usize,
    // torlerance
    tol:    f32,
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
        let matrix = Array2::<f32>::read_npy(reader)?;
        let m = args.next().unwrap().parse::<usize>()?;
        let n = args.next().unwrap().parse::<usize>()?;
        let tol = args.next().unwrap().parse::<f32>()?;
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

    pub fn get_matrix(&self) -> &Array2<f32> {
        &self.matrix
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn get_n(&self) -> usize {
        self.n
    }

    pub fn get_tol(&self) -> f32 {
        self.tol
    }

    pub fn get_row(&self) -> usize {
        self.row
    }

    pub fn get_col(&self) -> usize {
        self.col
    }
}

// cargo run -- /home/zihan/amazon_data/feature_matrix.npy 4 5 1e-4
// Running `target/debug/fast_cocluster /home/zihan/amazon_data/feature_matrix.npy 4 5 1e-4`
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_config() {
        let args = vec![
            "target/debug/fast_cocluster".to_string(),
            "/home/zihan/amazon_data/feature_matrix.npy".to_string(),
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