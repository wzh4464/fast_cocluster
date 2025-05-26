// use crate::{Matrix, Submatrix};
use crate::submatrix::Submatrix;
use crate::matrix::Matrix;
/**
 * File: ./src/scoring.rs
 * Created Date: Monday, May 26th 2025
 * Author: Zihan
 * -----
 * Last Modified: Monday, 26th May 2025 11:43:22 am
 * Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 * -----
 * HISTORY:
 * Date      		By   	Comments
 * ----------		------	---------------------------------------------------------
**/
// src/scoring.rs
use nalgebra::DMatrix;
use statrs::statistics::Statistics;
use ndarray::Array2;

/// 评分器trait，所有评分方法都需要实现这个trait
pub trait Scorer: Send + Sync {
    fn score<'a>(
        &self,
        matrix: &'a Matrix<f64>,
        submatrix: &Submatrix<'a, f64>,
    ) -> f64;

    fn score_all<'a>(
        &self,
        matrix: &'a Matrix<f64>,
        submatrices: &[Submatrix<'a, f64>],
    ) -> Vec<f64> {
        submatrices
            .iter()
            .map(|sub| self.score(matrix, sub))
            .collect()
    }
}

/// Pearson相关系数评分器
pub struct PearsonScorer {
    pub min_size: (usize, usize),
}

impl PearsonScorer {
    pub fn new(min_rows: usize, min_cols: usize) -> Self {
        Self {
            min_size: (min_rows, min_cols),
        }
    }

    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x == 0.0 || var_y == 0.0 {
            return 0.0;
        }

        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

impl Scorer for PearsonScorer {
    fn score<'a>(
        &self,
        matrix: &'a Matrix<f64>,
        submatrix: &Submatrix<'a, f64>,
    ) -> f64 {
        // 检查最小尺寸
        if submatrix.row_indices.len() < self.min_size.0 || submatrix.col_indices.len() < self.min_size.1 {
            return 0.0;
        }

        let data_array2 = &matrix.data; // This is Array2<f64>
        let mut correlations = Vec::new();

        // 计算行之间的相关性
        for i in 0..submatrix.row_indices.len() {
            for j in i + 1..submatrix.row_indices.len() {
                let row1_idx = submatrix.row_indices[i];
                let row2_idx = submatrix.row_indices[j];

                let values1: Vec<f64> = submatrix
                    .col_indices
                    .iter()
                    .map(|&col| data_array2[(row1_idx, col)])
                    .collect();
                let values2: Vec<f64> = submatrix
                    .col_indices
                    .iter()
                    .map(|&col| data_array2[(row2_idx, col)])
                    .collect();

                let corr = Self::pearson_correlation(&values1, &values2);
                correlations.push(corr.abs());
            }
        }

        // 计算列之间的相关性
        for i in 0..submatrix.col_indices.len() {
            for j in i + 1..submatrix.col_indices.len() {
                let col1_idx = submatrix.col_indices[i];
                let col2_idx = submatrix.col_indices[j];

                let values1: Vec<f64> = submatrix
                    .row_indices
                    .iter()
                    .map(|&row| data_array2[(row, col1_idx)])
                    .collect();
                let values2: Vec<f64> = submatrix
                    .row_indices
                    .iter()
                    .map(|&row| data_array2[(row, col2_idx)])
                    .collect();

                let corr = Self::pearson_correlation(&values1, &values2);
                correlations.push(corr.abs());
            }
        }

        // 返回平均相关性
        if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        }
    }
}

/// 指数评分器
pub struct ExponentialScorer {
    pub tau: f64,
}

impl ExponentialScorer {
    pub fn new(tau: f64) -> Self {
        Self { tau }
    }
}

impl Scorer for ExponentialScorer {
    fn score<'a>(
        &self,
        matrix: &'a Matrix<f64>,
        submatrix: &Submatrix<'a, f64>,
    ) -> f64 {
        let data_array2 = &matrix.data;
        let mut sum = 0.0;
        let mut count = 0;

        // 计算子矩阵中所有值的指数和
        for &row in &submatrix.row_indices {
            for &col in &submatrix.col_indices {
                sum += (data_array2[(row, col)] / self.tau).exp();
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            (sum / count as f64).ln() * self.tau
        }
    }
}

/// 兼容性评分器
pub struct CompatibilityScorer {
    pub row_threshold: f64,
    pub col_threshold: f64,
}

impl CompatibilityScorer {
    pub fn new(row_threshold: f64, col_threshold: f64) -> Self {
        Self {
            row_threshold,
            col_threshold,
        }
    }

    fn calculate_variance(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        variance
    }
}

impl Scorer for CompatibilityScorer {
    fn score<'a>(
        &self,
        matrix: &'a Matrix<f64>,
        submatrix: &Submatrix<'a, f64>,
    ) -> f64 {
        let data_array2 = &matrix.data;

        // 计算行的兼容性
        let mut row_variances = Vec::new();
        for &row_idx in &submatrix.row_indices {
            let row_values: Vec<f64> = submatrix
                .col_indices
                .iter()
                .map(|&col_idx| data_array2[(row_idx, col_idx)])
                .collect();
            if !row_values.is_empty() {
                row_variances.push(Self::calculate_variance(&row_values));
            }
        }

        // 计算列的兼容性
        let mut col_variances = Vec::new();
        for &col_idx in &submatrix.col_indices {
            let col_values: Vec<f64> = submatrix
                .row_indices
                .iter()
                .map(|&row_idx| data_array2[(row_idx, col_idx)])
                .collect();
            if !col_values.is_empty() {
                col_variances.push(Self::calculate_variance(&col_values));
            }
        }

        let avg_row_variance = if row_variances.is_empty() {
            0.0
        } else {
            row_variances.iter().sum::<f64>() / row_variances.len() as f64
        };

        let avg_col_variance = if col_variances.is_empty() {
            0.0
        } else {
            col_variances.iter().sum::<f64>() / col_variances.len() as f64
        };

        // 兼容性分数：方差越小，兼容性越高
        // 这里使用一个简单的反函数形式，确保分数在0到1之间，并且随着方差增大而减小
        let row_compatibility = 1.0 / (1.0 + avg_row_variance / self.row_threshold);
        let col_compatibility = 1.0 / (1.0 + avg_col_variance / self.col_threshold);

        // 综合行和列的兼容性，例如取平均值
        (row_compatibility + col_compatibility) / 2.0
    }
}

/// 组合评分器，可以组合多个评分方法
pub struct CompositeScorer {
    scorers: Vec<(Box<dyn Scorer>, f64)>, // (评分器, 权重)
}

impl CompositeScorer {
    pub fn new() -> Self {
        Self {
            scorers: Vec::new(),
        }
    }

    pub fn add_scorer(mut self, scorer: Box<dyn Scorer>, weight: f64) -> Self {
        self.scorers.push((scorer, weight));
        self
    }
}

impl Scorer for CompositeScorer {
    fn score<'a>(
        &self,
        matrix: &'a Matrix<f64>,
        submatrix: &Submatrix<'a, f64>,
    ) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (scorer, weight) in &self.scorers {
            weighted_sum += scorer.score(matrix, submatrix) * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use ndarray::Array2;

    #[test]
    fn test_pearson_scorer() {
        let data_nalgebra = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, // 与第一行完全相关
                5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, // 与第三行完全相关
            ],
        );
        let data_ndarray = Array2::from_shape_vec(
            (data_nalgebra.nrows(), data_nalgebra.ncols()),
            data_nalgebra.as_slice().to_vec(),
        )
        .unwrap();

        let matrix_instance = crate::matrix::Matrix {
            data: data_ndarray.clone(),
            rows: data_ndarray.nrows(),
            cols: data_ndarray.ncols(),
        };
        let submatrix_option = Submatrix::new(&matrix_instance.data, vec![0, 1], vec![0, 1, 2, 3]);
        
        let submatrix = match submatrix_option {
            Some(sm) => sm,
            None => panic!("Submatrix creation failed in test"),
        };

        let scorer = PearsonScorer::new(2, 2);
        let score = scorer.score(&matrix_instance, &submatrix);

        assert!(score > 0.9); // 应该接近1.0，因为行完全相关
    }
}

// 添加到Cargo.toml：
// statrs = "0.16"
