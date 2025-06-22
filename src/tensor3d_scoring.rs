use crate::tensor3d::{Tensor3D, TensorSubspace};
use crate::tucker_decomposition::{TuckerDecomposition, TuckerDecomposer, TuckerRank};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::error::Error;

/// 3D张量评分器的trait
pub trait TensorScorer: Send + Sync {
    /// 为张量子空间计算分数
    fn score(&self, tensor: &Tensor3D<f64>, subspace: &TensorSubspace) -> f64;
    
    /// 为多个子空间批量计算分数
    fn score_all(&self, tensor: &Tensor3D<f64>, subspaces: &[TensorSubspace]) -> Vec<f64> {
        subspaces.iter().map(|sub| self.score(tensor, sub)).collect()
    }
    
    /// 获取评分器名称
    fn name(&self) -> &str;
}

/// 基于Tucker分解的评分器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuckerScorer {
    /// Tucker rank配置
    pub tucker_rank: TuckerRank,
    /// 最小子空间大小
    pub min_subspace_size: usize,
    /// 权重配置
    pub weights: TuckerScoringWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuckerScoringWeights {
    /// 分解质量权重
    pub decomposition_quality: f64,
    /// 压缩比权重
    pub compression_ratio: f64,
    /// 密度权重
    pub density: f64,
    /// 一致性权重
    pub coherence: f64,
}

impl Default for TuckerScoringWeights {
    fn default() -> Self {
        Self {
            decomposition_quality: 0.4,
            compression_ratio: 0.2,
            density: 0.2,
            coherence: 0.2,
        }
    }
}

impl TuckerScorer {
    /// 创建新的Tucker评分器
    pub fn new(tucker_rank: TuckerRank) -> Self {
        Self {
            tucker_rank,
            min_subspace_size: 8,
            weights: TuckerScoringWeights::default(),
        }
    }
    
    /// 创建均匀rank的Tucker评分器
    pub fn with_uniform_rank(rank: usize) -> Self {
        Self::new(TuckerRank::uniform(rank))
    }
    
    /// 设置最小子空间大小
    pub fn with_min_size(mut self, min_size: usize) -> Self {
        self.min_subspace_size = min_size;
        self
    }
    
    /// 设置权重
    pub fn with_weights(mut self, weights: TuckerScoringWeights) -> Self {
        self.weights = weights;
        self
    }
    
    /// 计算分解质量分数
    fn compute_decomposition_quality(&self, subspace: &TensorSubspace) -> f64 {
        let sub_data = subspace.extract_data();
        let sub_tensor = Tensor3D::from_data(sub_data);
        
        // 尝试Tucker分解
        let mut config = crate::tucker_decomposition::TuckerConfig::default();
        config.ranks = self.tucker_rank.clone();
        config.max_iterations = 20; // 减少迭代次数以提高速度
        
        let decomposer = TuckerDecomposer::new(config);
        
        match decomposer.decompose(&sub_tensor) {
            Ok(decomposition) => {
                // 计算相对重构误差
                let original_norm = sub_tensor.frobenius_norm();
                if original_norm > 0.0 {
                    let relative_error = decomposition.reconstruction_error / original_norm;
                    // 分数 = 1 - 相对误差（越小的误差，越好的分数）
                    (1.0 - relative_error).max(0.0)
                } else {
                    0.0
                }
            },
            Err(_) => 0.0, // 分解失败给0分
        }
    }
    
    /// 计算压缩比分数
    fn compute_compression_ratio(&self, subspace: &TensorSubspace) -> f64 {
        let shape = subspace.shape();
        let original_size = shape[0] * shape[1] * shape[2];
        
        // Tucker分解的参数数量
        let tucker_params = self.tucker_rank.rank1 * self.tucker_rank.rank2 * self.tucker_rank.rank3 +
                           shape[0] * self.tucker_rank.rank1 +
                           shape[1] * self.tucker_rank.rank2 +
                           shape[2] * self.tucker_rank.rank3;
        
        if original_size > 0 {
            let compression_ratio = tucker_params as f64 / original_size as f64;
            // 压缩比越小越好，但要避免过度压缩
            if compression_ratio < 0.5 {
                1.0 - compression_ratio
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// 计算密度分数
    fn compute_density(&self, subspace: &TensorSubspace) -> f64 {
        let sub_data = subspace.extract_data();
        let mean = sub_data.mean().unwrap_or(0.0);
        let std = sub_data.std(0.0);
        
        // 密度 = 信噪比
        if std > 0.0 {
            (mean.abs() / std).tanh() // 使用tanh归一化到[0,1]
        } else if mean.abs() > 0.0 {
            1.0 // 如果方差为0但均值不为0，说明是完全一致的
        } else {
            0.0
        }
    }
    
    /// 计算一致性分数
    fn compute_coherence(&self, subspace: &TensorSubspace) -> f64 {
        let sub_data = subspace.extract_data();
        let shape = subspace.shape();
        
        if shape[0] < 2 || shape[1] < 2 || shape[2] < 2 {
            return 0.0;
        }
        
        // 计算各个模式的内部相关性
        let mut correlations = Vec::new();
        
        // 模式1：计算行之间的平均相关性
        for i in 0..(shape[0] - 1) {
            for j in (i + 1)..shape[0] {
                let mut slice1 = Vec::new();
                let mut slice2 = Vec::new();
                
                for y in 0..shape[1] {
                    for z in 0..shape[2] {
                        slice1.push(sub_data[[i, y, z]]);
                        slice2.push(sub_data[[j, y, z]]);
                    }
                }
                
                if let Some(corr) = self.pearson_correlation(&slice1, &slice2) {
                    correlations.push(corr.abs());
                }
            }
        }
        
        // 模式2：计算列之间的平均相关性
        for i in 0..(shape[1] - 1) {
            for j in (i + 1)..shape[1] {
                let mut slice1 = Vec::new();
                let mut slice2 = Vec::new();
                
                for x in 0..shape[0] {
                    for z in 0..shape[2] {
                        slice1.push(sub_data[[x, i, z]]);
                        slice2.push(sub_data[[x, j, z]]);
                    }
                }
                
                if let Some(corr) = self.pearson_correlation(&slice1, &slice2) {
                    correlations.push(corr.abs());
                }
            }
        }
        
        // 模式3：计算第三维切片之间的平均相关性
        for i in 0..(shape[2] - 1) {
            for j in (i + 1)..shape[2] {
                let mut slice1 = Vec::new();
                let mut slice2 = Vec::new();
                
                for x in 0..shape[0] {
                    for y in 0..shape[1] {
                        slice1.push(sub_data[[x, y, i]]);
                        slice2.push(sub_data[[x, y, j]]);
                    }
                }
                
                if let Some(corr) = self.pearson_correlation(&slice1, &slice2) {
                    correlations.push(corr.abs());
                }
            }
        }
        
        // 返回平均相关性
        if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        }
    }
    
    /// 计算Pearson相关系数
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }
        
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator > 0.0 {
            Some(numerator / denominator)
        } else {
            None
        }
    }
}

impl TensorScorer for TuckerScorer {
    fn score(&self, _tensor: &Tensor3D<f64>, subspace: &TensorSubspace) -> f64 {
        // 检查最小大小要求
        if subspace.size() < self.min_subspace_size {
            return 0.0;
        }
        
        // 计算各个组件分数
        let decomp_score = self.compute_decomposition_quality(subspace);
        let compression_score = self.compute_compression_ratio(subspace);
        let density_score = self.compute_density(subspace);
        let coherence_score = self.compute_coherence(subspace);
        
        // 加权组合
        self.weights.decomposition_quality * decomp_score +
        self.weights.compression_ratio * compression_score +
        self.weights.density * density_score +
        self.weights.coherence * coherence_score
    }
    
    fn name(&self) -> &str {
        "TuckerScorer"
    }
}

/// 密度基础的评分器
#[derive(Debug, Clone)]
pub struct DensityScorer {
    pub min_density: f64,
}

impl DensityScorer {
    pub fn new(min_density: f64) -> Self {
        Self { min_density }
    }
}

impl TensorScorer for DensityScorer {
    fn score(&self, _tensor: &Tensor3D<f64>, subspace: &TensorSubspace) -> f64 {
        let sub_data = subspace.extract_data();
        let mean = sub_data.mean().unwrap_or(0.0);
        let std = sub_data.std(0.0);
        
        if mean.abs() < self.min_density {
            return 0.0;
        }
        
        if std > 0.0 {
            (mean.abs() / std).min(10.0) / 10.0 // 归一化到[0,1]
        } else {
            1.0
        }
    }
    
    fn name(&self) -> &str {
        "DensityScorer"
    }
}

/// 方差基础的评分器
#[derive(Debug, Clone)]
pub struct VarianceScorer {
    pub max_variance: f64,
}

impl VarianceScorer {
    pub fn new(max_variance: f64) -> Self {
        Self { max_variance }
    }
}

impl TensorScorer for VarianceScorer {
    fn score(&self, _tensor: &Tensor3D<f64>, subspace: &TensorSubspace) -> f64 {
        let sub_data = subspace.extract_data();
        let variance = sub_data.var(0.0);
        
        if variance <= self.max_variance {
            1.0 - (variance / self.max_variance)
        } else {
            0.0
        }
    }
    
    fn name(&self) -> &str {
        "VarianceScorer"
    }
}

/// 组合评分器
pub struct CompositeTensorScorer {
    scorers: Vec<(Box<dyn TensorScorer>, f64)>,
}

impl CompositeTensorScorer {
    pub fn new() -> Self {
        Self {
            scorers: Vec::new(),
        }
    }
    
    pub fn add_scorer(mut self, scorer: Box<dyn TensorScorer>, weight: f64) -> Self {
        self.scorers.push((scorer, weight));
        self
    }
}

impl TensorScorer for CompositeTensorScorer {
    fn score(&self, tensor: &Tensor3D<f64>, subspace: &TensorSubspace) -> f64 {
        if self.scorers.is_empty() {
            return 0.0;
        }
        
        let total_weight: f64 = self.scorers.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        
        let weighted_sum: f64 = self.scorers.iter()
            .map(|(scorer, weight)| scorer.score(tensor, subspace) * weight)
            .sum();
        
        weighted_sum / total_weight
    }
    
    fn name(&self) -> &str {
        "CompositeTensorScorer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor3d::Tensor3D;
    
    #[test]
    fn test_tucker_scorer() {
        let tensor = Tensor3D::random([10, 8, 6]);
        let subspace = TensorSubspace::new(&tensor, vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]).unwrap();
        
        let scorer = TuckerScorer::with_uniform_rank(2);
        let score = scorer.score(&tensor, &subspace);
        
        assert!(score >= 0.0 && score <= 1.0);
        println!("Tucker score: {:.3}", score);
    }
    
    #[test]
    fn test_density_scorer() {
        let tensor = Tensor3D::random([5, 5, 5]);
        let subspace = TensorSubspace::new(&tensor, vec![0, 1], vec![0, 1], vec![0, 1]).unwrap();
        
        let scorer = DensityScorer::new(0.1);
        let score = scorer.score(&tensor, &subspace);
        
        assert!(score >= 0.0 && score <= 1.0);
        println!("Density score: {:.3}", score);
    }
    
    #[test]
    fn test_variance_scorer() {
        let tensor = Tensor3D::random([5, 5, 5]);
        let subspace = TensorSubspace::new(&tensor, vec![0, 1], vec![0, 1], vec![0, 1]).unwrap();
        
        let scorer = VarianceScorer::new(1.0);
        let score = scorer.score(&tensor, &subspace);
        
        assert!(score >= 0.0 && score <= 1.0);
        println!("Variance score: {:.3}", score);
    }
    
    #[test]
    fn test_composite_scorer() {
        let tensor = Tensor3D::random([8, 6, 5]);
        let subspace = TensorSubspace::new(&tensor, vec![0, 1, 2], vec![0, 1], vec![0, 1]).unwrap();
        
        let scorer = CompositeTensorScorer::new()
            .add_scorer(Box::new(TuckerScorer::with_uniform_rank(2)), 0.6)
            .add_scorer(Box::new(DensityScorer::new(0.1)), 0.2)
            .add_scorer(Box::new(VarianceScorer::new(1.0)), 0.2);
        
        let score = scorer.score(&tensor, &subspace);
        
        assert!(score >= 0.0 && score <= 1.0);
        println!("Composite score: {:.3}", score);
    }
}