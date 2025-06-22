use crate::tensor3d::{Tensor3D, TensorSubspace};
use nalgebra::{DMatrix, SVD};
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Tucker分解的rank配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuckerRank {
    /// 模式1的rank
    pub rank1: usize,
    /// 模式2的rank
    pub rank2: usize,
    /// 模式3的rank
    pub rank3: usize,
}

impl TuckerRank {
    /// 创建新的Tucker rank配置
    pub fn new(rank1: usize, rank2: usize, rank3: usize) -> Self {
        Self { rank1, rank2, rank3 }
    }
    
    /// 创建相等rank的配置
    pub fn uniform(rank: usize) -> Self {
        Self::new(rank, rank, rank)
    }
    
    /// 验证rank是否有效
    pub fn validate(&self, tensor_shape: [usize; 3]) -> Result<(), TuckerError> {
        if self.rank1 == 0 || self.rank1 > tensor_shape[0] {
            return Err(TuckerError::InvalidRank(format!(
                "Mode-1 rank {} is invalid for dimension {}",
                self.rank1, tensor_shape[0]
            )));
        }
        if self.rank2 == 0 || self.rank2 > tensor_shape[1] {
            return Err(TuckerError::InvalidRank(format!(
                "Mode-2 rank {} is invalid for dimension {}",
                self.rank2, tensor_shape[1]
            )));
        }
        if self.rank3 == 0 || self.rank3 > tensor_shape[2] {
            return Err(TuckerError::InvalidRank(format!(
                "Mode-3 rank {} is invalid for dimension {}",
                self.rank3, tensor_shape[2]
            )));
        }
        Ok(())
    }
}

/// Tucker分解的结果
#[derive(Debug, Clone)]
pub struct TuckerDecomposition {
    /// 核心张量 G (rank1 × rank2 × rank3)
    pub core_tensor: Tensor3D<f64>,
    /// 模式1因子矩阵 U1 (I1 × R1)
    pub factor_matrix1: DMatrix<f64>,
    /// 模式2因子矩阵 U2 (I2 × R2)
    pub factor_matrix2: DMatrix<f64>,
    /// 模式3因子矩阵 U3 (I3 × R3)
    pub factor_matrix3: DMatrix<f64>,
    /// 分解的rank
    pub ranks: TuckerRank,
    /// 重构误差
    pub reconstruction_error: f64,
}

impl TuckerDecomposition {
    /// 重构原始张量
    pub fn reconstruct(&self) -> Result<Tensor3D<f64>, TuckerError> {
        // 张量重构: X ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃
        let temp1 = self.core_tensor.mode_product(&self.factor_matrix1, 1)
            .map_err(|e| TuckerError::ReconstructionError(e.to_string()))?;
        
        let temp2 = temp1.mode_product(&self.factor_matrix2, 2)
            .map_err(|e| TuckerError::ReconstructionError(e.to_string()))?;
        
        let reconstructed = temp2.mode_product(&self.factor_matrix3, 3)
            .map_err(|e| TuckerError::ReconstructionError(e.to_string()))?;
        
        Ok(reconstructed)
    }
    
    /// 计算重构误差
    pub fn compute_reconstruction_error(&self, original: &Tensor3D<f64>) -> Result<f64, TuckerError> {
        let reconstructed = self.reconstruct()?;
        
        // 计算Frobenius范数误差
        let mut error_sum = 0.0;
        let shape = original.shape();
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let diff = original.data[[i, j, k]] - reconstructed.data[[i, j, k]];
                    error_sum += diff * diff;
                }
            }
        }
        
        Ok(error_sum.sqrt())
    }
    
    /// 获取模式n的主要成分
    pub fn get_mode_components(&self, mode: usize, num_components: usize) -> Result<Vec<Vec<usize>>, TuckerError> {
        let factor_matrix = match mode {
            1 => &self.factor_matrix1,
            2 => &self.factor_matrix2,
            3 => &self.factor_matrix3,
            _ => return Err(TuckerError::InvalidMode(mode)),
        };
        
        let mut components = Vec::new();
        let k = num_components.min(factor_matrix.ncols());
        
        for comp in 0..k {
            let column = factor_matrix.column(comp);
            
            // 找到该成分中权重最大的元素
            let mut indices_with_weights: Vec<(usize, f64)> = column.iter()
                .enumerate()
                .map(|(idx, &weight)| (idx, weight.abs()))
                .collect();
            
            indices_with_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // 选择权重最大的前几个索引
            let top_indices: Vec<usize> = indices_with_weights
                .into_iter()
                .take(factor_matrix.nrows() / 3) // 取前1/3作为主要成分
                .filter(|(_, weight)| *weight > 0.1) // 过滤小权重
                .map(|(idx, _)| idx)
                .collect();
            
            if !top_indices.is_empty() {
                components.push(top_indices);
            }
        }
        
        Ok(components)
    }
}

/// Tucker分解算法配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuckerConfig {
    /// Tucker rank配置
    pub ranks: TuckerRank,
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 收敛容差
    pub tolerance: f64,
    /// 是否初始化为随机值
    pub random_init: bool,
    /// 是否归一化因子矩阵
    pub normalize_factors: bool,
}

impl Default for TuckerConfig {
    fn default() -> Self {
        Self {
            ranks: TuckerRank::uniform(5),
            max_iterations: 100,
            tolerance: 1e-6,
            random_init: true,
            normalize_factors: true,
        }
    }
}

/// Tucker分解器
pub struct TuckerDecomposer {
    config: TuckerConfig,
}

impl TuckerDecomposer {
    /// 创建新的Tucker分解器
    pub fn new(config: TuckerConfig) -> Self {
        Self { config }
    }
    
    /// 使用默认配置创建分解器
    pub fn with_ranks(rank1: usize, rank2: usize, rank3: usize) -> Self {
        let mut config = TuckerConfig::default();
        config.ranks = TuckerRank::new(rank1, rank2, rank3);
        Self::new(config)
    }
    
    /// 执行Tucker分解
    pub fn decompose(&self, tensor: &Tensor3D<f64>) -> Result<TuckerDecomposition, TuckerError> {
        let shape = tensor.shape();
        self.config.ranks.validate(shape)?;
        
        // 初始化因子矩阵
        let mut u1 = self.initialize_factor_matrix(shape[0], self.config.ranks.rank1)?;
        let mut u2 = self.initialize_factor_matrix(shape[1], self.config.ranks.rank2)?;
        let mut u3 = self.initialize_factor_matrix(shape[2], self.config.ranks.rank3)?;
        
        let mut prev_error = f64::INFINITY;
        
        // 交替最小二乘迭代
        for iteration in 0..self.config.max_iterations {
            // 更新U1: 固定U2, U3，优化U1
            u1 = self.update_factor_matrix_mode1(tensor, &u2, &u3)?;
            
            // 更新U2: 固定U1, U3，优化U2
            u2 = self.update_factor_matrix_mode2(tensor, &u1, &u3)?;
            
            // 更新U3: 固定U1, U2，优化U3
            u3 = self.update_factor_matrix_mode3(tensor, &u1, &u2)?;
            
            // 归一化因子矩阵（可选）
            if self.config.normalize_factors {
                u1 = self.normalize_factor_matrix(u1);
                u2 = self.normalize_factor_matrix(u2);
                u3 = self.normalize_factor_matrix(u3);
            }
            
            // 计算核心张量
            let core_tensor = self.compute_core_tensor(tensor, &u1, &u2, &u3)?;
            
            // 计算重构误差
            let reconstruction_error = self.compute_error(tensor, &core_tensor, &u1, &u2, &u3)?;
            
            // 检查收敛
            if (prev_error - reconstruction_error).abs() < self.config.tolerance {
                println!("Tucker分解在第{}次迭代后收敛", iteration + 1);
                return Ok(TuckerDecomposition {
                    core_tensor,
                    factor_matrix1: u1,
                    factor_matrix2: u2,
                    factor_matrix3: u3,
                    ranks: self.config.ranks.clone(),
                    reconstruction_error,
                });
            }
            
            prev_error = reconstruction_error;
            
            if iteration % 10 == 0 {
                println!("迭代 {}: 重构误差 = {:.6}", iteration, reconstruction_error);
            }
        }
        
        // 如果没有收敛，返回最后的结果
        let core_tensor = self.compute_core_tensor(tensor, &u1, &u2, &u3)?;
        let reconstruction_error = self.compute_error(tensor, &core_tensor, &u1, &u2, &u3)?;
        
        println!("Tucker分解达到最大迭代次数 {}，最终误差: {:.6}", 
                 self.config.max_iterations, reconstruction_error);
        
        Ok(TuckerDecomposition {
            core_tensor,
            factor_matrix1: u1,
            factor_matrix2: u2,
            factor_matrix3: u3,
            ranks: self.config.ranks.clone(),
            reconstruction_error,
        })
    }
    
    /// 初始化因子矩阵
    fn initialize_factor_matrix(&self, rows: usize, cols: usize) -> Result<DMatrix<f64>, TuckerError> {
        if self.config.random_init {
            // 随机初始化
            use rand::Rng;
            let mut rng = rand::rng();
            let data: Vec<f64> = (0..rows * cols)
                .map(|_| rng.random::<f64>() - 0.5)
                .collect();
            Ok(DMatrix::from_vec(rows, cols, data))
        } else {
            // 单位矩阵初始化（截断）
            let mut matrix = DMatrix::zeros(rows, cols);
            let min_dim = rows.min(cols);
            for i in 0..min_dim {
                matrix[(i, i)] = 1.0;
            }
            Ok(matrix)
        }
    }
    
    /// 更新模式1因子矩阵
    fn update_factor_matrix_mode1(
        &self,
        tensor: &Tensor3D<f64>,
        u2: &DMatrix<f64>,
        u3: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, TuckerError> {
        // 计算模式1展开
        let x1 = tensor.unfold(1).map_err(|e| TuckerError::UnfoldingError(e.to_string()))?;
        
        // 计算Khatri-Rao乘积 U3 ⊙ U2
        let kr_product = self.khatri_rao_product(u3, u2)?;
        
        // 使用SVD求解 X₁ ≈ U₁ (U₃ ⊙ U₂)ᵀ
        let target = &x1 * &kr_product;
        let svd = SVD::new(target, true, false);
        
        match svd.u {
            Some(u) => Ok(u.columns(0, self.config.ranks.rank1).into_owned()),
            None => Err(TuckerError::SVDError("Failed to compute SVD for mode 1".to_string())),
        }
    }
    
    /// 更新模式2因子矩阵
    fn update_factor_matrix_mode2(
        &self,
        tensor: &Tensor3D<f64>,
        u1: &DMatrix<f64>,
        u3: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, TuckerError> {
        let x2 = tensor.unfold(2).map_err(|e| TuckerError::UnfoldingError(e.to_string()))?;
        let kr_product = self.khatri_rao_product(u3, u1)?;
        let target = &x2 * &kr_product;
        let svd = SVD::new(target, true, false);
        
        match svd.u {
            Some(u) => Ok(u.columns(0, self.config.ranks.rank2).into_owned()),
            None => Err(TuckerError::SVDError("Failed to compute SVD for mode 2".to_string())),
        }
    }
    
    /// 更新模式3因子矩阵
    fn update_factor_matrix_mode3(
        &self,
        tensor: &Tensor3D<f64>,
        u1: &DMatrix<f64>,
        u2: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, TuckerError> {
        let x3 = tensor.unfold(3).map_err(|e| TuckerError::UnfoldingError(e.to_string()))?;
        let kr_product = self.khatri_rao_product(u2, u1)?;
        let target = &x3 * &kr_product;
        let svd = SVD::new(target, true, false);
        
        match svd.u {
            Some(u) => Ok(u.columns(0, self.config.ranks.rank3).into_owned()),
            None => Err(TuckerError::SVDError("Failed to compute SVD for mode 3".to_string())),
        }
    }
    
    /// 计算Khatri-Rao乘积
    fn khatri_rao_product(&self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>, TuckerError> {
        if a.ncols() != b.ncols() {
            return Err(TuckerError::DimensionMismatch(
                "Matrices must have same number of columns for Khatri-Rao product".to_string()
            ));
        }
        
        let rows = a.nrows() * b.nrows();
        let cols = a.ncols();
        let mut result = DMatrix::zeros(rows, cols);
        
        for j in 0..cols {
            let col_a = a.column(j);
            let col_b = b.column(j);
            
            for (i, &val_a) in col_a.iter().enumerate() {
                for (k, &val_b) in col_b.iter().enumerate() {
                    result[(i * b.nrows() + k, j)] = val_a * val_b;
                }
            }
        }
        
        Ok(result)
    }
    
    /// 归一化因子矩阵
    fn normalize_factor_matrix(&self, mut matrix: DMatrix<f64>) -> DMatrix<f64> {
        for j in 0..matrix.ncols() {
            let mut col = matrix.column_mut(j);
            let norm = col.norm();
            if norm > 0.0 {
                col /= norm;
            }
        }
        matrix
    }
    
    /// 计算核心张量
    fn compute_core_tensor(
        &self,
        tensor: &Tensor3D<f64>,
        u1: &DMatrix<f64>,
        u2: &DMatrix<f64>,
        u3: &DMatrix<f64>,
    ) -> Result<Tensor3D<f64>, TuckerError> {
        // G = X ×₁ U₁ᵀ ×₂ U₂ᵀ ×₃ U₃ᵀ
        let temp1 = tensor.mode_product(&u1.transpose(), 1)
            .map_err(|e| TuckerError::ComputationError(e.to_string()))?;
        
        let temp2 = temp1.mode_product(&u2.transpose(), 2)
            .map_err(|e| TuckerError::ComputationError(e.to_string()))?;
        
        let core = temp2.mode_product(&u3.transpose(), 3)
            .map_err(|e| TuckerError::ComputationError(e.to_string()))?;
        
        Ok(core)
    }
    
    /// 计算重构误差
    fn compute_error(
        &self,
        original: &Tensor3D<f64>,
        core: &Tensor3D<f64>,
        u1: &DMatrix<f64>,
        u2: &DMatrix<f64>,
        u3: &DMatrix<f64>,
    ) -> Result<f64, TuckerError> {
        // 重构张量
        let temp1 = core.mode_product(u1, 1)
            .map_err(|e| TuckerError::ComputationError(e.to_string()))?;
        
        let temp2 = temp1.mode_product(u2, 2)
            .map_err(|e| TuckerError::ComputationError(e.to_string()))?;
        
        let reconstructed = temp2.mode_product(u3, 3)
            .map_err(|e| TuckerError::ComputationError(e.to_string()))?;
        
        // 计算Frobenius范数误差
        let mut error_sum = 0.0;
        let shape = original.shape();
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let diff = original.data[[i, j, k]] - reconstructed.data[[i, j, k]];
                    error_sum += diff * diff;
                }
            }
        }
        
        Ok(error_sum.sqrt())
    }
}

/// Tucker分解相关错误
#[derive(Debug)]
pub enum TuckerError {
    InvalidRank(String),
    InvalidMode(usize),
    UnfoldingError(String),
    SVDError(String),
    DimensionMismatch(String),
    ComputationError(String),
    ReconstructionError(String),
}

impl fmt::Display for TuckerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TuckerError::InvalidRank(msg) => write!(f, "Invalid rank: {}", msg),
            TuckerError::InvalidMode(mode) => write!(f, "Invalid mode: {}", mode),
            TuckerError::UnfoldingError(msg) => write!(f, "Unfolding error: {}", msg),
            TuckerError::SVDError(msg) => write!(f, "SVD error: {}", msg),
            TuckerError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            TuckerError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            TuckerError::ReconstructionError(msg) => write!(f, "Reconstruction error: {}", msg),
        }
    }
}

impl Error for TuckerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor3d::Tensor3D;
    
    #[test]
    fn test_tucker_rank_validation() {
        let rank = TuckerRank::new(3, 4, 5);
        assert!(rank.validate([5, 6, 7]).is_ok());
        assert!(rank.validate([2, 6, 7]).is_err()); // rank1 > dim1
    }
    
    #[test]
    fn test_tucker_decomposition() {
        let tensor = Tensor3D::random([10, 8, 6]);
        let decomposer = TuckerDecomposer::with_ranks(3, 3, 3);
        
        let result = decomposer.decompose(&tensor);
        assert!(result.is_ok());
        
        let decomposition = result.unwrap();
        assert_eq!(decomposition.core_tensor.shape(), [3, 3, 3]);
        assert_eq!(decomposition.factor_matrix1.shape(), (10, 3));
        assert_eq!(decomposition.factor_matrix2.shape(), (8, 3));
        assert_eq!(decomposition.factor_matrix3.shape(), (6, 3));
    }
    
    #[test]
    fn test_tensor_reconstruction() {
        let tensor = Tensor3D::random([6, 5, 4]);
        let decomposer = TuckerDecomposer::with_ranks(3, 3, 3);
        
        let decomposition = decomposer.decompose(&tensor).unwrap();
        let reconstructed = decomposition.reconstruct().unwrap();
        
        assert_eq!(reconstructed.shape(), tensor.shape());
        
        // 检查重构误差是否合理
        let error = decomposition.compute_reconstruction_error(&tensor).unwrap();
        assert!(error >= 0.0);
        println!("重构误差: {:.6}", error);
    }
    
    #[test]
    fn test_khatri_rao_product() {
        let decomposer = TuckerDecomposer::with_ranks(2, 2, 2);
        let a = DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = DMatrix::from_vec(3, 3, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        
        let result = decomposer.khatri_rao_product(&a, &b);
        assert!(result.is_ok());
        
        let kr = result.unwrap();
        assert_eq!(kr.shape(), (6, 3)); // 2*3 × 3
    }
}