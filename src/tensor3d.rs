use ndarray::{Array3, ArrayView3, Axis, Dimension, Ix3};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::fmt;

/// 3D张量数据结构
#[derive(Debug, Clone)]
pub struct Tensor3D<T> {
    /// 3D数组数据 (mode1 × mode2 × mode3)
    pub data: Array3<T>,
    /// 模式1标签 (例如：用户、基因、时间点等)
    pub mode1_labels: Vec<String>,
    /// 模式2标签 (例如：物品、条件、空间点等)
    pub mode2_labels: Vec<String>,
    /// 模式3标签 (例如：时间、样本、特征等)
    pub mode3_labels: Vec<String>,
}

impl<T> Tensor3D<T> 
where
    T: Clone + Default,
{
    /// 创建新的3D张量
    pub fn new(
        data: Array3<T>,
        mode1_labels: Vec<String>,
        mode2_labels: Vec<String>,
        mode3_labels: Vec<String>,
    ) -> Result<Self, &'static str> {
        let shape = data.shape();
        
        if mode1_labels.len() != shape[0] {
            return Err("Mode1 labels length doesn't match tensor dimension 0");
        }
        if mode2_labels.len() != shape[1] {
            return Err("Mode2 labels length doesn't match tensor dimension 1");
        }
        if mode3_labels.len() != shape[2] {
            return Err("Mode3 labels length doesn't match tensor dimension 2");
        }
        
        Ok(Tensor3D {
            data,
            mode1_labels,
            mode2_labels,
            mode3_labels,
        })
    }
    
    /// 从原始数据创建张量（不含标签）
    pub fn from_data(data: Array3<T>) -> Self {
        let shape = data.shape();
        let mode1_labels = (0..shape[0]).map(|i| format!("mode1_{}", i)).collect();
        let mode2_labels = (0..shape[1]).map(|i| format!("mode2_{}", i)).collect();
        let mode3_labels = (0..shape[2]).map(|i| format!("mode3_{}", i)).collect();
        
        Tensor3D {
            data,
            mode1_labels,
            mode2_labels,
            mode3_labels,
        }
    }
    
    /// 获取张量形状
    pub fn shape(&self) -> [usize; 3] {
        let shape = self.data.shape();
        [shape[0], shape[1], shape[2]]
    }
    
    /// 获取张量大小（总元素数）
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// 获取模式n的大小
    pub fn mode_size(&self, mode: usize) -> Result<usize, &'static str> {
        match mode {
            1 => Ok(self.shape()[0]),
            2 => Ok(self.shape()[1]),
            3 => Ok(self.shape()[2]),
            _ => Err("Invalid mode: must be 1, 2, or 3"),
        }
    }
    
    /// 检查张量是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Tensor3D<f64> {
    /// 创建随机张量
    pub fn random(shape: [usize; 3]) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;
        
        let data = Array3::random(shape, Uniform::new(0.0, 1.0));
        Self::from_data(data)
    }
    
    /// 创建零张量
    pub fn zeros(shape: [usize; 3]) -> Self {
        let data = Array3::zeros(shape);
        Self::from_data(data)
    }
    
    /// 创建一张量
    pub fn ones(shape: [usize; 3]) -> Self {
        let data = Array3::ones(shape);
        Self::from_data(data)
    }
    
    /// 张量模式n展开 (matricization)
    pub fn unfold(&self, mode: usize) -> Result<DMatrix<f64>, &'static str> {
        let shape = self.shape();
        
        match mode {
            1 => {
                // Mode-1 unfolding: tensor(I1 × I2 × I3) -> matrix(I1 × I2*I3)
                let unfolded = self.data.view()
                    .into_shape((shape[0], shape[1] * shape[2]))
                    .map_err(|_| "Failed to unfold tensor in mode 1")?;
                Ok(DMatrix::from_row_slice(shape[0], shape[1] * shape[2], unfolded.as_slice().unwrap()))
            },
            2 => {
                // Mode-2 unfolding: tensor(I1 × I2 × I3) -> matrix(I2 × I1*I3)
                let mut unfolded_data = Vec::with_capacity(shape[1] * shape[0] * shape[2]);
                for j in 0..shape[1] {
                    for i in 0..shape[0] {
                        for k in 0..shape[2] {
                            unfolded_data.push(self.data[[i, j, k]]);
                        }
                    }
                }
                Ok(DMatrix::from_row_slice(shape[1], shape[0] * shape[2], &unfolded_data))
            },
            3 => {
                // Mode-3 unfolding: tensor(I1 × I2 × I3) -> matrix(I3 × I1*I2)
                let mut unfolded_data = Vec::with_capacity(shape[2] * shape[0] * shape[1]);
                for k in 0..shape[2] {
                    for i in 0..shape[0] {
                        for j in 0..shape[1] {
                            unfolded_data.push(self.data[[i, j, k]]);
                        }
                    }
                }
                Ok(DMatrix::from_row_slice(shape[2], shape[0] * shape[1], &unfolded_data))
            },
            _ => Err("Invalid mode: must be 1, 2, or 3"),
        }
    }
    
    /// 计算张量的Frobenius范数
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
    
    /// 张量归一化
    pub fn normalize(&mut self) {
        let norm = self.frobenius_norm();
        if norm > 0.0 {
            self.data.mapv_inplace(|x| x / norm);
        }
    }
    
    /// 获取归一化后的张量（不修改原张量）
    pub fn normalized(&self) -> Self {
        let norm = self.frobenius_norm();
        if norm > 0.0 {
            let normalized_data = self.data.mapv(|x| x / norm);
            Tensor3D {
                data: normalized_data,
                mode1_labels: self.mode1_labels.clone(),
                mode2_labels: self.mode2_labels.clone(),
                mode3_labels: self.mode3_labels.clone(),
            }
        } else {
            self.clone()
        }
    }
    
    /// 张量的模式n乘积 (mode-n product)
    pub fn mode_product(&self, matrix: &DMatrix<f64>, mode: usize) -> Result<Tensor3D<f64>, &'static str> {
        let shape = self.shape();
        
        match mode {
            1 => {
                // 模式1乘积: Y = X ×₁ U, Y(j,i2,i3) = Σᵢ₁ X(i1,i2,i3) * U(j,i1)
                if matrix.ncols() != shape[0] {
                    return Err("Matrix columns must match tensor mode-1 dimension");
                }
                
                let new_shape = [matrix.nrows(), shape[1], shape[2]];
                let mut result_data = Array3::zeros(new_shape);
                
                for j in 0..matrix.nrows() {
                    for i2 in 0..shape[1] {
                        for i3 in 0..shape[2] {
                            let mut sum = 0.0;
                            for i1 in 0..shape[0] {
                                sum += self.data[[i1, i2, i3]] * matrix[(j, i1)];
                            }
                            result_data[[j, i2, i3]] = sum;
                        }
                    }
                }
                
                let mode1_labels = (0..matrix.nrows()).map(|i| format!("comp1_{}", i)).collect();
                Ok(Tensor3D::new(result_data, mode1_labels, self.mode2_labels.clone(), self.mode3_labels.clone())?)
            },
            2 => {
                // 模式2乘积: Y = X ×₂ U, Y(i1,j,i3) = Σᵢ₂ X(i1,i2,i3) * U(j,i2)
                if matrix.ncols() != shape[1] {
                    return Err("Matrix columns must match tensor mode-2 dimension");
                }
                
                let new_shape = [shape[0], matrix.nrows(), shape[2]];
                let mut result_data = Array3::zeros(new_shape);
                
                for i1 in 0..shape[0] {
                    for j in 0..matrix.nrows() {
                        for i3 in 0..shape[2] {
                            let mut sum = 0.0;
                            for i2 in 0..shape[1] {
                                sum += self.data[[i1, i2, i3]] * matrix[(j, i2)];
                            }
                            result_data[[i1, j, i3]] = sum;
                        }
                    }
                }
                
                let mode2_labels = (0..matrix.nrows()).map(|i| format!("comp2_{}", i)).collect();
                Ok(Tensor3D::new(result_data, self.mode1_labels.clone(), mode2_labels, self.mode3_labels.clone())?)
            },
            3 => {
                // 模式3乘积: Y = X ×₃ U, Y(i1,i2,j) = Σᵢ₃ X(i1,i2,i3) * U(j,i3)
                if matrix.ncols() != shape[2] {
                    return Err("Matrix columns must match tensor mode-3 dimension");
                }
                
                let new_shape = [shape[0], shape[1], matrix.nrows()];
                let mut result_data = Array3::zeros(new_shape);
                
                for i1 in 0..shape[0] {
                    for i2 in 0..shape[1] {
                        for j in 0..matrix.nrows() {
                            let mut sum = 0.0;
                            for i3 in 0..shape[2] {
                                sum += self.data[[i1, i2, i3]] * matrix[(j, i3)];
                            }
                            result_data[[i1, i2, j]] = sum;
                        }
                    }
                }
                
                let mode3_labels = (0..matrix.nrows()).map(|i| format!("comp3_{}", i)).collect();
                Ok(Tensor3D::new(result_data, self.mode1_labels.clone(), self.mode2_labels.clone(), mode3_labels)?)
            },
            _ => Err("Invalid mode: must be 1, 2, or 3"),
        }
    }
}

impl<T: fmt::Display + Clone + Default> fmt::Display for Tensor3D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.shape();
        writeln!(f, "Tensor3D shape: {} × {} × {}", shape[0], shape[1], shape[2])?;
        writeln!(f, "Mode1 labels: {:?}", self.mode1_labels)?;
        writeln!(f, "Mode2 labels: {:?}", self.mode2_labels)?;
        writeln!(f, "Mode3 labels: {:?}", self.mode3_labels)?;
        
        // 显示张量的一小部分数据
        writeln!(f, "Data preview:")?;
        let preview_size = 3.min(shape[0]);
        for i in 0..preview_size {
            writeln!(f, "Slice {} (mode1 = {}):", i, self.mode1_labels.get(i).unwrap_or(&format!("{}", i)))?;
            for j in 0..3.min(shape[1]) {
                write!(f, "  ")?;
                for k in 0..3.min(shape[2]) {
                    write!(f, "{:8.3} ", self.data[[i, j, k]])?;
                }
                if shape[2] > 3 {
                    write!(f, "... ")?;
                }
                writeln!(f)?;
            }
            if shape[1] > 3 {
                writeln!(f, "  ...")?;
            }
        }
        if shape[0] > 3 {
            writeln!(f, "...")?;
        }
        
        Ok(())
    }
}

/// 3D张量切片，用于co-clustering结果
#[derive(Debug, Clone)]
pub struct TensorSubspace<'a> {
    /// 原始张量的引用
    pub tensor: &'a Tensor3D<f64>,
    /// 模式1索引
    pub mode1_indices: Vec<usize>,
    /// 模式2索引
    pub mode2_indices: Vec<usize>,
    /// 模式3索引
    pub mode3_indices: Vec<usize>,
    /// 子空间的质量分数
    pub score: Option<f64>,
}

impl<'a> TensorSubspace<'a> {
    /// 创建新的张量子空间
    pub fn new(
        tensor: &'a Tensor3D<f64>,
        mode1_indices: Vec<usize>,
        mode2_indices: Vec<usize>,
        mode3_indices: Vec<usize>,
    ) -> Option<Self> {
        let shape = tensor.shape();
        
        // 验证索引有效性
        if mode1_indices.iter().any(|&i| i >= shape[0]) ||
           mode2_indices.iter().any(|&i| i >= shape[1]) ||
           mode3_indices.iter().any(|&i| i >= shape[2]) {
            return None;
        }
        
        Some(TensorSubspace {
            tensor,
            mode1_indices,
            mode2_indices,
            mode3_indices,
            score: None,
        })
    }
    
    /// 获取子空间的形状
    pub fn shape(&self) -> [usize; 3] {
        [
            self.mode1_indices.len(),
            self.mode2_indices.len(),
            self.mode3_indices.len(),
        ]
    }
    
    /// 获取子空间的大小
    pub fn size(&self) -> usize {
        self.mode1_indices.len() * self.mode2_indices.len() * self.mode3_indices.len()
    }
    
    /// 提取子张量数据
    pub fn extract_data(&self) -> Array3<f64> {
        let sub_shape = self.shape();
        let mut sub_data = Array3::zeros(sub_shape);
        
        for (i, &mode1_idx) in self.mode1_indices.iter().enumerate() {
            for (j, &mode2_idx) in self.mode2_indices.iter().enumerate() {
                for (k, &mode3_idx) in self.mode3_indices.iter().enumerate() {
                    sub_data[[i, j, k]] = self.tensor.data[[mode1_idx, mode2_idx, mode3_idx]];
                }
            }
        }
        
        sub_data
    }
    
    /// 计算子空间的Frobenius范数
    pub fn frobenius_norm(&self) -> f64 {
        let sub_data = self.extract_data();
        sub_data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
    
    /// 更新子空间分数
    pub fn update_score(&mut self) {
        // 简单的分数计算：基于子空间的密度和一致性
        let sub_data = self.extract_data();
        let mean = sub_data.mean().unwrap_or(0.0);
        let variance = sub_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / sub_data.len() as f64;
        
        // 分数 = 密度 - 方差（我们希望高密度、低方差的子空间）
        self.score = Some(mean.abs() - variance.sqrt());
    }
}

impl<'a> fmt::Display for TensorSubspace<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let shape = self.shape();
        write!(f, "TensorSubspace {}×{}×{}", shape[0], shape[1], shape[2])?;
        if let Some(score) = self.score {
            write!(f, " (score: {:.3})", score)?;
        }
        writeln!(f)?;
        writeln!(f, "  Mode1 indices: {:?}", self.mode1_indices)?;
        writeln!(f, "  Mode2 indices: {:?}", self.mode2_indices)?;
        writeln!(f, "  Mode3 indices: {:?}", self.mode3_indices)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[test]
    fn test_tensor3d_creation() {
        let data = Array3::<f64>::zeros([3, 4, 5]);
        let tensor = Tensor3D::from_data(data);
        
        assert_eq!(tensor.shape(), [3, 4, 5]);
        assert_eq!(tensor.size(), 60);
        assert_eq!(tensor.mode_size(1).unwrap(), 3);
        assert_eq!(tensor.mode_size(2).unwrap(), 4);
        assert_eq!(tensor.mode_size(3).unwrap(), 5);
    }
    
    #[test]
    fn test_tensor3d_random() {
        let tensor = Tensor3D::random([2, 3, 4]);
        assert_eq!(tensor.shape(), [2, 3, 4]);
        assert!(!tensor.is_empty());
    }
    
    #[test]
    fn test_tensor_unfold() {
        let data = Array3::from_shape_fn([2, 3, 4], |(i, j, k)| (i * 12 + j * 4 + k) as f64);
        let tensor = Tensor3D::from_data(data);
        
        // Test mode-1 unfolding
        let unfolded1 = tensor.unfold(1).unwrap();
        assert_eq!(unfolded1.shape(), (2, 12)); // 2 × (3*4)
        
        // Test mode-2 unfolding
        let unfolded2 = tensor.unfold(2).unwrap();
        assert_eq!(unfolded2.shape(), (3, 8)); // 3 × (2*4)
        
        // Test mode-3 unfolding
        let unfolded3 = tensor.unfold(3).unwrap();
        assert_eq!(unfolded3.shape(), (4, 6)); // 4 × (2*3)
    }
    
    #[test]
    fn test_tensor_normalization() {
        let mut tensor = Tensor3D::random([2, 2, 2]);
        let original_norm = tensor.frobenius_norm();
        
        tensor.normalize();
        let normalized_norm = tensor.frobenius_norm();
        
        assert!((normalized_norm - 1.0).abs() < 1e-10);
        assert!(original_norm > 0.0);
    }
    
    #[test]
    fn test_tensor_subspace() {
        let tensor = Tensor3D::random([5, 4, 3]);
        let subspace = TensorSubspace::new(&tensor, vec![0, 2], vec![1, 3], vec![0, 1, 2]);
        
        assert!(subspace.is_some());
        let subspace = subspace.unwrap();
        assert_eq!(subspace.shape(), [2, 2, 3]);
        assert_eq!(subspace.size(), 12);
        
        let sub_data = subspace.extract_data();
        assert_eq!(sub_data.shape(), &[2, 2, 3]);
    }
    
    #[test]
    fn test_mode_product() {
        let tensor = Tensor3D::random([3, 4, 5]);
        let matrix = DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        let result = tensor.mode_product(&matrix, 1).unwrap();
        assert_eq!(result.shape(), [2, 4, 5]);
    }
}