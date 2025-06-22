use nalgebra::{DMatrix, DVector};
use ndarray::Array2;
use std::error::Error;
use crate::util::clone_to_dmatrix;
use kmeans_smid::{KMeans, KMeansConfig};

/// 矩阵归一化策略
pub trait MatrixNormalizer {
    fn normalize(&self, matrix: &DMatrix<f64>) -> DMatrix<f64>;
}

/// 降维算法策略
pub trait DimensionalityReducer {
    fn reduce(&self, matrix: &DMatrix<f64>, k: usize) -> Result<(DMatrix<f64>, DMatrix<f64>), Box<dyn Error>>;
}

/// 特征组合策略
pub trait FeatureCombiner {
    fn combine(&self, u: &DMatrix<f64>, v: &DMatrix<f64>) -> DMatrix<f64>;
}

/// 聚类分配策略
pub trait ClusterAssigner {
    fn assign(&self, features: &DMatrix<f64>, k: usize) -> Result<Vec<usize>, Box<dyn Error>>;
}

// ============ 具体实现 ============

/// 标准矩阵归一化器 (原始方法)
pub struct StandardNormalizer;

impl MatrixNormalizer for StandardNormalizer {
    fn normalize(&self, matrix: &DMatrix<f64>) -> DMatrix<f64> {
        let rows = matrix.nrows();
        let cols = matrix.ncols();
        
        let one_column = DVector::from_element(cols, 1.0);
        let one_row = DVector::from_element(rows, 1.0).transpose();
        
        let du = matrix * one_column;
        let dv = (one_row * matrix);
        
        let du_inv_sqrt = du.map(|x| x.powf(-0.5));
        let dv_inv_sqrt = dv.map(|x| x.powf(-0.5));
        
        let mut normalized = matrix.clone();
        for (i, mut row) in normalized.row_iter_mut().enumerate() {
            row *= du_inv_sqrt[i];
        }
        for (j, mut col) in normalized.column_iter_mut().enumerate() {
            col *= dv_inv_sqrt[j];
        }
        
        normalized
    }
}

/// Z-score归一化器
pub struct ZScoreNormalizer;

impl MatrixNormalizer for ZScoreNormalizer {
    fn normalize(&self, matrix: &DMatrix<f64>) -> DMatrix<f64> {
        let mean = matrix.mean();
        let std = matrix.map(|x| (x - mean).powi(2)).mean().sqrt();
        
        if std > 0.0 {
            matrix.map(|x| (x - mean) / std)
        } else {
            matrix.clone()
        }
    }
}

/// SVD降维器
pub struct SVDReducer;

impl DimensionalityReducer for SVDReducer {
    fn reduce(&self, matrix: &DMatrix<f64>, k: usize) -> Result<(DMatrix<f64>, DMatrix<f64>), Box<dyn Error>> {
        use nalgebra::SVD;
        
        let svd_result = SVD::new(matrix.clone(), true, true);
        
        let u_mat = match svd_result.u {
            Some(u) => u.columns(0, k).into_owned(),
            None => return Err("SVD failed to compute U matrix".into()),
        };
        
        let v_t_mat = match svd_result.v_t {
            Some(vt) => vt.rows(0, k).into_owned().transpose(),
            None => return Err("SVD failed to compute V matrix".into()),
        };
        
        Ok((u_mat, v_t_mat))
    }
}

/// PCA降维器 (简化版本)
pub struct PCAReducer;

impl DimensionalityReducer for PCAReducer {
    fn reduce(&self, matrix: &DMatrix<f64>, k: usize) -> Result<(DMatrix<f64>, DMatrix<f64>), Box<dyn Error>> {
        // 简化的PCA实现 - 在实际应用中可以使用更完善的PCA算法
        // 这里为了演示，我们使用SVD作为PCA的底层实现
        let svd_reducer = SVDReducer;
        svd_reducer.reduce(matrix, k)
    }
}

/// 垂直拼接特征组合器
pub struct VerticalCombiner;

impl FeatureCombiner for VerticalCombiner {
    fn combine(&self, u: &DMatrix<f64>, v: &DMatrix<f64>) -> DMatrix<f64> {
        let rows = u.nrows();
        let cols = u.ncols();
        
        DMatrix::from_fn(rows + v.nrows(), cols, |r, c| {
            if r < rows {
                u[(r, c)]
            } else {
                v[(r - rows, c)]
            }
        })
    }
}

/// 加权特征组合器
pub struct WeightedCombiner {
    pub row_weight: f64,
    pub col_weight: f64,
}

impl FeatureCombiner for WeightedCombiner {
    fn combine(&self, u: &DMatrix<f64>, v: &DMatrix<f64>) -> DMatrix<f64> {
        let rows = u.nrows();
        let cols = u.ncols();
        
        DMatrix::from_fn(rows + v.nrows(), cols, |r, c| {
            if r < rows {
                u[(r, c)] * self.row_weight
            } else {
                v[(r - rows, c)] * self.col_weight
            }
        })
    }
}

/// K-means聚类分配器
pub struct KMeansAssigner;

impl ClusterAssigner for KMeansAssigner {
    fn assign(&self, features: &DMatrix<f64>, k: usize) -> Result<Vec<usize>, Box<dyn Error>> {
        let f_data: Vec<f64> = features.transpose().data.as_slice().iter().copied().collect();
        let kmeans_f: KMeans<f64, 8> = KMeans::new(f_data, features.nrows(), features.ncols());
        
        let result = kmeans_f.kmeans_lloyd(k, 100, KMeans::init_kmeanplusplus, &KMeansConfig::default());
        Ok(result.assignments)
    }
}

/// 谱聚类分配器 (简化版本)
pub struct SpectralAssigner;

impl ClusterAssigner for SpectralAssigner {
    fn assign(&self, features: &DMatrix<f64>, k: usize) -> Result<Vec<usize>, Box<dyn Error>> {
        // 简化的谱聚类实现 - 实际应用中需要更完善的实现
        // 这里为了演示，我们回退到K-means
        let kmeans_assigner = KMeansAssigner;
        kmeans_assigner.assign(features, k)
    }
}

// ============ 模块化Coclusterer ============

/// 模块化的Coclusterer，支持组件替换
pub struct ModularCoclusterer {
    matrix: Array2<f64>,
    k: usize,
    normalizer: Box<dyn MatrixNormalizer>,
    reducer: Box<dyn DimensionalityReducer>,
    combiner: Box<dyn FeatureCombiner>,
    assigner: Box<dyn ClusterAssigner>,
}

impl ModularCoclusterer {
    /// 创建新的模块化Coclusterer
    pub fn new(
        matrix: Array2<f64>,
        k: usize,
        normalizer: Box<dyn MatrixNormalizer>,
        reducer: Box<dyn DimensionalityReducer>,
        combiner: Box<dyn FeatureCombiner>,
        assigner: Box<dyn ClusterAssigner>,
    ) -> Self {
        Self {
            matrix,
            k,
            normalizer,
            reducer,
            combiner,
            assigner,
        }
    }
    
    /// 使用默认组件的构造器
    pub fn with_defaults(matrix: Array2<f64>, k: usize) -> Self {
        Self::new(
            matrix,
            k,
            Box::new(StandardNormalizer),
            Box::new(SVDReducer),
            Box::new(VerticalCombiner),
            Box::new(KMeansAssigner),
        )
    }
    
    /// 使用Z-score归一化的构造器
    pub fn with_zscore(matrix: Array2<f64>, k: usize) -> Self {
        Self::new(
            matrix,
            k,
            Box::new(ZScoreNormalizer),
            Box::new(SVDReducer),
            Box::new(VerticalCombiner),
            Box::new(KMeansAssigner),
        )
    }
    
    /// 使用加权特征组合的构造器
    pub fn with_weighted_features(matrix: Array2<f64>, k: usize, row_weight: f64, col_weight: f64) -> Self {
        Self::new(
            matrix,
            k,
            Box::new(StandardNormalizer),
            Box::new(SVDReducer),
            Box::new(WeightedCombiner { row_weight, col_weight }),
            Box::new(KMeansAssigner),
        )
    }
    
    /// 运行模块化co-clustering
    pub fn cocluster(&mut self) -> Result<Vec<usize>, Box<dyn Error>> {
        // 1. 转换矩阵格式
        let na_matrix = clone_to_dmatrix(self.matrix.view());
        
        // 2. 矩阵归一化 (可替换)
        let normalized_matrix = self.normalizer.normalize(&na_matrix);
        
        // 3. 降维 (可替换)
        let (u_mat, v_mat) = self.reducer.reduce(&normalized_matrix, self.k)?;
        
        // 4. 特征组合 (可替换)
        let combined_features = self.combiner.combine(&u_mat, &v_mat);
        
        // 5. 聚类分配 (可替换)
        let assignments = self.assigner.assign(&combined_features, self.k)?;
        
        Ok(assignments)
    }
}

// ============ Builder模式 ============

pub struct ModularCoclustererBuilder {
    matrix: Option<Array2<f64>>,
    k: Option<usize>,
    normalizer: Option<Box<dyn MatrixNormalizer>>,
    reducer: Option<Box<dyn DimensionalityReducer>>,
    combiner: Option<Box<dyn FeatureCombiner>>,
    assigner: Option<Box<dyn ClusterAssigner>>,
}

impl ModularCoclustererBuilder {
    pub fn new() -> Self {
        Self {
            matrix: None,
            k: None,
            normalizer: None,
            reducer: None,
            combiner: None,
            assigner: None,
        }
    }
    
    pub fn matrix(mut self, matrix: Array2<f64>) -> Self {
        self.matrix = Some(matrix);
        self
    }
    
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }
    
    pub fn normalizer(mut self, normalizer: Box<dyn MatrixNormalizer>) -> Self {
        self.normalizer = Some(normalizer);
        self
    }
    
    pub fn reducer(mut self, reducer: Box<dyn DimensionalityReducer>) -> Self {
        self.reducer = Some(reducer);
        self
    }
    
    pub fn combiner(mut self, combiner: Box<dyn FeatureCombiner>) -> Self {
        self.combiner = Some(combiner);
        self
    }
    
    pub fn assigner(mut self, assigner: Box<dyn ClusterAssigner>) -> Self {
        self.assigner = Some(assigner);
        self
    }
    
    pub fn build(self) -> Result<ModularCoclusterer, &'static str> {
        let matrix = self.matrix.ok_or("Matrix not set")?;
        let k = self.k.ok_or("K not set")?;
        
        Ok(ModularCoclusterer::new(
            matrix,
            k,
            self.normalizer.unwrap_or_else(|| Box::new(StandardNormalizer)),
            self.reducer.unwrap_or_else(|| Box::new(SVDReducer)),
            self.combiner.unwrap_or_else(|| Box::new(VerticalCombiner)),
            self.assigner.unwrap_or_else(|| Box::new(KMeansAssigner)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    
    #[test]
    fn test_modular_cocluster_with_defaults() {
        let test_matrix: Array2<f64> = Array2::random((20, 15), Uniform::new(0.0, 1.0));
        let mut coclusterer = ModularCoclusterer::with_defaults(test_matrix, 3);
        
        let result = coclusterer.cocluster();
        assert!(result.is_ok());
        
        let assignments = result.unwrap();
        assert_eq!(assignments.len(), 20 + 15); // rows + cols
    }
    
    #[test]
    fn test_modular_cocluster_with_zscore() {
        let test_matrix: Array2<f64> = Array2::random((10, 8), Uniform::new(0.0, 1.0));
        let mut coclusterer = ModularCoclusterer::with_zscore(test_matrix, 2);
        
        let result = coclusterer.cocluster();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_modular_cocluster_builder() {
        let test_matrix: Array2<f64> = Array2::random((15, 10), Uniform::new(0.0, 1.0));
        
        let mut coclusterer = ModularCoclustererBuilder::new()
            .matrix(test_matrix)
            .k(3)
            .normalizer(Box::new(ZScoreNormalizer))
            .reducer(Box::new(SVDReducer))
            .combiner(Box::new(WeightedCombiner { row_weight: 0.7, col_weight: 0.3 }))
            .assigner(Box::new(KMeansAssigner))
            .build()
            .unwrap();
        
        let result = coclusterer.cocluster();
        assert!(result.is_ok());
    }
}