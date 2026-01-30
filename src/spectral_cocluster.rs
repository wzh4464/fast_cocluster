// use crate::{Matrix, Submatrix};
use crate::submatrix::Submatrix;
use crate::matrix::Matrix;
use linfa::traits::Transformer;
use linfa_clustering::KMeans;
use linfa::prelude::{Fit, Predict};
use log::debug;

// src/spectral_cocluster.rs
use nalgebra::{DMatrix, DVector, SVD};
use ndarray::{Array1, Array2};
use crate::util::clone_to_dmatrix;

pub struct SpectralCocluster {
    n_clusters: (usize, usize),
    n_init: usize,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
}

impl SpectralCocluster {
    pub fn new(n_row_clusters: usize, n_col_clusters: usize) -> Self {
        Self {
            n_clusters: (n_row_clusters, n_col_clusters),
            n_init: 10,
            max_iter: 300,
            tol: 1e-4,
            random_state: None,
        }
    }

    // 辅助函数：将 ndarray::Array2 转换为 nalgebra::DMatrix
    fn array2_to_dmatrix(array2: &Array2<f64>) -> DMatrix<f64> {
        let nrows = array2.nrows();
        let ncols = array2.ncols();
        let data_vec: Vec<f64> = array2.iter().cloned().collect();
        DMatrix::from_vec(nrows, ncols, data_vec)
    }

    pub fn fit<'matrix_life>(
        &self, 
        matrix: &'matrix_life Matrix<f64>
    ) -> Result<Vec<Submatrix<'matrix_life, f64>>, Box<dyn std::error::Error>> {
        let matrix_array2 = &matrix.data; // matrix.data is Array2<f64>
        let data_dmatrix = Self::array2_to_dmatrix(matrix_array2); // 转换为 DMatrix for nalgebra operations

        let (n_rows, n_cols) = (data_dmatrix.nrows(), data_dmatrix.ncols());

        // Step 1: 构建二部图的拉普拉斯矩阵
        let laplacian = self.build_laplacian(&data_dmatrix);

        // Step 2: 计算特征向量
        let svd = SVD::new(laplacian, true, true);
        let u = svd.u.ok_or("Failed to compute U matrix")?;
        let v_t = svd.v_t.ok_or("Failed to compute V^T matrix")?;

        // Step 3: 选择前k个特征向量
        let n_features = self.n_clusters.0.min(self.n_clusters.1);
        let row_features = u.columns(1, n_features).clone_owned();
        let col_features = v_t.transpose().columns(1, n_features).clone_owned();

        // Step 4: 对行和列分别进行k-means聚类
        let row_labels = self.kmeans_cluster(&row_features, self.n_clusters.0)?;
        let col_labels = self.kmeans_cluster(&col_features, self.n_clusters.1)?;

        // Step 5: 构建子矩阵
        // Submatrix::from_indices 需要 &Array2<f64>, 我们已经有了 data_array2
        let submatrices = self.build_submatrices(matrix_array2, &row_labels, &col_labels, n_rows, n_cols);

        Ok(submatrices)
    }

    fn build_laplacian(&self, data: &DMatrix<f64>) -> DMatrix<f64> {
        let (n_rows, n_cols) = (data.nrows(), data.ncols());
        let total = n_rows + n_cols;

        // 构建邻接矩阵 A = [0, B; B^T, 0]
        let mut adjacency = DMatrix::zeros(total, total);

        // 上右块放置原始矩阵B
        for i in 0..n_rows {
            for j in 0..n_cols {
                adjacency[(i, n_rows + j)] = data[(i, j)];
            }
        }

        // 下左块放置B的转置
        for i in 0..n_rows {
            for j in 0..n_cols {
                adjacency[(n_rows + j, i)] = data[(i, j)];
            }
        }

        // 计算度矩阵
        let degrees: Vec<f64> = (0..total).map(|i| adjacency.row(i).sum()).collect();

        // 构建归一化拉普拉斯矩阵
        let mut laplacian = DMatrix::zeros(total, total);
        for i in 0..total {
            for j in 0..total {
                if degrees[i] > 0.0 && degrees[j] > 0.0 {
                    laplacian[(i, j)] = adjacency[(i, j)] / (degrees[i] * degrees[j]).sqrt();
                }
            }
        }

        laplacian
    }

    fn kmeans_cluster(
        &self,
        features: &DMatrix<f64>,
        n_clusters: usize,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        // 转换为ndarray格式
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let mut arr = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            for j in 0..n_features {
                arr[[i, j]] = features[(i, j)];
            }
        }

        // 使用linfa进行k-means聚类
        let dataset = linfa::Dataset::from(arr);
        let model = KMeans::params(n_clusters)
            .n_runs(self.n_init)
            .max_n_iterations(self.max_iter as u64)
            .tolerance(self.tol)
            .fit(&dataset)?;

        Ok(model.predict(dataset.records()).to_vec())
    }

    fn build_submatrices<'matrix_life>(
        &self,
        original_matrix_array2: &'matrix_life Array2<f64>,
        row_labels: &[usize],
        col_labels: &[usize],
        n_rows: usize,
        n_cols: usize,
    ) -> Vec<Submatrix<'matrix_life, f64>> {
        // 对每个行簇和列簇的组合创建子矩阵
        // Parallelize submatrix creation using Rayon
        use rayon::prelude::*;
        
        // Create all cluster pairs
        let cluster_pairs: Vec<(usize, usize)> = (0..self.n_clusters.0)
            .flat_map(|r| (0..self.n_clusters.1).map(move |c| (r, c)))
            .collect();
        
        // Process cluster pairs in parallel
        let submatrices: Vec<_> = cluster_pairs
            .par_iter()
            .filter_map(|&(row_cluster, col_cluster)| {
                let rows: Vec<usize> = (0..n_rows)
                    .filter(|&i| row_labels[i] == row_cluster)
                    .collect();
                let cols: Vec<usize> = (0..n_cols)
                    .filter(|&j| col_labels[j] == col_cluster)
                    .collect();

                if !rows.is_empty() && !cols.is_empty() {
                    crate::submatrix::Submatrix::from_indices(original_matrix_array2, &rows, &cols)
                } else {
                    None
                }
            })
            .collect();

        submatrices
    }
}

// 添加到Cargo.toml的依赖：
// nalgebra = "0.32"
// ndarray = "0.15"
// linfa = "0.7"
// linfa-clustering = "0.7"
