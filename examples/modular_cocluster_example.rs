use fast_cocluster::modular_cocluster::*;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("=== 模块化 Co-clustering 示例 ===\n");
    
    // 创建测试数据
    let test_matrix: Array2<f64> = Array2::random((50, 30), Uniform::new(0.0, 10.0));
    println!("创建了 {}×{} 的测试矩阵", test_matrix.nrows(), test_matrix.ncols());
    
    // 方法1: 使用默认组件
    println!("\n1. 使用标准归一化 + SVD + K-means:");
    let mut coclusterer1 = ModularCoclusterer::with_defaults(test_matrix.clone(), 5);
    let result1 = coclusterer1.cocluster()?;
    println!("   找到的聚类数: {}", result1.len());
    println!("   前10个分配: {:?}", &result1[..10.min(result1.len())]);
    
    // 方法2: 使用Z-score归一化
    println!("\n2. 使用Z-score归一化:");
    let mut coclusterer2 = ModularCoclusterer::with_zscore(test_matrix.clone(), 5);
    let result2 = coclusterer2.cocluster()?;
    println!("   找到的聚类数: {}", result2.len());
    println!("   前10个分配: {:?}", &result2[..10.min(result2.len())]);
    
    // 方法3: 使用加权特征组合
    println!("\n3. 使用加权特征组合 (行权重0.7, 列权重0.3):");
    let mut coclusterer3 = ModularCoclusterer::with_weighted_features(test_matrix.clone(), 5, 0.7, 0.3);
    let result3 = coclusterer3.cocluster()?;
    println!("   找到的聚类数: {}", result3.len());
    println!("   前10个分配: {:?}", &result3[..10.min(result3.len())]);
    
    // 方法4: 使用Builder模式自定义组件
    println!("\n4. 使用Builder模式自定义组件:");
    let mut coclusterer4 = ModularCoclustererBuilder::new()
        .matrix(test_matrix.clone())
        .k(3)
        .normalizer(Box::new(ZScoreNormalizer))
        .reducer(Box::new(SVDReducer))
        .combiner(Box::new(WeightedCombiner { row_weight: 0.6, col_weight: 0.4 }))
        .assigner(Box::new(KMeansAssigner))
        .build()?;
    
    let result4 = coclusterer4.cocluster()?;
    println!("   找到的聚类数: {}", result4.len());
    println!("   前10个分配: {:?}", &result4[..10.min(result4.len())]);
    
    // 比较不同方法的结果
    println!("\n=== 结果比较 ===");
    println!("标准方法聚类分布: {:?}", count_clusters(&result1, 5));
    println!("Z-score聚类分布: {:?}", count_clusters(&result2, 5));
    println!("加权特征聚类分布: {:?}", count_clusters(&result3, 5));
    println!("自定义组件聚类分布: {:?}", count_clusters(&result4, 3));
    
    Ok(())
}

/// 统计每个聚类的元素数量
fn count_clusters(assignments: &[usize], k: usize) -> Vec<usize> {
    let mut counts = vec![0; k];
    for &assignment in assignments {
        if assignment < k {
            counts[assignment] += 1;
        }
    }
    counts
}

/// 演示如何创建自定义组件
#[allow(dead_code)]
fn demonstrate_custom_components() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 自定义组件演示 ===");
    
    // 自定义归一化器
    struct CustomNormalizer;
    impl MatrixNormalizer for CustomNormalizer {
        fn normalize(&self, matrix: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
            // 简单的min-max归一化
            let min_val = matrix.min();
            let max_val = matrix.max();
            let range = max_val - min_val;
            
            if range > 0.0 {
                matrix.map(|x| (x - min_val) / range)
            } else {
                matrix.clone()
            }
        }
    }
    
    // 自定义特征组合器
    struct CustomCombiner;
    impl FeatureCombiner for CustomCombiner {
        fn combine(&self, u: &nalgebra::DMatrix<f64>, v: &nalgebra::DMatrix<f64>) -> nalgebra::DMatrix<f64> {
            // 交替组合行和列特征
            let total_rows = u.nrows() + v.nrows();
            let cols = u.ncols();
            
            nalgebra::DMatrix::from_fn(total_rows, cols, |r, c| {
                if r % 2 == 0 && r / 2 < u.nrows() {
                    u[(r / 2, c)]
                } else if r % 2 == 1 && (r - 1) / 2 < v.nrows() {
                    v[((r - 1) / 2, c)]
                } else {
                    0.0
                }
            })
        }
    }
    
    let test_matrix: Array2<f64> = Array2::random((20, 15), Uniform::new(0.0, 1.0));
    
    let mut custom_coclusterer = ModularCoclustererBuilder::new()
        .matrix(test_matrix)
        .k(4)
        .normalizer(Box::new(CustomNormalizer))
        .combiner(Box::new(CustomCombiner))
        .build()?;
    
    let result = custom_coclusterer.cocluster()?;
    println!("自定义组件结果聚类数: {}", result.len());
    println!("聚类分布: {:?}", count_clusters(&result, 4));
    
    Ok(())
}