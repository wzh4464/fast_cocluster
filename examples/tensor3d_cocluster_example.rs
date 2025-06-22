use fast_cocluster::tensor3d::*;
use fast_cocluster::tensor3d_scoring::*;
use fast_cocluster::tensor3d_cocluster::*;
use fast_cocluster::tucker_decomposition::*;
use ndarray::Array3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("=== 3D张量Co-clustering示例 ===\n");
    
    // 创建一个具有明显结构的3D张量
    println!("1. 创建结构化3D张量:");
    let tensor = create_structured_tensor([20, 15, 12]);
    println!("   张量形状: {:?}", tensor.shape());
    println!("   Frobenius范数: {:.3}", tensor.frobenius_norm());
    
    // 演示Tucker分解
    println!("\n2. Tucker分解:");
    let tucker_rank = TuckerRank::new(3, 3, 3);
    let mut config = TuckerConfig::default();
    config.ranks = tucker_rank.clone();
    config.max_iterations = 10; // 减少迭代次数
    
    let decomposer = TuckerDecomposer::new(config);
    match decomposer.decompose(&tensor) {
        Ok(decomposition) => {
            println!("   ✅ Tucker分解成功!");
            println!("   核心张量形状: {:?}", decomposition.core_tensor.shape());
            println!("   重构误差: {:.6}", decomposition.reconstruction_error);
            
            // 演示重构
            match decomposition.reconstruct() {
                Ok(reconstructed) => {
                    println!("   ✅ 张量重构成功!");
                    let reconstruction_error = decomposition.compute_reconstruction_error(&tensor)?;
                    println!("   验证重构误差: {:.6}", reconstruction_error);
                },
                Err(e) => println!("   ❌ 重构失败: {}", e),
            }
        },
        Err(e) => println!("   ❌ Tucker分解失败: {}", e),
    }
    
    // 演示3D张量评分
    println!("\n3. 3D张量评分:");
    demonstrate_tensor_scoring(&tensor)?;
    
    // 演示简化的co-clustering
    println!("\n4. 简化3D Co-clustering:");
    demonstrate_simple_cocluster(&tensor)?;
    
    // 演示不同的张量结构
    println!("\n5. 不同张量结构的分析:");
    analyze_different_structures()?;
    
    Ok(())
}

/// 创建具有明显块结构的3D张量
fn create_structured_tensor(shape: [usize; 3]) -> Tensor3D<f64> {
    let mut data = Array3::zeros(shape);
    
    // 创建几个明显的块结构
    
    // 块1: 高值区域
    for i in 0..5.min(shape[0]) {
        for j in 0..5.min(shape[1]) {
            for k in 0..4.min(shape[2]) {
                data[[i, j, k]] = 5.0 + (i + j + k) as f64 * 0.1;
            }
        }
    }
    
    // 块2: 中等值区域
    for i in 8..13.min(shape[0]) {
        for j in 6..11.min(shape[1]) {
            for k in 5..9.min(shape[2]) {
                data[[i, j, k]] = 3.0 + (i + j + k) as f64 * 0.05;
            }
        }
    }
    
    // 块3: 低值但一致的区域
    for i in 15..shape[0] {
        for j in 12..shape[1] {
            for k in 9..shape[2] {
                data[[i, j, k]] = 1.0 + 0.1 * ((i * j * k) as f64).sin();
            }
        }
    }
    
    // 添加一些噪声
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                if data[[i, j, k]] == 0.0 {
                    data[[i, j, k]] = 0.1 * rand::random::<f64>();
                }
            }
        }
    }
    
    Tensor3D::from_data(data)
}

/// 演示张量评分功能
fn demonstrate_tensor_scoring(tensor: &Tensor3D<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // 创建几个测试子空间
    let subspace1 = TensorSubspace::new(tensor, vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]).unwrap();
    let subspace2 = TensorSubspace::new(tensor, vec![8, 9, 10], vec![6, 7, 8], vec![5, 6, 7]).unwrap();
    let subspace3 = TensorSubspace::new(tensor, vec![15, 16, 17], vec![12, 13, 14], vec![9, 10, 11]).unwrap();
    
    // 使用不同的评分器
    let scorers: Vec<(Box<dyn TensorScorer>, &str)> = vec![
        (Box::new(TuckerScorer::with_uniform_rank(2)), "Tucker评分器"),
        (Box::new(DensityScorer::new(0.5)), "密度评分器"),
        (Box::new(VarianceScorer::new(2.0)), "方差评分器"),
    ];
    
    let subspaces = [&subspace1, &subspace2, &subspace3];
    let subspace_names = ["块1 (高值)", "块2 (中值)", "块3 (低值)"];
    
    for (scorer, scorer_name) in scorers {
        println!("   使用{}:", scorer_name);
        for (i, subspace) in subspaces.iter().enumerate() {
            let score = scorer.score(tensor, subspace);
            println!("     {}: {:.4}", subspace_names[i], score);
        }
    }
    
    Ok(())
}

/// 演示简化的co-clustering过程
fn demonstrate_simple_cocluster(tensor: &Tensor3D<f64>) -> Result<(), Box<dyn std::error::Error>> {
    // 创建简化的聚类器（避免完整的Tucker分解）
    println!("   使用基于密度的聚类方法:");
    
    let shape = tensor.shape();
    let mut high_value_regions = Vec::new();
    
    // 寻找高密度区域
    let threshold = 2.0;
    for i in 0..(shape[0] - 2) {
        for j in 0..(shape[1] - 2) {
            for k in 0..(shape[2] - 2) {
                // 检查3x3x3的小块
                let mut sum = 0.0;
                let mut count = 0;
                
                for di in 0..3 {
                    for dj in 0..3 {
                        for dk in 0..3 {
                            if i + di < shape[0] && j + dj < shape[1] && k + dk < shape[2] {
                                sum += tensor.data[[i + di, j + dj, k + dk]];
                                count += 1;
                            }
                        }
                    }
                }
                
                let avg = sum / count as f64;
                if avg > threshold {
                    let rows = (i..i+3).collect();
                    let cols = (j..j+3).collect();
                    let depths = (k..k+3).collect();
                    
                    if let Some(subspace) = TensorSubspace::new(tensor, rows, cols, depths) {
                        high_value_regions.push((subspace, avg));
                    }
                }
            }
        }
    }
    
    // 按平均值排序
    high_value_regions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // 显示前几个高质量区域
    println!("   找到{}个高密度区域:", high_value_regions.len());
    for (i, (subspace, avg)) in high_value_regions.iter().take(5).enumerate() {
        let shape = subspace.shape();
        println!("     区域{}: {}×{}×{}, 平均值: {:.3}", 
                 i+1, shape[0], shape[1], shape[2], avg);
        
        // 计算该区域的详细评分
        let tucker_scorer = TuckerScorer::with_uniform_rank(2);
        let score = tucker_scorer.score(tensor, subspace);
        println!("       Tucker分数: {:.4}", score);
    }
    
    Ok(())
}

/// 分析不同结构的张量
fn analyze_different_structures() -> Result<(), Box<dyn std::error::Error>> {
    let structures = [
        ("随机张量", create_random_tensor([10, 8, 6])),
        ("对角张量", create_diagonal_tensor([10, 8, 6])),
        ("分层张量", create_layered_tensor([10, 8, 6])),
    ];
    
    for (name, tensor) in structures {
        println!("   分析{}:", name);
        println!("     Frobenius范数: {:.3}", tensor.frobenius_norm());
        
        // 尝试Tucker分解
        let tucker_rank = TuckerRank::uniform(2);
        let mut config = TuckerConfig::default();
        config.ranks = tucker_rank;
        config.max_iterations = 5;
        
        let decomposer = TuckerDecomposer::new(config);
        match decomposer.decompose(&tensor) {
            Ok(decomposition) => {
                println!("     Tucker分解成功, 重构误差: {:.6}", decomposition.reconstruction_error);
                
                // 分析因子矩阵的特性
                let f1_norm = decomposition.factor_matrix1.norm();
                let f2_norm = decomposition.factor_matrix2.norm();
                let f3_norm = decomposition.factor_matrix3.norm();
                println!("     因子矩阵范数: {:.3}, {:.3}, {:.3}", f1_norm, f2_norm, f3_norm);
            },
            Err(e) => println!("     Tucker分解失败: {}", e),
        }
    }
    
    Ok(())
}

/// 创建随机张量
fn create_random_tensor(shape: [usize; 3]) -> Tensor3D<f64> {
    Tensor3D::random(shape)
}

/// 创建对角结构张量
fn create_diagonal_tensor(shape: [usize; 3]) -> Tensor3D<f64> {
    let mut data = Array3::zeros(shape);
    let min_dim = shape[0].min(shape[1]).min(shape[2]);
    
    for i in 0..min_dim {
        data[[i, i, i]] = 5.0;
        
        // 添加一些对角邻近元素
        if i + 1 < shape[0] && i + 1 < shape[1] && i + 1 < shape[2] {
            data[[i + 1, i, i]] = 2.0;
            data[[i, i + 1, i]] = 2.0;
            data[[i, i, i + 1]] = 2.0;
        }
    }
    
    Tensor3D::from_data(data)
}

/// 创建分层结构张量
fn create_layered_tensor(shape: [usize; 3]) -> Tensor3D<f64> {
    let mut data = Array3::zeros(shape);
    
    for k in 0..shape[2] {
        let layer_value = (k as f64 + 1.0) / shape[2] as f64 * 3.0;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                data[[i, j, k]] = layer_value + 0.1 * (i + j) as f64;
            }
        }
    }
    
    Tensor3D::from_data(data)
}