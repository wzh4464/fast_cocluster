# Fast Co-clustering (DiMergeCo)

## Development Workflow

- **Local machine**: Edit code, `cargo check` only (no OpenBLAS, linking fails)
- **Server**: Run full benchmarks and evaluations via `cargo run --release`
- **Sync**: Server pulls code from local via `git pull`
- Local cannot run `cargo test` or `cargo run` (需要 OpenBLAS 链接)
- Local 验证用 `cargo check --lib --tests --examples`

## Quick Reference

```bash
# Build
cargo build --release

# Run evaluation
cargo run --release --example evaluate_classic4

# Run tests (local-safe)
cargo test

# Run with logging enabled
RUST_LOG=info cargo run --release --example evaluate_classic4
```

## PR Review 反复出现的问题（编码规范）

以下问题在多个 PR review 中反复出现，写新代码时务必避免：

### 1. pred_labels 必须从 row_labels 获取，不能从 submatrices 遍历
`cluster_local` 返回的是 co-clusters (行簇 × 列簇)，每行会出现在多个 submatrix 中（每个列簇一次）。遍历 submatrices 赋 `pred_labels[r] = cid` 会被后面的覆盖，结果不正确。
```rust
// WRONG — 行会被多次覆盖
for (cid, sub) in submatrices.iter().enumerate() {
    for &r in &sub.row_indices { pred_labels[r] = cid; }
}
// RIGHT — 直接用 tri-factorization 的行标签
let result = clusterer.run_tri_factorization(&x)?;
let pred_labels = result.row_labels;
```

### 2. 大矩阵不要 clone，用 move
`Matrix::new(array.clone())` 会让 RCV1/MovieLens 级矩阵内存翻倍，可能 OOM。应 move array 进 Matrix，后续用 `&matrix.data` 借用。
```rust
// WRONG — 内存翻倍
let matrix = Matrix::new(array.clone());
// RIGHT — 只保留一份
let matrix = Matrix::new(array);
let array_ref = &matrix.data;
```

### 3. NNLS success 标志必须检查
`nnlsm_blockpivot` 返回的 `success` 标志不能忽略。求解失败时应中止当前初始化或回退。

### 4. ARI 计算需要 n<2 保护
`comb2(n)=0` 当 `n<2` 时导致除零/NaN。在计算 `comb_n` 前加 `if n < 2 { return 0.0; }`。

### 5. i64→usize 转换必须校验非负
标签从 npy 加载为 i64 时，直接 `as usize` 会静默环绕负值为极大正数。转换前 assert 或 `try_into()`。

### 6. NNLS 线性求解用 LAPACK，不要手写高斯消元
手写消元在 `A^T A` 等病态矩阵上数值不稳定。使用 ndarray-linalg 的 LU/SVD 求解，失败时做 least-squares 回退。

### 7. NMI/ARI 等工具函数不要在每个 example 中重复
提取到共享模块或 crate 内的 pub 工具函数中，避免多处维护。

### 8. n_init 和其他配置参数需校验
`n_init=0` 会导致循环不执行、后续 `unwrap()` panic。公开字段需要校验 `>= 1`。

### 9. 避免在 NMF 更新中产生 O(m²) 或 O(n²) 中间矩阵
将乘法链重新结合，使所有中间矩阵为 k×k 或 m×k/n×k 级别。例如 ONMTF 的 `F*S*G^T*G*S^T*F^T*F` 应拆为小矩阵连乘。

### 10. 脚本中不要硬编码绝对路径
`/home/jie/fast_cocluster` 等路径不可移植。用 `__file__` 相对路径或 `git rev-parse --show-toplevel`。
