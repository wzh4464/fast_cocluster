# DiMergeCo Benchmark Findings

> Last updated: 2026-03-05
> Branch: `dimerge-atom-integration` (PR #9)
> All results are mean +/- std over 10 seeds unless stated otherwise.

---

## Table of Contents

1. [Datasets](#1-datasets)
2. [Standalone Baseline Comparison](#2-standalone-baseline-comparison)
3. [DiMergeCo Integration Results](#3-dimergeco-integration-results)
4. [Grid Size Sensitivity](#4-grid-size-sensitivity)
5. [Merge Strategy Ablation](#5-merge-strategy-ablation)
6. [Large-Scale MovieLens Experiments](#6-large-scale-movielens-experiments)
7. [Performance Optimizations](#7-performance-optimizations)
8. [Technical Findings](#8-technical-findings)
9. [Open Problems and Next Steps](#9-open-problems-and-next-steps)

---

## 1. Datasets

| Dataset | Shape | k | Sparsity | Source |
|---------|-------|---|----------|--------|
| BCW (Breast Cancer Wisconsin) | 569 x 30 | 2 | Dense | UCI |
| Classic4 | 6,460 x 4,667 | 4 | Sparse (TF-IDF) | Text corpora |
| RCV1-train | 23,149 x 47,236 | 4 | Sparse | Reuters |
| RCV1-test | 781,265 x 47,236 | 4 | Sparse | Reuters |
| RCV1-all | 804,414 x 47,236 | 4 | Sparse | Reuters |
| MovieLens 25M | 162,541 x 59,047 | 20 | Dense (ratings) | GroupLens |

---

## 2. Standalone Baseline Comparison

### 2.1 Classic4 (6460 x 4667, k=4)

Python baselines (10 seeds):

| Method | NMI | ARI | Time (s) | Reference |
|--------|-----|-----|----------|-----------|
| **SCC-Dhillon** | **0.7903 +/- 0.0002** | **0.7490 +/- 0.0004** | 1.50 | Dhillon 2001 |
| FNMF | 0.6094 +/- 0.0465 | 0.5357 +/- 0.0987 | 0.98 | Kim & Park 2011 |
| SpectralCC (sklearn) | 0.4313 +/- 0.0005 | 0.2727 +/- 0.0027 | 0.27 | sklearn |

Notes:
- SCC-Dhillon is our Rust implementation's reference; matches Python closely.
- FNMF is bimodal: converges to NMI ~0.578 (7/10 seeds) or ~0.681 (3/10 seeds), indicating sensitivity to initialization.
- SpectralCC (sklearn's `SpectralCoclustering`) uses log(k)+1 SVs vs our k SVs, explaining the large gap.

### 2.2 BCW (569 x 30, k=2)

| Method | NMI | ARI | Time (s) |
|--------|-----|-----|----------|
| **SCC-Dhillon** | **0.4911 +/- 0.0000** | **0.6079 +/- 0.0000** | 0.45 |
| PNMTF | 0.4019 +/- 0.1277 | 0.4916 +/- 0.1546 | 2.17 |
| FNMF | 0.3475 +/- 0.0042 | 0.3527 +/- 0.0054 | 0.04 |
| SpectralCC | 0.1278 +/- 0.1649 | 0.1258 +/- 0.1906 | 0.22 |
| ONM3F | 0.0350 +/- 0.0099 | 0.0343 +/- 0.0374 | 15.68 |
| ONMTF | 0.0334 +/- 0.0006 | 0.0216 +/- 0.0370 | 7.28 |
| NBVD | 0.0314 +/- 0.0112 | 0.0105 +/- 0.0412 | 2.35 |

Notes:
- SCC-Dhillon is deterministic on BCW (all 10 seeds identical).
- PNMTF occasionally fails (seed 2: NMI=0.038), but is strong otherwise (NMI=0.442).
- SpectralCC is highly unstable on this small dense dataset.
- NMF methods (NBVD, ONM3F, ONMTF) all perform poorly on BCW.

### 2.3 RCV1 (Sparse, k=4)

Only FNMF baselines available in Python (other NMF methods too slow for full matrix):

| Split | Shape | NMI | ARI | Time (s) |
|-------|-------|-----|-----|----------|
| train | 23K x 47K | 0.3110 +/- 0.0078 | 0.1898 +/- 0.0058 | 12.1 |
| test | 781K x 47K | 0.2964 +/- 0.0017 | 0.1519 +/- 0.0030 | 612.4 |
| all | 804K x 47K | 0.2963 +/- 0.0020 | 0.1537 +/- 0.0038 | 629.8 |

Rust implementation results on RCV1-train (from PR #9 commits):

| Method | NMI | ARI | Time (s) | Notes |
|--------|-----|-----|----------|-------|
| NBVD | 0.2773 | 0.1740 | ~530/block | 8x8 grid, iter=10 |
| PNMTF | 0.1169 | - | - | tau=0.1 |
| ONM3F | 0.0923 | - | - | |
| FNMF | 0.0003 | - | - | Was broken (wrong signature), later fixed |

---

## 3. DiMergeCo Integration Results

### 3.1 Classic4 - DiMergeCo + SCC-Dhillon (2x2, T_p=30)

| Method | NMI | ARI | Time (s) |
|--------|-----|-----|----------|
| SCC Standalone | 0.7903 +/- 0.0002 | 0.7490 +/- 0.0004 | 1.50 |
| **DiMergeCo + SCC** | **0.7479 +/- 0.0033** | **0.6634 +/- 0.0077** | 44.3 |

- **Quality retention: 94.6% NMI, 88.6% ARI** relative to standalone.
- Time overhead: ~30x slower due to merge process with T_p=30 repetitions.
- Standard deviation is low (0.003), showing DiMergeCo is stable.

### 3.2 BCW - DiMergeCo + SCC-Dhillon (2x2, T_p=10)

| Method | NMI | ARI | Time (s) |
|--------|-----|-----|----------|
| SCC Standalone | 0.4911 +/- 0.0000 | 0.6079 +/- 0.0000 | 0.45 |
| **DiMergeCo + SCC** | **0.4682 +/- 0.0189** | **0.5855 +/- 0.0148** | 8.47 |

- **Quality retention: 95.3% NMI, 96.3% ARI**.
- Slight quality loss is expected from the block partitioning.

### 3.3 RCV1-train - DiMergeCo + SCC-Dhillon (2x2, T_p=10)

| Method | NMI | ARI | Time (s) |
|--------|-----|-----|----------|
| DiMergeCo + SCC | **~0.0001** | **~0.0** | 374 |

**COLLAPSED.** The DiMergeCo framework fails on RCV1-train with SCC as the atom method. Likely cause: when sub-blocks of a sparse matrix are too dilute, spectral co-clustering produces trivial partitions, and the merge step cannot recover meaningful structure.

### 3.4 RCV1-train - DiMergeCo + NMF methods (Rust, PR #9 era)

Earlier experiments with the Rust implementation (from PR #9 commit messages):

| Method | NMI | ARI | Time (s) | Config |
|--------|-----|-----|----------|--------|
| DiMergeCo + spectral | 0.0003 | -0.0040 | 52.6 | 8x8 |
| DiMergeCo + spectral | 0.4137 | 0.3134 | 2.7 | 4x4 (Classic4) |
| DiMergeCo + NBVD | 0.5395 | 0.4269 | 1293 | 4x4 (Classic4) |
| DiMergeCo + ONM3F | 0.5324 | 0.4170 | 1329 | 4x4 (Classic4) |
| DiMergeCo + ONMTF | 0.5307 | 0.5229 | 90.4 | 2x2 (Classic4-small) |
| DiMergeCo + PNMTF | 0.5004 | 0.4871 | 21.9 | 2x2 (Classic4-small) |

Best RCV1-all result with parameter sweep:
- **NBVD, T_p=20, 6x6 grid: NMI=0.1518 (+12.9% over baseline 0.1345)**

---

## 4. Grid Size Sensitivity

### Classic4 (T_p=10, SCC-Dhillon, Centralized merge)

| Grid | Subproblems | NMI | ARI | Time (s) | Status |
|------|-------------|-----|-----|----------|--------|
| Standalone | 1 | 0.7905 | 0.7491 | 0.31 | OK |
| **2x2** | 4 | **0.7431** | **0.6687** | 14.2 | OK |
| **3x3** | 9 | **0.7273** | **0.6244** | 20.7 | OK |
| 4x4 | 16 | 0.0006 | 0.0001 | 26.6 | COLLAPSED |
| 5x5 | 25 | 0.0005 | 0.0007 | 33.5 | COLLAPSED |
| 6x6 | 36 | 0.0010 | 0.0043 | 39.0 | COLLAPSED |
| 8x8 | 64 | 0.0422 | 0.0412 | 56.0 | COLLAPSED |

**Critical finding:** For Classic4 (k=4), DiMergeCo collapses when grid > 3x3.

**Root cause analysis:**
- Classic4 has 4 true clusters. With 4x4 grid = 16 sub-blocks, each sub-block has ~400 rows.
- With k=4 clusters applied to a 400-row sub-block, the spectral signal is too diluted.
- The merge step cannot recover the global cluster structure from noisy sub-results.
- **Rule of thumb:** grid size M should satisfy `n_rows / M >> k` (i.e., each sub-block needs significantly more samples than clusters).

---

## 5. Merge Strategy Ablation

### 5.1 Classic4 (2x2, T_p=10)

| Strategy | NMI | ARI | Merge Time (s) | Total (s) |
|----------|-----|-----|-----------------|-----------|
| **Union** | **0.7390 +/- 0.0055** | **0.6459 +/- 0.0233** | 0.28 | 13.33 |
| **Centralized** | **0.7394 +/- 0.0049** | **0.6470 +/- 0.0231** | 0.43 | 13.64 |
| Random-pair | 0.3183 +/- 0.0841 | 0.2332 +/- 0.0926 | 0.006 | 13.20 |
| Hierarchical | 0.1992 +/- 0.0879 | 0.1377 +/- 0.0622 | 2.31 | 14.84 |
| Greedy-overlap | 0.0376 +/- 0.1189 | 0.0225 +/- 0.0713 | 17.81 | 28.31 |

### 5.2 BCW (2x2, T_p=10)

| Strategy | NMI | ARI | Merge Time (s) | Total (s) |
|----------|-----|-----|-----------------|-----------|
| **Union** | **0.4619 +/- 0.0290** | **0.5804 +/- 0.0261** | 0.20 | 4.34 |
| **Centralized** | **0.4619 +/- 0.0290** | **0.5804 +/- 0.0261** | 0.24 | 4.24 |
| Random-pair | 0.2415 +/- 0.1566 | 0.2662 +/- 0.1884 | 0.003 | 4.02 |
| Hierarchical | 0.1418 +/- 0.0625 | 0.1285 +/- 0.0938 | 3.86 | 10.49 |
| Greedy-overlap | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.34 | 4.37 |

**Key findings on merge strategies:**

1. **Union and Centralized are equivalent** in both quality and variance. They are the only viable strategies.
2. **Hierarchical merge is broken.** Despite being the theoretically motivated approach (progressive pairwise merging), it produces NMI ~0.20 on Classic4 vs ~0.74 for Union/Centralized. The issue likely lies in information loss during successive merge rounds.
3. **Random-pair** is fast but noisy (high variance: 0.08-0.16 std).
4. **Greedy-overlap** is both slow (17s merge time on Classic4) and nearly always produces trivial all-in-one clustering (NMI=0).

---

## 6. Large-Scale MovieLens Experiments

### MovieLens 25M (162,541 x 59,047, k=20)

Config: T_p=15, 8x8 grid (64 partitions), 72 Rayon threads, OPENBLAS_NUM_THREADS=72

| Method | Standalone | DiMergeCo | Speedup |
|--------|-----------|-----------|---------|
| Spectral | 650.6s | 780.6s | 0.83x |
| **NBVD** | 7,287.6s | **6,810.4s** | **1.07x** |
| ONM3F | 6,682.4s | 6,995.5s | 0.96x |

**NBVD is the only method that benefits from DiMergeCo on MovieLens**, achieving a modest 1.07x speedup even under unfavorable conditions (Rayon/OpenBLAS nesting conflict causing single-threaded BLAS).

Historical attempt (single-threaded BLAS):
- NBVD Standalone on full matrix: **~9 days** (single-threaded on 76 GB dense matrix)
- Conclusion: Full-matrix NMF on MovieLens-scale is impractical without parallelization.

---

## 7. Performance Optimizations

Optimizations implemented during PR #9 development:

### 7.1 Memory: O(n^2) Intermediate Elimination

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| NBVD denominator | F\*S\*G^T creates m x n | F\*S\*(G^T\*G)\*S^T | **8 GB -> 100 MB** (RCV1) |
| ONM3F updates | F\*F^T\*X creates m x m | F\*(F^T\*X) | **Eliminates n^2 matrix** |
| PNMTF | Same as NBVD | Same fix | |

### 7.2 NNLS Solver Optimization (FNMF)

| Version | Method | Time on RCV1-train |
|---------|--------|--------------------|
| Before | Manual Gaussian elimination | 350s (standalone) |
| After | LAPACK LU + SVD fallback | **107s (3.3x faster)** |

### 7.3 NMF Iteration Bug Fix

- **Bug:** `tri_factor_base` had a double nested loop doing `max_iter^2` updates (20x20=400 instead of 20).
- **Fix:** Restored correct double-loop structure (outer: convergence check, inner: max_iter updates).
- **Note:** Initially "fixed" by removing inner loop, but this broke algorithm correctness. The double loop matches the Python NMTFcoclust reference.

### 7.4 Parallel Merge Operations

- `merge_union`: Parallel signature hash computation using Rayon.
- `merge_intersection`: Parallel cluster pair comparisons with `flat_map`.
- `cluster_signature_hash`: Lightweight u64 hash instead of Vec clone.
- `parallel_level`: Increased default from 4 to 10.

### 7.5 OpenBLAS Integration

Added system OpenBLAS linking for `ndarray-linalg`, accelerating all matrix multiplications in NMF update rules and SVD.

---

## 8. Technical Findings

### 8.1 Rayon/OpenBLAS Nesting Conflict

**Root cause:** When OpenBLAS runs inside a Rayon worker thread, it detects the nested threading context and disables its own parallelism to avoid deadlocks.

- `tri_factor_base.rs::build_submatrices_from_labels()` uses Rayon `.into_par_iter()`
- This initializes Rayon's global thread pool even in Standalone mode
- All subsequent `.dot()` calls (BLAS operations) run single-threaded
- **Impact:** Both Standalone and DiMergeCo get single-threaded BLAS, making speedup comparison fair but absolute times slower than expected.

**Potential fix:** Set `OPENBLAS_NUM_THREADS=1` and rely solely on Rayon for parallelism. Or use `std::thread::spawn` instead of Rayon for the outer partition loop to avoid the nesting issue.

### 8.2 DiMergeCo Overhead Model

With T_p repetitions and M x N grid:

```
Total subproblems = T_p * M * N
Wall time (ideal) = ceil(T_p * M * N / n_threads) * time_per_sub + merge_time
```

For MovieLens with T_p=15, 8x8 grid, 72 threads:
- 960 subproblems, each ~530s on single-threaded BLAS
- Wall time ~= ceil(960/72) * 530 = 7,067s
- Standalone: ~7,000s
- **Result: DiMergeCo roughly breaks even** when BLAS is single-threaded.

### 8.3 SpectralCC (sklearn) vs SCC-Dhillon Implementation Gap

On Classic4:
- Our Rust SCC-Dhillon: NMI=0.790
- sklearn SpectralCoclustering: NMI=0.431

The gap comes from SVD dimensionality: sklearn uses `ceil(log2(k))` singular vectors while our implementation uses `k` singular vectors, retaining significantly more spectral information.

### 8.4 FNMF Bimodal Convergence

FNMF on Classic4 converges to two distinct local optima:
- **Mode A** (7/10 seeds): NMI=0.578, ARI=0.455
- **Mode B** (3/10 seeds): NMI=0.681, ARI=0.680

This is a known property of NMF methods with non-convex objectives. The initialization strategy (random vs. NNDSVD) significantly affects which mode is reached.

### 8.5 DiMergeCo Collapse Conditions

DiMergeCo produces trivial (all-in-one or random) clustering when:
1. **Grid too fine:** Each sub-block has too few rows relative to k (observed: 4x4 on Classic4 with n=6460, k=4).
2. **Sparse matrix + spectral atom:** Sub-blocks of sparse matrices may have near-zero entries, breaking SVD-based methods.
3. **Wrong merge strategy:** Hierarchical and Greedy-overlap consistently fail.
4. **Large sparse data + SCC atom:** RCV1-train with 2x2 grid still collapses (NMI~0.0001), suggesting SCC is fundamentally incompatible with the block-partition approach on sparse data.

---

## 9. Open Problems and Next Steps

### 9.1 Unresolved Issues

1. **Hierarchical merge is broken.** It should theoretically be the best strategy, but produces much worse results than the simple Union/Centralized approach. Needs investigation of information loss during successive pairwise merges.

2. **Sparse matrix handling.** DiMergeCo + SCC collapses on RCV1 even with 2x2 grid. The framework needs sparse-aware block partitioning or a different atom method for sparse data.

3. **NBVD is the only promising atom for large-scale.** But it's still very slow (~7000s on MovieLens). DiMergeCo's 1.07x speedup is insufficient for a compelling paper contribution.

### 9.2 Proposed Experiments

#### Experiment 1: Multi-Scale Speed Comparison
- Matrix sizes: 10K x 5K, 20K x 10K, 50K x 20K
- OPENBLAS_NUM_THREADS=1, RAYON_NUM_THREADS=72
- All 5 NMF methods + Spectral
- Goal: Show speedup grows with matrix size

#### Experiment 2: Thread Scaling (50K x 20K, NBVD)
- Thread counts: 1, 4, 8, 16, 36, 72
- Goal: Demonstrate near-linear scaling

#### Experiment 3: Quality Preservation
- Same configs as Experiment 1
- Measure coverage, cluster count, NMI, ARI
- Goal: Quantify quality-speed tradeoff

#### Experiment 4: Config Sensitivity (50K x 20K, NBVD)
- Vary T_p: 1, 3, 5, 10
- Vary grid: 2x2, 4x4, 8x8
- Goal: Find optimal parameter regimes

### 9.3 Summary Table

| Finding | Status | Impact |
|---------|--------|--------|
| Union/Centralized merge best | Confirmed | High - simplifies implementation |
| Hierarchical merge broken | Confirmed | High - contradicts theory |
| Grid > 3x3 collapses on Classic4 | Confirmed | Medium - limits parallelism |
| NBVD shows speedup on MovieLens | Confirmed (1.07x) | Low - marginal improvement |
| Rayon/OpenBLAS conflict | Confirmed | High - limits all experiments |
| DiMergeCo + SCC fails on sparse | Confirmed | High - limits applicability |
| O(n^2) memory optimizations | Implemented | High - 80x memory reduction |
| NNLS solver LAPACK optimization | Implemented | Medium - 3.3x FNMF speedup |
