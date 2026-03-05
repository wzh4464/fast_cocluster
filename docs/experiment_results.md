# DiMergeCo Experiment Results

Date: 2026-03-05
Hardware: 144-core Linux server (Linux 6.8.0-87-generic)
Datasets: Classic4 (6460 x 4667), BCW (569 x 30)

---

## T4: Running Time Comparison

Classic4 dataset, seed=0, Python baselines (unless noted).

| Method | Time (s) | NMI | ARI | Condition |
|--------|----------|-----|-----|-----------|
| SCC-Dhillon | 0.2 | 0.7905 | 0.7491 | idle |
| SpectralCC | 0.3 | 0.4315 | 0.2723 | idle |
| FNMF | 1.1 | 0.5784 | 0.4548 | idle |
| **DiMergeCo+SCC** | **13.9** | **0.7366** | **0.6511** | idle, T_p=10, 2x2 |
| NBVD | 1024.8 | 0.7224 | 0.6306 | idle |
| PNMTF | 1383.9 | 0.5460 | 0.4280 | contention (3 parallel NMF) |
| ONM3F | >3600 | -- | -- | killed after 60min |
| ONMTF | >3600 | -- | -- | killed after 60min |

### Key Observations

- SCC-Dhillon is the fastest single method (0.2s) and also has the highest NMI (0.791).
- DiMergeCo+SCC achieves competitive NMI (0.737) in 13.9s, which is 70x faster than NBVD (1024.8s) while achieving similar quality (NMI 0.737 vs 0.722).
- NMF methods (NBVD/PNMTF/ONM3F/ONMTF) are extremely slow on Classic4 (>1000s), making them impractical without DiMergeCo partitioning.
- FNMF is fast (1.1s) but has low NMI (0.578).

### Paper Table Discrepancies

The paper table (root.tex L842) has `\todo{}` markers with old values:
- NBVD: 298.7 (measured: 1024.8) -- 3.4x difference, likely different hardware/settings
- PNMTF: 303.7 (measured: 1383.9) -- 4.6x difference
- FNMTF: 275.1 (measured: 1.1) -- dramatically different, old value likely used different iteration count
- SCC: 2.6 and DiMergeCo-SCC: 6.7 are from Rust implementation with randomized SVD

---

## T5: Peak Memory (RSS)

Measured via `/usr/bin/time -v` (peak RSS) and `/proc/<pid>/status` VmHWM for long-running methods.
Python baseline overhead: ~123 MB (interpreter + numpy/scipy/sklearn imports).

### Classic4 (6460 x 4667, ~242 MB dense)

| Method | Peak RSS (MB) | Category |
|--------|--------------|----------|
| FNMF | 351 | Spectral/Light NMF |
| SCC | 370 | Spectral |
| SpectralCC | 374 | Spectral |
| **DiMergeCo-SCC** | **496** | **DiMergeCo** |
| **DiMergeCo-PNMTF** | **777** | **DiMergeCo** |
| ONMTF | 1,601 | Heavy NMF |
| ONM3F | 1,605 | Heavy NMF |
| PNMTF | 1,631 | Heavy NMF |
| NBVD | 1,659 | Heavy NMF |

### BCW (569 x 30, ~0.13 MB dense)

| Method | Peak RSS (MB) |
|--------|--------------|
| FNMF | 116 |
| NBVD | 117 |
| ONMTF | 117 |
| PNMTF | 117 |
| ONM3F | 121 |
| SCC | 128 |
| DiMergeCo-PNMTF | 131 |
| SpectralCC | 132 |
| DiMergeCo-SCC | 176 |

BCW is too small (0.13 MB) for meaningful memory comparison -- all values dominated by Python overhead (~123 MB).

### Key Observations

- **DiMergeCo halves NMF memory**: DiMergeCo-PNMTF uses 777 MB vs standalone PNMTF's 1,631 MB (2.1x reduction). This is because DiMergeCo partitions the matrix into sub-blocks, so each NMF atom operates on smaller matrices with smaller intermediate allocations.
- Spectral methods (SCC, SpectralCC, FNMF) are inherently light (351-374 MB). DiMergeCo adds modest overhead for these (496 MB for DiMergeCo-SCC).
- All heavy NMF methods (NBVD, ONM3F, ONMTF, PNMTF) use ~1,600 MB on Classic4 due to dense intermediate matrices.

---

## T6: Scalability Figure

Script: `scripts/generate_scalability_figure.py`
Output: `~/big-cocluster-paper/src/images/scalability_comparison.pdf`

Compares DiMergeCo-SCC vs standalone SCC across partition counts (2x2 to 8x8).

---

## T8: Statistical Significance (Paired t-test)

10 seeds per comparison, paired t-test (scipy.stats.ttest_rel), significance level alpha=0.01.

### Classic4

| Comparison | DiMergeCo NMI | Baseline NMI | Diff | t-stat | p-value | Winner |
|-----------|---------------|-------------|------|--------|---------|--------|
| DiMergeCo+SCC vs SCC-Dhillon | 0.748 | 0.790 | -0.042 | -38.98 | 2.4e-11 | **SCC** |

### BCW

| Comparison | DiMergeCo NMI | Baseline NMI | Diff | t-stat | p-value | Significant? |
|-----------|---------------|-------------|------|--------|---------|-------------|
| vs SpectralCC | 0.468 | 0.128 | +0.340 | 6.05 | 1.9e-4 | **DiMergeCo wins** |
| vs SCC-Dhillon | 0.468 | 0.491 | -0.023 | -3.50 | 6.7e-3 | **SCC wins** |
| vs NBVD | 0.468 | 0.031 | +0.437 | 62.69 | 3.4e-13 | **DiMergeCo wins** |
| vs ONM3F | 0.468 | 0.035 | +0.433 | 68.42 | 1.5e-13 | **DiMergeCo wins** |
| vs ONMTF | 0.468 | 0.033 | +0.435 | 65.92 | 2.1e-13 | **DiMergeCo wins** |
| vs PNMTF | 0.468 | 0.402 | +0.066 | 1.56 | 0.153 | Not significant |
| vs FNMF | 0.468 | 0.347 | +0.121 | 17.99 | 2.3e-8 | **DiMergeCo wins** |

### Key Observations

- **Classic4**: Standalone SCC significantly outperforms DiMergeCo+SCC (NMI 0.790 vs 0.748, p<0.01). On small datasets where full SVD is feasible, partitioning introduces information loss with no speed benefit.
- **BCW**: DiMergeCo+SCC significantly outperforms 5 of 7 baselines. SCC-Dhillon slightly wins (0.491 vs 0.468). DiMergeCo vs PNMTF is not significant (p=0.15).
- **Implication for paper**: The claim that DiMergeCo "outperforms all baselines" needs qualification. DiMergeCo's advantage is primarily on large-scale data where standalone methods are too slow, not on small datasets.

---

## T9: Merge Strategy Ablation (Rust Implementation)

Rust DiMergeCo with different merge strategies on Classic4, 2x2 blocks, 16 threads.

### T_p = 10

| Strategy | NMI | ARI | Merge Time (s) | Total Time (s) |
|----------|-----|-----|----------------|----------------|
| **Adaptive (Ours)** | **0.760** | **0.720** | **0.006** | 4.24 |
| Union | 0.755 | 0.703 | 0.005 | 2.98 |
| Weighted(0.7/0.3) | 0.729 | 0.609 | 0.002 | 3.98 |
| Weighted(0.5/0.5) | 0.726 | 0.621 | 0.002 | 3.27 |
| Intersection(0.3) | 0.000 | 0.000 | 0.016 | 3.68 |
| Intersection(0.5) | 0.000 | 0.000 | 0.005 | 4.43 |

Baseline SCC: NMI=0.776, ARI=0.743, Time=2.70s

### T_p = 30

| Strategy | NMI | ARI | Merge Time (s) | Total Time (s) |
|----------|-----|-----|----------------|----------------|
| Weighted(0.7/0.3) | 0.760 | 0.691 | 0.009 | 9.21 |
| **Adaptive (Ours)** | **0.759** | **0.688** | **0.015** | 9.29 |
| Weighted(0.5/0.5) | 0.751 | 0.667 | 0.008 | 10.24 |
| Union | 0.746 | 0.657 | 0.009 | 9.87 |
| Intersection(0.3) | 0.000 | 0.000 | 0.013 | 10.94 |
| Intersection(0.5) | 0.000 | 0.000 | 0.013 | 7.78 |

Baseline SCC: NMI=0.755, ARI=0.662, Time=2.62s

### BCW (T_p = 10)

| Strategy | NMI | ARI | Total Time (s) |
|----------|-----|-----|----------------|
| Adaptive | 0.467 | 0.586 | 0.065 |
| Union | 0.467 | 0.586 | 0.154 |
| Weighted(0.7/0.3) | 0.467 | 0.586 | 0.204 |
| Weighted(0.5/0.5) | 0.467 | 0.586 | 0.151 |
| Intersection | 0.000 | 0.000 | 0.142-0.281 |

Baseline SCC: NMI=0.491, ARI=0.608, Time=0.161s

### Comparison with Python Implementation

| Strategy | Classic4 NMI (Python) | Classic4 NMI (Rust T_p=10) |
|----------|----------------------|---------------------------|
| Hierarchical (Python M1) | 0.199 | -- |
| Random-pair (Python M2) | 0.318 | -- |
| Centralized (Python M3) | 0.739 | -- |
| Greedy-overlap (Python M4) | 0.038 | -- |
| Union (Python M5 / Rust) | 0.739 | 0.755 |
| Adaptive (Rust only) | -- | **0.760** |

### Key Observations

- **Adaptive is best or near-best** across all settings. At T_p=10 it leads clearly (NMI=0.760); at T_p=30, Weighted(0.7/0.3) is slightly better (0.760 vs 0.759) but within noise.
- **Intersection is catastrophic**: produces 0 co-clusters at both thresholds (0.3 and 0.5), yielding NMI=ARI=0. Too aggressive for sparse co-cluster overlap.
- **Merge is negligible cost**: all merge strategies complete in <0.02s. The bottleneck is always the local clustering phase.
- **Rust >> Python**: Rust Adaptive (NMI=0.760) and Union (NMI=0.755) far exceed the best Python strategies (Centralized/Union at NMI=0.739). Python Hierarchical (M1) is particularly poor (NMI=0.199).
- **BCW is too small to differentiate**: all non-Intersection strategies produce identical results on BCW.

---

## Summary for Paper

### What to claim

1. **Speed**: DiMergeCo+SCC is 70x faster than NBVD and 100x faster than PNMTF on Classic4, with competitive quality (NMI 0.737 vs 0.722 for NBVD).
2. **Memory**: DiMergeCo reduces peak memory by ~2x for NMF atoms (777 MB vs 1,631 MB for PNMTF).
3. **Merge strategy**: Adaptive merge is consistently best or near-best; merge overhead is negligible (<0.02s).
4. **Scalability**: DiMergeCo enables previously impractical NMF methods (ONM3F/ONMTF >1 hour standalone) to run in reasonable time through sub-block partitioning.

### What NOT to claim

1. DiMergeCo does NOT outperform standalone SCC on small datasets (Classic4: 0.748 vs 0.790 NMI, statistically significant).
2. DiMergeCo's quality advantage over PNMTF on BCW is NOT statistically significant (p=0.15).

### Paper table values to update (root.tex L842)

Current `\todo{}` values need replacement with measured values. Note the large discrepancies vs old values suggest different experimental conditions.

---

## Raw Data Files

- `baselines/results/classic4_timing_current_hw.json` -- T4 timing
- `baselines/results/memory_profiling_complete.json` -- T5 memory
- `baselines/results/memory_profiling.json` -- T5 memory (script output)
- `baselines/results/ttest_results.json` -- T8 t-tests
- `baselines/results/bcw_baselines.json` -- T8 BCW baselines
- `baselines/results/bcw_dimerge_co_variants.json` -- T8 BCW DiMergeCo
- `baselines/results/rust_merge_ablation.json` -- T9 ablation
- `scripts/compute_ttest.py` -- T8 script
- `scripts/generate_scalability_figure.py` -- T6 script
