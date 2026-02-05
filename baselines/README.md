# Baseline Methods for DiMergeCo Paper

## Method Coverage

| Paper Method | Citation | Implementation | Status |
|---|---|---|---|
| SCC | Dhillon 2001 | `sklearn.cluster.SpectralCoclustering` (same algorithm) | pip install |
| SpectralCC | sklearn | `sklearn.cluster.SpectralCoclustering` | pip install |
| NMTF | Long 2005 | `NMTFcoclust` (NBVD) | vendored |
| ONMTF | Ding 2006 | `NMTFcoclust` (ONM3F) | vendored |
| FNMTF | Kim 2011 | `nonnegfac` (fast NMF, 2-factor) + argmax | vendored |
| PNMTF | Chen 2023 | **No code released** - use NMTFcoclust PNMTF (Wang 2017) as proxy | vendored |
| WC-NMTF | Salah 2018 | **No code released** | missing |
| DiMergeCo-SCC | Ours | Rust impl in `src/` | this repo |
| DiMergeCo-PNMTF | Ours | Rust impl in `src/atom/` | this repo |

## Vendored Packages

- `NMTFcoclust/` - NBVD, ONMTF, ONM3F, PNMTF (7 NMTF variants)
- `nonnegfac/` - Kim & Park fast NMF (ANLS-BPP)

## Setup

```bash
pip install numpy scipy scikit-learn
```

## Dataset

Classic4: `data/classic4_benchmark_small.npy`
Labels: `data/classic4_benchmark_small_labels.npy`

## Run All Baselines

```bash
python baselines/run_classic4_baselines.py
```

## Notes

- PNMTF (Chen 2023) is a **Parallel** (MPI-distributed) NMTF. No code released.
  NMTFcoclust's PNMTF is a **Penalized** NMTF (Wang 2017) - different algorithm.
  We use NMTFcoclust's PNMTF as a reasonable NMTF baseline.
- FNMTF (Kim 2011) is actually fast **NMF** (2-factor A~WH), not tri-factorization.
  For co-clustering, row labels = argmax(W), col labels = argmax(H^T).
- WC-NMTF (Salah 2018) has no public code. May need to re-implement or drop.
