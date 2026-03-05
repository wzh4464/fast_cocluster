#!/bin/bash
# Master script: run all experiments needed for the paper.
# Usage: bash scripts/run_all_paper_experiments.sh
#
# This script runs Python-based experiments. For Rust experiments,
# see the cargo commands at the bottom (commented out, run on server).

set -e
cd "$(dirname "$0")/.."

PYTHON="uv run python"
echo "============================================================"
echo "DiMergeCo Paper Experiments — $(date)"
echo "Working directory: $(pwd)"
echo "============================================================"

# ── Step 0: Prepare BCW dataset ──────────────────────────────────────────────
echo ""
echo ">>> Step 0: Prepare BCW dataset"
$PYTHON scripts/prepare_bcw_data.py

# ── Step 1: Standalone baselines on Classic4 (10 seeds) ──────────────────────
echo ""
echo ">>> Step 1: Classic4 standalone baselines (10 seeds)"
$PYTHON baselines/run_classic4_baselines.py \
    --methods SCC-Dhillon,SpectralCC,NBVD,ONM3F,ONMTF,PNMTF,FNMF \
    --seeds 0,1,2,3,4,5,6,7,8,9

# ── Step 2: Standalone baselines on BCW (10 seeds) ───────────────────────────
echo ""
echo ">>> Step 2: BCW standalone baselines (10 seeds)"
$PYTHON baselines/run_bcw_baselines.py \
    --methods SCC-Dhillon,SpectralCC,NBVD,ONM3F,ONMTF,PNMTF,FNMF \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --n-clusters 2

# ── Step 3: DiMergeCo variants on Classic4 (10 seeds) ────────────────────────
echo ""
echo ">>> Step 3: DiMergeCo-SCC on Classic4 (10 seeds, 2x2, tp=30)"
$PYTHON baselines/run_dimerge_co_variants.py \
    --dataset classic4 \
    --methods SCC-Dhillon \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --m-blocks 2 --n-blocks 2 --t-p 30 \
    --n-clusters 4

# ── Step 4: DiMergeCo variants on BCW (10 seeds) ─────────────────────────────
echo ""
echo ">>> Step 4: DiMergeCo-SCC on BCW (10 seeds, 2x2, tp=10)"
$PYTHON baselines/run_dimerge_co_variants.py \
    --dataset bcw \
    --methods SCC-Dhillon \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --m-blocks 2 --n-blocks 2 --t-p 10 \
    --n-clusters 2

# ── Step 5: RCV1-train baselines (10 seeds, FNMF only — others may be slow) ─
echo ""
echo ">>> Step 5: RCV1-train standalone baselines (10 seeds)"
$PYTHON baselines/run_rcv1_baselines.py \
    --subset train \
    --methods FNMF \
    --seeds 0,1,2,3,4,5,6,7,8,9

# ── Step 6: DiMergeCo-SCC on RCV1-train (10 seeds) ──────────────────────────
echo ""
echo ">>> Step 6: DiMergeCo-SCC on RCV1-train (10 seeds, 2x2, tp=30)"
$PYTHON baselines/run_dimerge_co_variants.py \
    --dataset rcv1 \
    --rcv1-subset train \
    --methods SCC-Dhillon \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --m-blocks 2 --n-blocks 2 --t-p 30 \
    --n-clusters 4

# ── Step 7: Scalability comparison (Classic4) ────────────────────────────────
echo ""
echo ">>> Step 7: Scalability comparison on Classic4"
$PYTHON scripts/run_scalability.py \
    --dataset classic4 \
    --seeds 0,1,2 \
    --t-p 10

# ── Step 8: Memory profiling ─────────────────────────────────────────────────
echo ""
echo ">>> Step 8: Memory profiling"
$PYTHON scripts/run_memory_profiling.py \
    --datasets classic4,bcw

# ── Step 9: Merge strategy ablation (Classic4) ───────────────────────────────
echo ""
echo ">>> Step 9: Merge strategy ablation on Classic4"
$PYTHON scripts/run_merge_ablation.py \
    --dataset classic4 \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --m-blocks 2 --n-blocks 2 --t-p 10

# ── Step 10: Merge strategy ablation (BCW) ───────────────────────────────────
echo ""
echo ">>> Step 10: Merge strategy ablation on BCW"
$PYTHON scripts/run_merge_ablation.py \
    --dataset bcw \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --m-blocks 2 --n-blocks 2 --t-p 10 \
    --n-clusters 2

echo ""
echo "============================================================"
echo "ALL PYTHON EXPERIMENTS COMPLETED — $(date)"
echo "Results saved in: baselines/results/"
echo "============================================================"
ls -la baselines/results/*.json

# ── Rust experiments (run on server with OpenBLAS) ───────────────────────────
# Uncomment and run on the server:
#
# # DiMergeCo parameter sweep on Classic4
# RUST_LOG=info cargo run --release --example evaluate_classic4
#
# # DiMergeCo with all atom methods on Classic4 and RCV1
# RUST_LOG=info cargo run --release --example evaluate_dimerge_atom -- all
#
# # Benchmark parallelism advantage
# OPENBLAS_NUM_THREADS=1 cargo run --release --example benchmark_dimerge_advantage
