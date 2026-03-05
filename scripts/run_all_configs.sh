#!/bin/bash
# Run all Classic4 experiment configurations in parallel.
# Each config gets 4 threads, we run ~36 configs concurrently on 144 cores.

set -e
cd "$(dirname "$0")/.."

BIN="target/release/examples/run_single_config"
DATASET="paper"  # 6460x4667
SEEDS=${1:-3}    # default 3 seeds (pass 10 for full)
THREADS=2        # threads per config (lower to allow more parallelism)
RESULTS_DIR="data/experiment_results"
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

export OPENBLAS_NUM_THREADS=2

echo "============================================================"
echo "Classic4 Full Experiment Suite"
echo "Dataset: $DATASET, Seeds: $SEEDS, Threads/config: $THREADS"
echo "Available cores: $(nproc)"
echo "============================================================"

# Generate all config commands
CMDS_FILE=$(mktemp)

# Baseline (use 8 threads since SVD benefits from more)
echo "$BIN $DATASET 0 0 0 8 $SEEDS > $RESULTS_DIR/baseline.txt 2>&1" >> "$CMDS_FILE"

# DiMergeCo configs: block grids × T_p values
# Focus on promising block grids (learned from small-dataset sweep):
# - Aspect ratio near M/N ≈ 6460/4667 ≈ 1.38
# - Not too many blocks (min block dim must be >> 50)
BLOCKS="2,2 2,3 3,2 3,3 2,4 3,4 4,3 4,4 5,4 4,5 5,5 6,5 8,6 10,7"

# T_p values
TPS="5 10 20 30"

for BLOCK in $BLOCKS; do
    M=$(echo $BLOCK | cut -d, -f1)
    N=$(echo $BLOCK | cut -d, -f2)
    for TP in $TPS; do
        LABEL="${M}x${N}_tp${TP}"
        echo "$BIN $DATASET $M $N $TP $THREADS $SEEDS > $RESULTS_DIR/${LABEL}.txt 2>&1" >> "$CMDS_FILE"
    done
done

TOTAL=$(wc -l < "$CMDS_FILE")
# Max parallel: use (cores / threads_per_config), cap at total
MAX_PARALLEL=$(( $(nproc) / $THREADS ))
if [ "$MAX_PARALLEL" -gt "$TOTAL" ]; then
    MAX_PARALLEL=$TOTAL
fi

echo "Total configs: $TOTAL"
echo "Max parallel: $MAX_PARALLEL"
echo "Starting at: $(date)"
echo ""

# Run all in parallel
cat "$CMDS_FILE" | xargs -I {} -P "$MAX_PARALLEL" bash -c '{}'

rm -f "$CMDS_FILE"

echo ""
echo "============================================================"
echo "All experiments complete at: $(date)"
echo "============================================================"

# Aggregate results
echo ""
echo "Aggregating results..."
echo ""

printf "%-16s %4s %4s %4s %8s %8s %8s %8s %8s %8s\n" \
    "Config" "M" "N" "T_p" "NMI" "±" "ARI" "±" "Time" "±"
echo "------------------------------------------------------------------------------------"

for f in "$RESULTS_DIR"/*.txt; do
    if [ -s "$f" ]; then
        # Only print lines that look like results (start with a word, have numbers)
        grep -E '^[a-z0-9]' "$f" 2>/dev/null | while read line; do
            CONFIG=$(echo "$line" | awk '{print $1}')
            M=$(echo "$line" | awk '{print $2}')
            N=$(echo "$line" | awk '{print $3}')
            TP=$(echo "$line" | awk '{print $4}')
            NMI=$(echo "$line" | awk '{print $5}')
            NMI_STD=$(echo "$line" | awk '{print $6}')
            ARI=$(echo "$line" | awk '{print $7}')
            ARI_STD=$(echo "$line" | awk '{print $8}')
            TIME=$(echo "$line" | awk '{print $9}')
            TIME_STD=$(echo "$line" | awk '{print $10}')
            printf "%-16s %4s %4s %4s %8s %8s %8s %8s %8s %8s\n" \
                "$CONFIG" "$M" "$N" "$TP" "$NMI" "$NMI_STD" "$ARI" "$ARI_STD" "$TIME" "$TIME_STD"
        done
    fi
done | sort -t' ' -k5 -rn

echo ""
echo "Paper reported:"
echo "  SCC baseline:     NMI=0.922±0.003  ARI=0.771±0.005"
echo "  DiMergeCo-SCC:    NMI=0.865±0.007  ARI=0.776±0.006  Time=112.5s"
