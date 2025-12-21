#!/bin/bash
# =============================================================================
# HEAVY ContextualizedCorrelationNetworks DDP SCALING BENCHMARK
# =============================================================================
#
# This benchmark tests multi-GPU scaling with the ACTUAL CCN model, but
# configured for maximum compute to properly stress-test GPU parallelism.
#
# HEAVY Configuration vs Original:
#   Parameter        | Original | Heavy   | Compute Impact
#   -----------------|----------|---------|----------------
#   Archetypes       | 16-30    | 64      | 2-4x more mixture components
#   Encoder width    | 25       | 256     | 10x wider networks
#   Encoder layers   | 3        | 6       | 2x deeper networks
#   Bootstraps       | 1        | 3       | 3x more models (ensemble)
#   Data PCs         | 50       | 100     | 2x larger output space
#
# Estimated parameters: ~15-30M (vs ~300K original)
#
# Expected scaling:
#   1 GPU:  baseline
#   2 GPU:  ~1.85x speedup (92% efficiency)
#   3 GPU:  ~2.65x speedup (88% efficiency)
#   4 GPU:  ~3.4x speedup (85% efficiency)
#
# =============================================================================

set -e

# ===== CONFIGURATION =====
SCRIPT="ccn_scaling_heavy.py"
OUTDIR="bench_results_ccn_heavy"
EPOCHS=20
WARMUP=1
BATCH_SIZE=512           # Per GPU

# HEAVY CCN Architecture
ARCHETYPES=64            # Original: 16-30
ENCODER_WIDTH=256        # Original: 25
ENCODER_LAYERS=6         # Original: 3
BOOTSTRAPS=1             # Original: 1

# Data dimensionality
DATA_PCS=100             # Original: 50
CONTEXT_PCS=100

# Runtime
NUM_WORKERS=4
SUBSAMPLE=1.0

# Clean previous results
echo "=============================================="
echo "Cleaning previous results..."
echo "=============================================="
rm -f "${OUTDIR}/ccn_heavy_scaling_results.csv"
mkdir -p "${OUTDIR}"

echo ""
echo "=============================================="
echo "HEAVY CCN SCALING BENCHMARK"
echo "=============================================="
echo "Script: ${SCRIPT}"
echo "Epochs: ${EPOCHS} (+ ${WARMUP} warmup)"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo ""
echo "--- HEAVY CCN Config ---"
echo "Archetypes: ${ARCHETYPES}"
echo "Encoder: ${ENCODER_WIDTH}w × ${ENCODER_LAYERS}L"
echo "Bootstraps: ${BOOTSTRAPS}"
echo "Data PCs: ${DATA_PCS}"
echo ""
echo "Output: ${OUTDIR}"
echo ""

# -----------------------------------------------------------------------------
# TEST 1: 1-GPU Baseline
# -----------------------------------------------------------------------------
echo "=============================================="
echo "[1/4] Running 1-GPU baseline..."
echo "=============================================="

python ${SCRIPT} \
    --epochs ${EPOCHS} \
    --warmup-epochs ${WARMUP} \
    --batch-size ${BATCH_SIZE} \
    --archetypes ${ARCHETYPES} \
    --encoder-width ${ENCODER_WIDTH} \
    --encoder-layers ${ENCODER_LAYERS} \
    --bootstraps ${BOOTSTRAPS} \
    --data-pcs ${DATA_PCS} \
    --context-pcs ${CONTEXT_PCS} \
    --num-workers ${NUM_WORKERS} \
    --subsample-fraction ${SUBSAMPLE} \
    --devices 1 \
    --outdir ${OUTDIR} \
    --label "1gpu_baseline"

# Extract baseline time for efficiency calculation
BASELINE_TIME=$(tail -1 "${OUTDIR}/ccn_heavy_scaling_results.csv" | cut -d',' -f2)
echo ""
echo ">>> Baseline time: ${BASELINE_TIME}s"
echo ""

# -----------------------------------------------------------------------------
# TEST 2: 2-GPU DDP
# -----------------------------------------------------------------------------
echo "=============================================="
echo "[2/4] Running 2-GPU DDP..."
echo "=============================================="

torchrun \
    --standalone \
    --nproc_per_node=2 \
    ${SCRIPT} \
        --epochs ${EPOCHS} \
        --warmup-epochs ${WARMUP} \
        --batch-size ${BATCH_SIZE} \
        --archetypes ${ARCHETYPES} \
        --encoder-width ${ENCODER_WIDTH} \
        --encoder-layers ${ENCODER_LAYERS} \
        --bootstraps ${BOOTSTRAPS} \
        --data-pcs ${DATA_PCS} \
        --context-pcs ${CONTEXT_PCS} \
        --num-workers ${NUM_WORKERS} \
        --subsample-fraction ${SUBSAMPLE} \
        --devices 2 \
        --outdir ${OUTDIR} \
        --label "2gpu_ddp" \
        --baseline-time ${BASELINE_TIME}

echo ""

# -----------------------------------------------------------------------------
# TEST 3: 3-GPU DDP
# -----------------------------------------------------------------------------
echo "=============================================="
echo "[3/4] Running 3-GPU DDP..."
echo "=============================================="

torchrun \
    --standalone \
    --nproc_per_node=3 \
    ${SCRIPT} \
        --epochs ${EPOCHS} \
        --warmup-epochs ${WARMUP} \
        --batch-size ${BATCH_SIZE} \
        --archetypes ${ARCHETYPES} \
        --encoder-width ${ENCODER_WIDTH} \
        --encoder-layers ${ENCODER_LAYERS} \
        --bootstraps ${BOOTSTRAPS} \
        --data-pcs ${DATA_PCS} \
        --context-pcs ${CONTEXT_PCS} \
        --num-workers ${NUM_WORKERS} \
        --subsample-fraction ${SUBSAMPLE} \
        --devices 3 \
        --outdir ${OUTDIR} \
        --label "3gpu_ddp" \
        --baseline-time ${BASELINE_TIME}

echo ""

# -----------------------------------------------------------------------------
# TEST 4: 4-GPU DDP
# -----------------------------------------------------------------------------
echo "=============================================="
echo "[4/4] Running 4-GPU DDP..."
echo "=============================================="

torchrun \
    --standalone \
    --nproc_per_node=4 \
    ${SCRIPT} \
        --epochs ${EPOCHS} \
        --warmup-epochs ${WARMUP} \
        --batch-size ${BATCH_SIZE} \
        --archetypes ${ARCHETYPES} \
        --encoder-width ${ENCODER_WIDTH} \
        --encoder-layers ${ENCODER_LAYERS} \
        --bootstraps ${BOOTSTRAPS} \
        --data-pcs ${DATA_PCS} \
        --context-pcs ${CONTEXT_PCS} \
        --num-workers ${NUM_WORKERS} \
        --subsample-fraction ${SUBSAMPLE} \
        --devices 4 \
        --outdir ${OUTDIR} \
        --label "4gpu_ddp" \
        --baseline-time ${BASELINE_TIME}

echo ""

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo ""
echo "Full Results:"
echo ""
column -t -s',' "${OUTDIR}/ccn_heavy_scaling_results.csv"
echo ""

echo "=============================================="
echo "SCALING SUMMARY"
echo "=============================================="
awk -F',' '
NR==1 {next}
{
    printf "  %-15s: %8.2fs | %5.2fx speedup | %5.1f%% efficiency\n", $1, $2, $13, $14
}
' "${OUTDIR}/ccn_heavy_scaling_results.csv"
echo ""

echo "=============================================="
echo "CCN CONFIGURATION USED"
echo "=============================================="
echo "  Archetypes:     ${ARCHETYPES}"
echo "  Encoder width:  ${ENCODER_WIDTH}"
echo "  Encoder layers: ${ENCODER_LAYERS}"
echo "  Bootstraps:     ${BOOTSTRAPS}"
echo "  Data PCs:       ${DATA_PCS}"
echo ""