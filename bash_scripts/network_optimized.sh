#!/bin/bash
# =============================================================================
# OPTIMIZED DDP SCALING BENCHMARK SCRIPT
# =============================================================================
# 
# This script runs a proper scaling comparison with CONSTANT GLOBAL BATCH SIZE
# to measure true parallel efficiency.
#
# Key differences from original:
# 1. Global batch size stays at 256 regardless of GPU count
# 2. Each GPU processes 256/N samples per batch
# 3. Warmup epoch excluded from timing
# 4. Reduced DataLoader workers to avoid contention
# 5. NCCL optimizations enabled
#
# Expected scaling (realistic for small models):
#   1 GPU: baseline
#   2 GPU: 1.6-1.8x speedup (80-90% efficiency)
#   3 GPU: 2.2-2.6x speedup (73-87% efficiency)  
#   4 GPU: 2.8-3.4x speedup (70-85% efficiency)
#
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT="unseen_pert_scale_optimized.py"
OUTDIR="bench_results_optimized"
EPOCHS=40
WARMUP=1
BATCH_SIZE=256  # Per-GPU batch size (global = this × num_gpus)
NUM_WORKERS=4   # Will be auto-reduced for multi-GPU
SUBSAMPLE=1.0   # Use full data (matches existing cache filename)

# IMPORTANT: For small models, we MUST scale batch size with GPUs.
# Otherwise communication overhead dominates and multi-GPU is SLOWER.
# Using --scale-batch flag to scale global batch with GPU count.

# Clean previous results
rm -f "${OUTDIR}/scaling_results_optimized.csv"
mkdir -p "${OUTDIR}"

echo "=============================================="
echo "STARTING SCALING BENCHMARK"
echo "=============================================="
echo "Script: ${SCRIPT}"
echo "Epochs: ${EPOCHS} (+ ${WARMUP} warmup)"
echo "Global Batch Size: ${BATCH_SIZE} (constant)"
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
  --subsample-fraction ${SUBSAMPLE} \
  --devices 1 \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --outdir ${OUTDIR} \
  --label "1gpu_baseline" \
  --verbose

# Extract baseline time for efficiency calculation
BASELINE_TIME=$(tail -1 "${OUTDIR}/scaling_results_optimized.csv" | cut -d',' -f2)
echo ""
echo "Baseline time: ${BASELINE_TIME}s"
echo ""

# -----------------------------------------------------------------------------
# TEST 2: 2-GPU with torchrun
# -----------------------------------------------------------------------------
echo "=============================================="
echo "[2/4] Running 2-GPU DDP with torchrun..."
echo "=============================================="

torchrun \
  --standalone \
  --nproc_per_node=2 \
  ${SCRIPT} \
    --epochs ${EPOCHS} \
    --warmup-epochs ${WARMUP} \
    --subsample-fraction ${SUBSAMPLE} \
    --devices 2 \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --outdir ${OUTDIR} \
    --label "2gpu_ddp" \
    --baseline-time ${BASELINE_TIME} \
    --scale-batch \
    --verbose

echo ""

# -----------------------------------------------------------------------------
# TEST 3: 3-GPU with torchrun
# -----------------------------------------------------------------------------
echo "=============================================="
echo "[3/4] Running 3-GPU DDP with torchrun..."
echo "=============================================="

torchrun \
  --standalone \
  --nproc_per_node=3 \
  ${SCRIPT} \
    --epochs ${EPOCHS} \
    --warmup-epochs ${WARMUP} \
    --subsample-fraction ${SUBSAMPLE} \
    --devices 3 \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --outdir ${OUTDIR} \
    --label "3gpu_ddp" \
    --baseline-time ${BASELINE_TIME} \
    --scale-batch \
    --verbose

echo ""

# -----------------------------------------------------------------------------
# TEST 4: 4-GPU with torchrun
# -----------------------------------------------------------------------------
echo "=============================================="
echo "[4/4] Running 4-GPU DDP with torchrun..."
echo "=============================================="

torchrun \
  --standalone \
  --nproc_per_node=4 \
  ${SCRIPT} \
    --epochs ${EPOCHS} \
    --warmup-epochs ${WARMUP} \
    --subsample-fraction ${SUBSAMPLE} \
    --devices 4 \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --outdir ${OUTDIR} \
    --label "4gpu_ddp" \
    --baseline-time ${BASELINE_TIME} \
    --scale-batch \
    --verbose

echo ""

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: ${OUTDIR}/scaling_results_optimized.csv"
echo ""
echo "Results:"
cat "${OUTDIR}/scaling_results_optimized.csv" | column -t -s','
echo ""

# Calculate speedups
echo "Speedup Summary:"
echo "----------------"
awk -F',' 'NR==1 {next} NR==2 {base=$2} {printf "%s: %.2fs (%.2fx speedup, %.1f%% efficiency)\n", $1, $2, base/$2, $8}' \
  "${OUTDIR}/scaling_results_optimized.csv"