#!/bin/bash
set -euo pipefail

NUM_IMG=${NUM_IMG:-100}
GPU_ID=${GPU_ID:-0}
DATASET=${DATASET:-/mnt/nas5/suhyeon/datasets/valAGE-Set}
SAVE_BASE=${SAVE_BASE:-/mnt/nas5/suhyeon/projects/apt_rebuttal}
APT_COVER_DIR=${APT_COVER_DIR:-$SAVE_BASE/ours/cover_images}
OUTPUT=${OUTPUT:-$SAVE_BASE/timing_results.json}

CUDA_VISIBLE_DEVICES=$GPU_ID python benchmark_timing.py \
    --dataset "$DATASET" \
    --num-images "$NUM_IMG" \
    --gpu 0 \
    --output "$OUTPUT" \
    --apt-cover-dir "$APT_COVER_DIR" \
    --models wam,omniguard,stableguard,apt,apt_refiner \
    --anchor-type submitted
