#!/bin/bash
# Rebuttal evaluation for R2 loss/noisy-branch ablations.
# Fill the timestamped directories after running each embedding variant with locmark.main.

NUM_IMG=100
TAMPER_MODE=ldm
DATASET=/mnt/nas5/suhyeon/datasets/valAGE-Set
SAVE_BASE=/mnt/nas5/suhyeon/projects/apt_rebuttal
GPU_ID=${GPU_ID:-2}

target_dirs=(
    "ours_ablation_loss/ours_baseline/TIMESTAMP_TBD"
    "ours_ablation_loss/hnm_focal/TIMESTAMP_TBD"
    "ours_ablation_loss/noisy_vae_roundtrip/TIMESTAMP_TBD"
    "ours_ablation_loss/noisy_latent_full/TIMESTAMP_TBD"
)

for dir in "${target_dirs[@]}"; do
    echo "===== $dir ====="
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval_AGE.py \
        target_model=ours \
        src_image_path=$DATASET \
        save_path=$SAVE_BASE/$dir \
        eval_size=256 \
        start_idx=0 \
        end_idx=$NUM_IMG \
        tamper_mode=$TAMPER_MODE \
        use_refiner=True \
        anchor_type=rademacher
done
