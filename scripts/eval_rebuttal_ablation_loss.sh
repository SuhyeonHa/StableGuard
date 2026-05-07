#!/bin/bash
# Rebuttal evaluation for loss/noisy-branch ablations and R3 hyperparameter sensitivity.

NUM_IMG=100
TAMPER_MODE=ldm
DATASET=/mnt/nas5/suhyeon/datasets/valAGE-Set
SAVE_BASE=/mnt/nas5/suhyeon/projects/apt_rebuttal
GPU_ID=${GPU_ID:-2}

target_dirs=(
    # R2 loss / noisy-branch design ablations
    # "ours_ablation/focal/20260506-162523"
    "ours_ablation/focal-1/20260507-073349"
    # "ours_ablation/vae_roundtrip/20260506-163311"
    # "ours_ablation/latent_full/20260506-163756"

    # R3 hyperparameter sensitivity
    # "ours_ablation/hnm-weak/20260506-180155"
    # "ours_ablation/hnm-strong/20260506-180417"
    # "ours_ablation/noise-weak/20260506-180633"
    # "ours_ablation/noise-strong/20260506-180819"
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
        anchor_type=submitted
done
