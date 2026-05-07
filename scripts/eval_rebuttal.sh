#!/bin/bash
# Rebuttal evaluation: all target models vs. flux2 tamper mode

NUM_IMG=100
TAMPER_MODE=lora
DATASET=/mnt/nas5/suhyeon/datasets/valAGE-Set
SAVE_BASE=/mnt/nas5/suhyeon/projects/apt_rebuttal

# ours (LocMark, with refiner)
CUDA_VISIBLE_DEVICES=1 python eval_AGE.py \
    target_model=ours \
    src_image_path=$DATASET \
    save_path=$SAVE_BASE/ours \
    eval_size=256 \
    start_idx=0 \
    end_idx=$NUM_IMG \
    tamper_mode=$TAMPER_MODE \
    use_refiner=False \
    eval_dist=False \
    anchor_type=submitted

# # ours_e2e (LocMark end-to-end)
# CUDA_VISIBLE_DEVICES=0 python eval_AGE.py \
#     target_model=ours_e2e \
#     src_image_path=$DATASET \
#     save_path=$SAVE_BASE/ours_e2e \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODE

# # wam
# CUDA_VISIBLE_DEVICES=0 python eval_AGE.py \
#     target_model=wam \
#     src_image_path=$DATASET \
#     save_path=$SAVE_BASE/wam \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODE \
#     wm_strength=3.0

# # omniguard
# CUDA_VISIBLE_DEVICES=0 python eval_AGE.py \
#     target_model=omniguard \
#     src_image_path=$DATASET \
#     save_path=$SAVE_BASE/omniguard \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODE \
#     wm_strength=2.0

# # stableguard
# CUDA_VISIBLE_DEVICES=0 python eval_AGE.py \
#     target_model=stableguard \
#     src_image_path=$DATASET \
#     save_path=$SAVE_BASE/stableguard \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODE
