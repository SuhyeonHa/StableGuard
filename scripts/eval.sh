#!/bin/bash

# wam
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=wam \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/wam/concept_fig_3.0 \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=100 \
#     tamper_mode=ldm \
#     wm_strength=3.0

# omniguard
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=omniguard \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/omniguard/concept_fig_2.0 \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=100 \
#     tamper_mode=ldm \
#     wm_strength=2.0

# stableguard
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=stableguard \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/stableguard/256_valAGE_sd_1.2_wm_wofilter \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=100 \
#     tamper_mode=zero_mask

# # ours
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=ours \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/ours/20251205-015055 \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=100 \
#     tamper_mode=inpaint

# ours
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=ours \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/ours_full/test \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=100 \
#     tamper_mode=hdpainter

# ours_e2e
CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
    target_model=ours_e2e \
    src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
    save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/ours_e2e/test \
    eval_size=256 \
    start_idx=0 \
    end_idx=10 \
    tamper_mode=ldm