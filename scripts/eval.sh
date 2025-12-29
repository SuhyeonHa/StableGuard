#!/bin/bash

# wam
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=wam \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/wam/256_valAGE_sd_1.2_wm_wofilter \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=100 \
#     tamper_mode=inpaint \
#     aug_type=blur \
#     aug_param=15

# omniguard
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=omniguard \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/omniguard/all_512_eval_256 \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=100 \
#     tamper_mode=cover

# stableguard
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=stableguard \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/stableguard/256_valAGE_sd_1.2_wm_wofilter \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=100 \
#     tamper_mode=cover

# ours
CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
    target_model=ours \
    src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
    save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/ours/20251205-015055 \
    edit_model_name=sd-legacy/stable-diffusion-inpainting \
    eval_size=256 \
    start_idx=0 \
    end_idx=100 \
    tamper_mode=inpaint