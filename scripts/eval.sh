#!/bin/bash

NUM_IMG=100
TAMPER_MODEL=flux

# wam
python eval_AGE.py \
    target_model=wam \
    src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
    save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/wam \
    edit_model_name=sd-legacy/stable-diffusion-inpainting \
    eval_size=256 \
    start_idx=0 \
    end_idx=$NUM_IMG \
    tamper_mode=$TAMPER_MODEL \
    wm_strength=3.0

# # # # omniguard
# python eval_AGE.py \
#     target_model=omniguard \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODEL \
#     wm_strength=2.0

# # # # # stableguard
# python eval_AGE.py \
#     target_model=stableguard \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODEL

# # # ours
# python eval_AGE.py \
#     target_model=ours \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858 \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODEL \
#     use_refiner=False

# python eval_AGE.py \
#     target_model=ours \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858 \
#     edit_model_name=sd-legacy/stable-diffusion-inpainting \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=$NUM_IMG \
#     tamper_mode=$TAMPER_MODEL \
#     use_refiner=True

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
# CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
#     target_model=ours_e2e \
#     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
#     save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/ours_e2e/test \
#     eval_size=256 \
#     start_idx=0 \
#     end_idx=10 \
#     tamper_mode=ldm