#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python eval_AGE.py \
    target_model=omniguard \
    src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
    save_path=/mnt/nas5/suhyeon/projects/eval_spliceless/omniguard/all_512_eval_256 \
    edit_model_name=sd-legacy/stable-diffusion-inpainting \
    eval_size=256 \
    start_idx=0 \
    end_idx=100 \
    tamper_mode=zero_mask