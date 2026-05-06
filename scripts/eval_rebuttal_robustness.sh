#!/bin/bash

NUM_IMG=100
TAMPER_MODEL=ldm

AUG_TYPES=(
    # "resize"
    "crop"
    # "jpeg" 
)
AUG_PARAMS=(
    # 0.9
    0.9
    # 80
)


for i in "${!AUG_TYPES[@]}"; do
    AUG_TYPE=${AUG_TYPES[$i]}
    AUG_PARAM=${AUG_PARAMS[$i]}

    echo "========================================"
    echo "aug_type=$AUG_TYPE  aug_param=$AUG_PARAM"
    echo "========================================"

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
        wm_strength=3.0 \
        aug_type=$AUG_TYPE \
        aug_param=$AUG_PARAM

    # omniguard
    python eval_AGE.py \
        target_model=omniguard \
        src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
        save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard \
        edit_model_name=sd-legacy/stable-diffusion-inpainting \
        eval_size=256 \
        start_idx=0 \
        end_idx=$NUM_IMG \
        tamper_mode=$TAMPER_MODEL \
        wm_strength=2.0 \
        aug_type=$AUG_TYPE \
        aug_param=$AUG_PARAM

    # stableguard
    python eval_AGE.py \
        target_model=stableguard \
        src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
        save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard \
        edit_model_name=sd-legacy/stable-diffusion-inpainting \
        eval_size=256 \
        start_idx=0 \
        end_idx=$NUM_IMG \
        tamper_mode=$TAMPER_MODEL \
        aug_type=$AUG_TYPE \
        aug_param=$AUG_PARAM

done
