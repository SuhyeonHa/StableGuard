#!/bin/bash

NUM_IMG=100
TAMPER_MODEL=ldm

# AUG_TYPES=(
#     "gaussian_blur"  "gaussian_blur"
#     "gaussian_noise" "gaussian_noise" "gaussian_noise"
#     "brightness"     "brightness"
#     "median_filter"  "median_filter"
#     "jpeg"           "jpeg"           "jpeg"
# )
# AUG_PARAMS=(
#     3    5
#     1    3    5
#     -0.1  0.1
#     3    5
#     90   80   70
# )

AUG_TYPES=(
    "jpeg"
)
AUG_PARAMS=(
    95
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

    # ours (use_refiner=False)
    python eval_AGE.py \
        target_model=ours \
        src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
        save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858 \
        edit_model_name=sd-legacy/stable-diffusion-inpainting \
        eval_size=256 \
        start_idx=0 \
        end_idx=$NUM_IMG \
        tamper_mode=$TAMPER_MODEL \
        use_refiner=False \
        aug_type=$AUG_TYPE \
        aug_param=$AUG_PARAM

    # ours (use_refiner=True)
    python eval_AGE.py \
        target_model=ours \
        src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
        save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858 \
        edit_model_name=sd-legacy/stable-diffusion-inpainting \
        eval_size=256 \
        start_idx=0 \
        end_idx=$NUM_IMG \
        tamper_mode=$TAMPER_MODEL \
        use_refiner=True \
        aug_type=$AUG_TYPE \
        aug_param=$AUG_PARAM

done
