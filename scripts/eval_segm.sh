#!/bin/bash

TAMPER_MODELS=("controlnet" "hdpainter" "brushnet")

for mode in "${TAMPER_MODELS[@]}"; do
    echo "=================================================="
    echo "Running eval_AGE.py with tamper_mode: $mode"
    echo "=================================================="

    # wam
    # python eval_AGE.py \
    #     target_model=wam \
    #     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
    #     save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/wam \
    #     edit_model_name=sd-legacy/stable-diffusion-inpainting \
    #     eval_size=256 \
    #     start_idx=0 \
    #     end_idx=100 \
    #     tamper_mode=$mode \
    #     wm_strength=3.0 \
    #     segmentation_mask=True

    # omniguard
    # python eval_AGE.py \
    #     target_model=omniguard \
    #     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
    #     save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard \
    #     edit_model_name=sd-legacy/stable-diffusion-inpainting \
    #     eval_size=256 \
    #     start_idx=0 \
    #     end_idx=100 \
    #     tamper_mode=$mode \
    #     wm_strength=2.0 \
    #     segmentation_mask=True

    # stableguard
    # python eval_AGE.py \
    #     target_model=stableguard \
    #     src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
    #     save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard \
    #     edit_model_name=sd-legacy/stable-diffusion-inpainting \
    #     eval_size=256 \
    #     start_idx=0 \
    #     end_idx=100 \
    #     tamper_mode=$mode \
    #     segmentation_mask=True

    # ours
    python eval_AGE.py \
        target_model=ours \
        src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
        save_path=/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858 \
        edit_model_name=sd-legacy/stable-diffusion-inpainting \
        eval_size=256 \
        start_idx=0 \
        end_idx=100 \
        tamper_mode=$mode \
        segmentation_mask=True \
        use_refiner=False

done