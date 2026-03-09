#!/bin/bash

TAMPER_MODELS=("controlnet" "hdpainter" "brushnet")

for mode in "${TAMPER_MODELS[@]}"; do
    echo "=================================================="
    echo "Running eval_AGE.py with tamper_mode: $mode"
    echo "=================================================="

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
        segmentation_mask=True

done