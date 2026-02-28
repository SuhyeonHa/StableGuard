#!/bin/bash
#NOTE: check direction_vectors in locmark.py

target_dirs=(
    "ours_ablation/00-baseline/20260222-152841"
    "ours_ablation/01-noise/20260222-153139"
    "ours_ablation/02-noise-hinge/20260222-153450"
    "ours_ablation/03-noise-hard/20260222-153554"
    "ours_ablation/04-noise-hinge-hard/20260222-153719"
    "ours_ablation/05-hinge/20260222-153712"
    "ours_ablation/06-hinge-hard/20260222-153824"
    "ours_ablation/07-hard/20260222-154022"
)

base_prefix="/mnt/nas5/suhyeon/projects/locmark_table_1"

for sub_dir in "${target_dirs[@]}"
do
    full_save_path="${base_prefix}/${sub_dir}"
    
    echo "Processing save_path: $full_save_path"
    
    CUDA_VISIBLE_DEVICES=2 python eval_AGE.py \
        target_model=ours \
        src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
        save_path="$full_save_path" \
        edit_model_name=sd-legacy/stable-diffusion-inpainting \
        eval_size=256 \
        start_idx=0 \
        end_idx=100 \
        tamper_mode=ldm \
        use_refiner=True
done