#!/bin/bash
#NOTE: check direction_vectors in locmark.py

target_dirs=(
    # "ours_full/pgd-l2-20-255/20260115-113025"
    # "ours_full/pgd-l2-25-255/20260115-120417"
    # "ours_full/pgd-l2-30-255/20260115-113235"
    # "ours_full/pgd-l2-35-255/20260115-121555"

    # "ours_random_mean/20260104-080404"
    # "ours_random/20260104-080115"
    # "ours_ones/20260104-083833"
    # "ours_random_quan/20260104-080730"

    # "ours_full/pgd-l1-13500/20260119-063406"
    # "ours_full/pgd-l1-18000/20260119-051652"
    # "ours_full/pgd-l1-22500/20260119-051759"
    # "ours_full/pgd-l1-27000/20260119-051931"
    # "ours_full/pgd-l1-31500/20260119-052023"

    # "ours_full/clamp-8/20260201-123208"
    # "ours_full/clamp-12/20260201-065353"
    # "ours_full/clamp-16/20260201-065209"
    # "ours_full/clamp-20/20260201-063324"
    # "ours_full/clamp-24/20260201-064936"
    # "ours_full/clamp-28/20260201-064337"

    "ours_wam/baseline-hinge/20260215-215311"
)

base_prefix="/mnt/nas5/suhyeon/projects/eval_spliceless"

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
        tamper_mode=zero_mask
done