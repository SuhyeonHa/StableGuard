#!/bin/bash

# target_dirs=(
#     "ours/supp-steps-50/20260308-083200"
#     "ours/supp-steps-100/20260308-083700"
#     "ours/supp-steps-200/20260308-083855"
# )

# target_dirs=(
#     "ours/hinge-hard-noise-eps-12/20260228-205531"
#     "ours/hinge-hard-noise-eps-20/20260228-205659"
#     "ours/hinge-hard-noise-target-0.15/20260228-204847"
#     "ours/hinge-hard-noise-target-0.2/20260228-205133"
# )

# target_dirs=(
#     "ours/supp-target-0.05/20260308-141050"
#     "ours/supp-psnr-0.025/20260308-144231"
#     "ours/supp-psnr-0.05/20260308-144108"
#     "ours/supp-psnr-0.2/20260308-144449"
# )

# target_dirs=(
#     "ours/supp-anchor-map/20260307-124022"
#     "ours/supp-psnr-8/255/20260309-050736"
#     "ours/supp-psnr-0.15/20260309-044932"
#     "ours/supp-psnr-anchor-map-2/20260309-053450"
# )

# target_dirs=(
#     "ours/hinge-hard-noise-sd1/20260228-193241"
# )

target_dirs=(
    "ours/supp-vit-l5/20260310-110335"
)

base_prefix="/mnt/nas5/suhyeon/projects/locmark_table_1"

for sub_dir in "${target_dirs[@]}"
do
    full_save_path="${base_prefix}/${sub_dir}"
    
    echo "Processing save_path: $full_save_path"
    
    CUDA_VISIBLE_DEVICES=3 python eval_AGE.py \
        target_model=ours \
        src_image_path=/mnt/nas5/suhyeon/datasets/valAGE-Set \
        save_path="$full_save_path" \
        edit_model_name=sd-legacy/stable-diffusion-inpainting \
        eval_size=256 \
        start_idx=0 \
        end_idx=100 \
        tamper_mode=ldm \
        use_refiner=False
done