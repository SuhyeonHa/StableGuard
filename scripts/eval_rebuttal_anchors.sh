#!/bin/bash
NUM_IMG=100
TAMPER_MODE=ldm
DATASET=/mnt/nas5/suhyeon/datasets/valAGE-Set
SAVE_BASE=/mnt/nas5/suhyeon/projects/apt_rebuttal

target_dirs=(
    # "ours_ablation/gaussian/20260505-134927"
    # "ours_ablation/quantized/20260505-134555"
    # "ours_ablation/zeromean/20260505-134809"
    "ours_ablation/rademacher/20260505-135204"
    # "ours_ablation/pca/20260505-150424"
    # "ours_ablation/centering/20260505-155259"
)

for dir in "${target_dirs[@]}"; do
    # Extract anchor_type from second path segment (e.g. ours_ablation/gaussian/... → gaussian)
    anchor_type=$(echo "$dir" | cut -d'/' -f2)
    echo "===== $dir  [anchor_type=$anchor_type] ====="
    CUDA_VISIBLE_DEVICES=2 python eval_AGE.py \
        target_model=ours \
        src_image_path=$DATASET \
        save_path=$SAVE_BASE/$dir \
        eval_size=256 \
        start_idx=0 \
        end_idx=$NUM_IMG \
        tamper_mode=$TAMPER_MODE \
        use_refiner=True \
        anchor_type=$anchor_type
done
