NUM_IMG=100
TAMPER_MODE=z_image
DATASET=/mnt/nas5/suhyeon/datasets/valAGE-Set
SAVE_BASE=/mnt/nas5/suhyeon/projects/locmark_table_1/ours_ablation/06-hinge-hard/20260222-153824

# ours (LocMark, with refiner)
CUDA_VISIBLE_DEVICES=1 python eval_AGE.py \
    target_model=ours \
    src_image_path=$DATASET \
    save_path=$SAVE_BASE \
    eval_size=256 \
    start_idx=43 \
    end_idx=$NUM_IMG \
    tamper_mode=$TAMPER_MODE \
    use_refiner=True \
    eval_dist=False \
    anchor_type=submitted