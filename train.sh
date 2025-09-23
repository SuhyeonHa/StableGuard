 # Tips: You should replace data_root_path with your local coco dataset path
 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 58110 --multi-gpu --mixed_precision "bf16" \
  train.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --data_root_path="/home/yanghaoxin/dataset/coco" \
  --mask_pool_path="datasets/mask_pool" \
  --resolution=256 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-4 \
  --weight_decay=1e-2 \
  --output_dir="run/exp1" \
  --save_steps=10000 \
  --num_train_epochs=10 \
  --num_bits=48