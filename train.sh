 #!/bin/bash
 # Tips: You should replace data_root_path with your local coco dataset path
 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yaml train.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --data_root_path="/mnt/nas5/suhyeon/datasets/coco-2017" \
  --mask_pool_path="/mnt/nas5/suhyeon/datasets/coco-tamper-stableguard/datasets/mask_pool" \
  --resolution=256 \
  --train_batch_size=2 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-4 \
  --weight_decay=1e-2 \
  --output_dir="/mnt/nas5/suhyeon/projects/freqloc-ldm/exp03-add-vae-noise" \
  --save_steps=10000 \
  --num_train_epochs=10 \
  --noise_strength 0.0 0.8 \
  --cache_dir="/mnt/nas5/suhyeon/caches" \
  --watermark_size=32