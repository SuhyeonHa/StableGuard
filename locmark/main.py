import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import os
import warnings 
import torchvision
import json
import sys
from datetime import datetime
import yaml
import time
import torchvision.transforms as transforms
from .helper import load_images_from_path, load_image, Tee, save_images
from .locmark import LocMark
from tqdm import tqdm
warnings.filterwarnings('ignore')


class Params:
    """Hyperparameters and configuration settings for LocMark."""
    def __init__(self):
        # --- System & Paths ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_datasets = '/mnt/nas5/suhyeon/datasets/valAGE-Set'
        self.image_path = '/mnt/nas5/suhyeon/datasets/valAGE-Set/0034.png'
        self.exp_name = 'noise-hard-4'
        # self.output_dir = f'/mnt/nas5/suhyeon/projects/locmark/{self.exp_name}' # single image optimization
        self.output_dir = f'/mnt/nas5/suhyeon/projects/eval_spliceless/ours_convnext2' # NOTE: multi image optimization, exp_name
        # self.output_dir = "/mnt/nas5/suhyeon/projects/locmark/" # single image optimization
        self.single_image_mode = False # NOTE
        self.num_test_images = 100 # the first n images

        # --- Model Configurations ---
        self.vae_model_name = "stabilityai/stable-diffusion-2-1"
        self.vae_subfolder = "vae"
        
        # --- Image Size Parameters ---
        self.vae_image_size = 512
        self.image_size = 256
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        # --- LocMark Core Parameters ---
        self.margin = 1.0
        self.grid_size = 28
        self.mask_percentage = 0.3
        self.num_masks = 1
        self.seed = 42
        self.num_inference_steps = 100
        self.guidance_scale = 7.5
        self.temperature = 5.0
        self.target_cossim = 0.2

        # --- Optimization Parameters ---
        self.lr = 2.0
        self.steps = 300
        self.lambda_p = 0.025 #0.1 #0.05 #0.025 #0.01
        self.lambda_i = 0.005 #0.05 #0.01 #0.005 #0.001 
        self.lambda_clean = 1.0
        self.lambda_noisy = 1.0
        self.feat_layer = 1
        self.epsilon = 4/255

        # --- JND (Just Noticeable Difference) Parameters ---
        self.use_jnd = False  # Enable JND-based perceptual masking
        self.jnd_alpha = 1.0  # JND modulation strength (higher = more aggressive masking)

        # --- Robustness Parameters ---
        self.eps0_std = [0.0, 0.25] # Latent noise
        
        # --- Demo/Evaluation Parameters ---
        self.batch_size = 1
        # self.num_test_images = 1

        self.feature_dim = None
        # swinv2_small_window8_256
        if self.feat_layer == 0:
            self.feature_dim = 96
        elif self.feat_layer == 1:
            self.feature_dim = 256 #convnext2
        elif self.feat_layer == 2:
            self.feature_dim = 384
        elif self.feat_layer == 3:
            self.feature_dim = 768

def run_locmark(args=None, save_dir=None):
    """Run complete LocMark demonstration"""

    # Initialize LocMark
    locmark = LocMark(args=args)
    results = {}
    all_times = []

    if args.single_image_mode:
        args.num_test_images = 1

        print("Training on a single image mode.")
        print(f"Setup: {args.num_test_images} images, {args.image_size}x{args.image_size}")

        test_images = load_image(args.image_path, transform=args.transform)

        for i in tqdm(range(args.num_test_images), desc="Embedding Watermarks"):
            original = test_images[i:i+1].to(args.device)
            filename = '0034.png'

            # Embed watermarks
            print("Embedding watermarks for:", filename)
            start_time = time.time()
            watermarked, watermark = locmark.embed_watermark(original, img_size=args.image_size)
            end_time = time.time()
            all_times.append(end_time - start_time)

            # Decode watermarks
            print("Decoding watermarks...")
            original = F.interpolate(original, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            watermarked = F.interpolate(watermarked, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            watermark = F.interpolate(watermark, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            logits, prediction, bin_prediction = locmark.decode_watermark(watermarked)

            # Save images
            vis_results = [original, watermarked, prediction, watermark]
            save_images(vis_results, filename, save_dir)

            psnr = locmark._compute_psnr(original, watermarked)
            results[filename] = psnr
    
    else:
        test_images, filenames = load_images_from_path(args.train_datasets, args.num_test_images, transform=args.transform)

        print("Training on multiple images mode.")
        print(f"Setup: {args.num_test_images} images, {args.image_size}x{args.image_size}")
        
        os.makedirs(os.path.join(save_dir, "cover_images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "watermark"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "prediction"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "bin_prediction"), exist_ok=True)
        
        for i in range(args.num_test_images):
            original = test_images[i:i+1].to(args.device)
            filename = filenames[i]

            # Embed watermarks
            print("Embedding watermarks for:", filename)
            start_time = time.time()
            watermarked, watermark = locmark.embed_watermark(original, img_size=args.image_size)
            end_time = time.time()
            all_times.append(end_time - start_time)

            # Decode watermarks
            print("Finish optimization.")
            original = F.interpolate(original, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            watermarked = F.interpolate(watermarked, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            watermark = F.interpolate(watermark, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            logits, prediction, bin_prediction = locmark.decode_watermark(watermarked)

            torchvision.utils.save_image(watermarked.cpu(), os.path.join(save_dir, "cover_images", filename))
            torchvision.utils.save_image(watermark.cpu()*5, os.path.join(save_dir, "watermark", filename))
            torchvision.utils.save_image(prediction.cpu(), os.path.join(save_dir, "prediction", filename))
            torchvision.utils.save_image(bin_prediction.cpu(), os.path.join(save_dir, "bin_prediction", filename))

            psnr = locmark._compute_psnr(original, watermarked)
            results[filename] = psnr


    # Calculate metrics
    avg_psnr = np.mean(list(results.values()))
    avg_time = np.mean(all_times)

    print("\n=== Processing Complete ===")
    print(f"Processed {args.num_test_images} images.")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average Inference Time: {avg_time:.2f} seconds")

    results_path = os.path.join(exp_dir, 'results.json')
    results_for_json = {k: v for k, v in results.items() if not isinstance(v, torch.Tensor)}
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=4)
    print(f"Final results saved to {results_path}")
    
    return avg_psnr


if __name__ == "__main__":
    args = Params()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(args.output_dir, args.exp_name, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Print and save configuration
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        config_to_save = args.__dict__.copy()
        if 'transform' in config_to_save:
            config_to_save['transform'] = str(config_to_save['transform'])
        if 'device' in config_to_save:
            config_to_save['device'] = str(config_to_save['device'])

        config_dict = {k: v for k, v in config_to_save.items() if not k.startswith('__')}

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print("--- Configuration ---")
    print(yaml.dump(config_dict, default_flow_style=False))
    print("---------------------")

    log_path = os.path.join(exp_dir, 'training_log.txt')
    original_stdout = sys.stdout

    # Run FreqLoc
    with open(log_path, 'w') as log_file:
        tee = Tee(original_stdout, log_file)
        sys.stdout = tee

        print("Starting LocMark implementation and evaluation...")

        results = run_locmark(args=args, save_dir=exp_dir)

        sys.stdout = original_stdout

        print(f"Training log saved to {log_path}")
        print("Demo complete.")