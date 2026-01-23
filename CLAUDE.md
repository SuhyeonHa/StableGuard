# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LocMark is a local watermarking method for tamper localization in images. It embeds watermarks via latent space optimization using direction vectors and cosine similarity, enabling detection of tampered regions without requiring binary message encoding.

## Common Commands

### Environment Setup
```bash
conda create -n stableguard python=3.12
conda activate stableguard
pip install -r requirements.txt
# Install xformers matching your PyTorch version for GPU memory efficiency
```

### Running LocMark Watermark Embedding
```bash
python -m locmark.main
```
Configuration is in `locmark/main.py` via the `Params` class. Key parameters:
- `single_image_mode`: True for single image, False for batch processing
- `epsilon`: L1 perturbation budget (e.g., 13500)
- `steps`: Optimization steps (default 300)
- `target_cossim`: Target cosine similarity threshold (default 0.2)

### Evaluation Against Tampering
```bash
CUDA_VISIBLE_DEVICES=0 python eval_AGE.py \
    target_model=ours \
    src_image_path=/path/to/valAGE-Set \
    save_path=/path/to/output \
    eval_size=256 \
    tamper_mode=ldm
```

Configuration in `config.yaml`:
- `target_model`: ours (LocMark), omniguard, wam, stableguard (baselines)
- `tamper_mode`: ldm, controlnet, hdpainter, zero_mask, vae_recon
- `aug_type`/`aug_param`: optional augmentation for robustness testing

## Architecture

### LocMark Core (`locmark/`)

**LocMark** (`locmark.py`): Main watermarking class
- Uses pretrained ConvNeXt (`convnext_small.dinov3_lvd1689m`) as feature extractor
- Direction vectors loaded from pre-generated `.pt` files for cosine similarity computation
- `embed_watermark()`: Optimizes latent perturbation to maximize cosine similarity with direction vectors
- `decode_watermark()`: Extracts tampering mask via cosine similarity thresholding
- Loss functions: PSNR loss, LPIPS loss, hard negative mining loss

**Params** (`main.py`): Configuration class with hyperparameters
- `feat_layer`: ConvNeXt layer to extract features (0-3, default 1 → 192 dims)
- `temperature`: Sigmoid scaling for confidence map
- `eps0_std`: Latent noise range for robustness training

### Baseline Methods (for comparison)
- `stableguard/`: MPW-VAE + MoE-GFN (48-bit watermark)
- `omniguard/`: OmniGuard watermarking (64-bit)
- `watermark_anything/`: WAM (32-bit)
- `HD-Painter/`: HD inpainting for tampering simulation

### Evaluation Framework (`evaluation/`)
- `PixelF1`, `PixelAUC`, `PixelIOU`, `PixelAccuracy` for tampering localization
- `augmentation.py`: robustness transforms (blur, noise, compression)

### Datasets (`dataset.py`)
- `AGEDataset`: AGE-Set for tampering detection evaluation
- `CocoDataset`: COCO 2017 for training baselines

## Key Patterns

- Image size: 256×256 for evaluation, 512×512 for VAE encoding
- Watermark detection: Cosine similarity between image features and direction vectors
- Tampering localization: Regions with low cosine similarity indicate tampering
- Direction vectors: Pre-generated universal vectors stored in `.pt` files
- Perturbation constraint: L1 norm bounded by `epsilon` parameter
