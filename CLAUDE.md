# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains **StableGuard** (NeurIPS 2025) — a framework for AI-generated image tamper localization using watermarking. The repo includes the proposed **LocMark** method (latent-space optimization with direction vectors) and three baselines: StableGuard (MPW-VAE + MoE-GFN), OmniGuard, and WAM (Watermark Anything).

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
- `epsilon`: L1 perturbation budget (default `16/255`)
- `steps`: Optimization steps (default 150)
- `target_cossim`: Target cosine similarity threshold (default 0.1)
- `feat_layer`: ConvNeXt layer (0–3, default 1 → 192-dim features)

### Evaluation Against Tampering
```bash
CUDA_VISIBLE_DEVICES=0 python eval_AGE.py \
    target_model=ours \
    src_image_path=/path/to/valAGE-Set \
    save_path=/path/to/output \
    eval_size=256 \
    tamper_mode=ldm
```

`eval_AGE.py` uses Hydra-style CLI overrides against `config.yaml`. Key config fields:
- `target_model`: `ours`, `ours_e2e`, `omniguard`, `wam`, `stableguard`, `clean`
- `tamper_mode`: `ldm`, `controlnet`, `hdpainter`, `zero_mask`, `vae_regen`, `sdxl`, `flux`, `brushnet`
- `aug_type` / `aug_param`: optional augmentation for robustness testing
- `weight_paths`: per-model checkpoint directories
- `use_refiner`: use `ShallowUpDecoder` for LocMark mask refinement (vs. bilinear upsample)
- `eval_dist`: compute cosine similarity distribution (watermarked/clean/inpainted)

Evaluation scripts for specific setups are in `scripts/` (`eval.sh`, `eval_ablation.sh`, `eval_robustness.sh`).

## Architecture

### LocMark (`locmark/`)

**Core method** — no binary message encoding; uses direction vectors and cosine similarity.

- `locmark.py` — `LocMark` class:
  - Feature extractor: `convnext_small.dinov3_lvd1689m` (timm)
  - `embed_watermark()`: Adam-based latent perturbation optimization. Loss = cosine similarity loss + PSNR loss (`lambda_p`) + LPIPS loss (`lambda_i`) + hard negative mining loss (`lambda_clean`, `lambda_noisy`)
  - `decode_watermark()`: Returns `(logits, confidence_map, binary_prediction)` via cosine similarity thresholding; optionally refined by `ShallowUpDecoder`
  - Direction vectors: zero-mean, sign-quantized, L2-normalized random vectors stored as `.pt` files; loaded at init

- `main.py` — `Params` dataclass + `run_locmark()` entry point

- `train_decoder.py` — `ShallowUpDecoder` for mask refinement (loaded when `use_refiner=True`)

**ConvNeXt feature dimensions by layer:**
| `feat_layer` | Dims |
|---|---|
| 0 | 96 |
| 1 (default) | 192 |
| 2 | 384 |
| 3 | 768 |

### LocMark E2E (`locmark_e2e/`)

End-to-end variant built on top of `watermark_anything/`. Uses ImageNet normalization for both embedder and detector (`normalization: imagenet` in config).

### Baseline Methods

- `stableguard/models/mpw_vae.py` — `MultiplexingWatermarkVAEDecoder`: lightweight adapter on a pretrained VAE; `MsgAdapter` encodes 48-bit binary watermark into latent space
- `stableguard/models/moe_gfn.py` — `MoEGuidedForensicNet`: mixture-of-experts forensic network combining watermark patterns, tampering traces, and frequency-domain cues
- `omniguard/` — 64-bit watermarking via Vision Transformer + DWT/IWT
- `watermark_anything/` — 32-bit WAM baseline

**Normalization per model** (used in `eval_AGE.py` for preprocessing):
- `rescale` ([-1,1]): stableguard, omniguard, ours
- `imagenet`: wam, ours_e2e

### Evaluation Framework (`evaluation/`)

Pixel-level metrics: `PixelF1`, `PixelAUC`, `PixelIOU`, `PixelAccuracy`.
Augmentation robustness transforms in `augmentation.py` (blur, noise, compression).

### Datasets (`dataset.py`)

- `AGEDataset`: AGE-Set for tampering detection evaluation (256×256)
- `CocoDataset`: COCO 2017 for baseline training
- `ImageDataset`: generic loader

## Key Patterns

- Eval image size: 256×256; VAE encoding: 512×512
- Tampering localization: low cosine similarity regions → tampered
- `eval_AGE.py` drives all method comparisons; `config.yaml` is the single source of truth for paths and per-model settings
- All path configuration (checkpoint dirs, dataset paths) lives in `config.yaml` `weight_paths` section
