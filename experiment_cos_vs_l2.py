"""
실험: 동일한 epsilon(≈PSNR) 제약 하에서 cosine loss vs l2 loss
목적: "같은 perturbation 예산에서 cosine이 l2보다
      anchor 방향을 더 효율적으로 새긴다"를 보임

핵심 수정:
  - PSNR 제약을 if/else 분기 대신 delta clamp로 구현
  - PSNR target → epsilon 변환 후 매 step에서 delta.data clamp
  - 이로써 psnr_target마다 실제 PSNR이 달라짐

예상 시간 (A100 기준):
  USE_DIFFUSION_REGEN=False (VAE only):
    embed: 50 × 5 × 2 × 300steps × ~0.15s ≈ 2,250초 ≈ 38분
    regen: 50 × 5 × 2 × ~0.3s           ≈    150초 ≈  2분
    합계: ≈ 40분

  USE_DIFFUSION_REGEN=True (SD 20 steps):
    embed: 위와 동일                               ≈ 38분
    regen: 50 × 5 × 2 × ~4s                  ≈  3,300초 ≈ 55분
    합계: ≈ 1.5시간
"""

import os
import torch
import torch.nn.functional as F
import timm
import numpy as np
import lpips
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_DIR    = '/mnt/nas5/suhyeon/datasets/DIV2K_train_HR'
CACHE_DIR    = '/mnt/nas5/suhyeon/caches'
VAE_MODEL    = 'sd-legacy/stable-diffusion-inpainting'
ENCODER_NAME = 'convnext_small.dinov3_lvd1689m'
FEAT_LAYER   = 1           # locmark.py 기준 feat_layer
FEATURE_DIM  = 192         # convnext_small layer1 channel
VAE_SIZE     = 512
IMG_SIZE     = 256
N_IMAGES     = 10
N_STEPS      = 150
LR           = 0.01
EPS_NORM     = 1e-6        # feature normalization epsilon

# PSNR 제약 레벨
PSNR_TARGETS = [28, 30, 32]

# latent space epsilon 스케일 팩터
# SD VAE latent은 pixel 대비 분산이 크므로 스케일 조정
# 경험적으로 8.0 사용 (필요시 조정)
LATENT_SCALE = 8.0

# regeneration 방식 선택
# False: VAE encode-decode만 (순수 VAE robustness 측정, 빠름 ~40분)
# True : SD inpainting diffusion 20 steps (실제 공격 시나리오, ~1.5시간)
USE_DIFFUSION_REGEN = True


# ──────────────────────────────────────────────
# PSNR → latent epsilon 변환
# ──────────────────────────────────────────────

def psnr_to_latent_eps(psnr_target, latent_scale=LATENT_SCALE):
    """
    pixel PSNR target을 latent space epsilon으로 변환
    PSNR = 20*log10(1/sqrt(MSE_pixel))
    → MSE_pixel = 10^(-PSNR/10)
    → eps_pixel = sqrt(MSE_pixel)
    → eps_latent = eps_pixel * latent_scale
    """
    mse_pixel  = 10 ** (-psnr_target / 10)
    eps_pixel  = float(np.sqrt(mse_pixel))
    eps_latent = eps_pixel * latent_scale
    return eps_latent

def psnr_to_pixel_eps(psnr_target):
    """
    PSNR = 20*log10(1/sqrt(MSE))
    → MSE = 10^(-PSNR/10)
    → eps = sqrt(MSE)   (pixel [0,1] 기준 L-inf 근사)
    """
    mse = 10 ** (-psnr_target / 10)
    return float(np.sqrt(mse))


# ──────────────────────────────────────────────
# Helper: imagenet norm/denorm
# ──────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def norm_imagenet(x):
    mean = IMAGENET_MEAN.to(x.device)
    std  = IMAGENET_STD.to(x.device)
    return (x - mean) / std

def compute_psnr(pred, target):
    mse = F.mse_loss(pred.detach(), target.detach()).item()
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


# ──────────────────────────────────────────────
# Model 로드
# ──────────────────────────────────────────────

def load_models():
    print("Loading models...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        VAE_MODEL,
        cache_dir=CACHE_DIR,
        safety_checker=None,
    ).to(DEVICE)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    pipe.unet.requires_grad_(False)
    pipe.unet.eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()

    encoder = timm.create_model(
        ENCODER_NAME,
        pretrained=True,
        features_only=True
    ).to(DEVICE)
    encoder.requires_grad_(False)
    encoder.eval()

    loss_fn_lpips = lpips.LPIPS(net='alex').to(DEVICE)
    loss_fn_lpips.eval()

    return pipe, encoder, loss_fn_lpips


# ──────────────────────────────────────────────
# Anchor vector 생성 (locmark.py 방식 동일)
# ──────────────────────────────────────────────

def generate_anchor(feature_dim, device):
    vec = torch.randn(1, feature_dim)
    vec = vec - vec.mean(dim=1, keepdim=True)
    vec = torch.sign(vec + 1e-6)
    vec = vec / torch.norm(vec, p=2, dim=1, keepdim=True)
    return vec.to(device)

def load_anchor(feature_dim, device):
    return torch.load(f'/mnt/nas5/suhyeon/projects/freq-loc/ablation_full_{feature_dim}.pt').to(device)


# ──────────────────────────────────────────────
# 이미지 로드
# ──────────────────────────────────────────────

def load_images(image_dir, n_images, img_size):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    images = []
    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])[:n_images]

    for fname in files:
        img = Image.open(os.path.join(image_dir, fname)).convert('RGB')
        images.append(transform(img))

    print(f"Loaded {len(images)} images from {image_dir}")
    return torch.stack(images)  # [N, 3, H, W]


# ──────────────────────────────────────────────
# VAE regeneration
# ──────────────────────────────────────────────

def vae_regenerate(pipe, x_wm, use_diffusion=USE_DIFFUSION_REGEN):
    with torch.no_grad():
        x_up = F.interpolate(x_wm, size=(VAE_SIZE, VAE_SIZE),
                             mode='bilinear', align_corners=False)

        if use_diffusion:
            # SD inpainting: 전체 mask → diffusion이 완전히 재생성
            zero_mask = torch.zeros(
                (x_up.shape[0], 1, VAE_SIZE, VAE_SIZE),
                dtype=x_up.dtype, device=x_up.device
            )
            out_pil = pipe(
                prompt="",
                image=x_up,
                mask_image=zero_mask,
                num_inference_steps=20
            ).images[0]
            to_tensor = transforms.ToTensor()
            x_out = to_tensor(out_pil).unsqueeze(0).to(x_wm.device)
        else:
            # VAE only: encode → decode
            z = pipe.vae.encode(x_up * 2 - 1).latent_dist.sample()
            x_out = pipe.vae.decode(z).sample
            x_out = (x_out + 1) / 2
            x_out = torch.clamp(x_out, 0, 1)

    return x_out  # [1, 3, VAE_SIZE, VAE_SIZE]


# ──────────────────────────────────────────────
# Feature cosine 측정
# ──────────────────────────────────────────────

def measure_cosine(encoder, x, anchor, feat_layer):
    """x: [1, 3, H, W] in [0,1], returns mean cosine (scalar)"""
    with torch.no_grad():
        x_norm   = norm_imagenet(x)
        features = encoder(x_norm)[feat_layer]        # [1, C, H, W]
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).view(B, H * W, C)
        features = features / (torch.norm(features, p=2, dim=-1, keepdim=True) + EPS_NORM)
        cos_sim  = torch.matmul(features, anchor.T)   # [B, HW, 1]
        return cos_sim.mean().item()


# ──────────────────────────────────────────────
# Watermark 삽입 (epsilon clamp 방식)
# ──────────────────────────────────────────────

def embed_watermark(pipe, encoder, x, anchor,
                    loss_type='cosine',
                    psnr_target=32,
                    n_steps=N_STEPS,
                    lr=LR,
                    feat_layer=FEAT_LAYER):

    x = x.unsqueeze(0).to(DEVICE)
    pixel_eps = psnr_to_pixel_eps(psnr_target)

    # VAE encode (고정)
    x_vae = F.interpolate(x, size=(VAE_SIZE, VAE_SIZE),
                          mode='bilinear', align_corners=False)
    with torch.no_grad():
        latent = pipe.vae.encode(2 * x_vae - 1).latent_dist.sample()

    delta = torch.zeros_like(latent, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    x_orig = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE),
                           mode='bilinear', align_corners=False).detach()

    for step in range(n_steps):
        optimizer.zero_grad()

        # decode
        x_wm_vae = pipe.vae.decode(latent + delta).sample
        x_wm_vae = (x_wm_vae + 1) / 2

        # ── pixel space에서 직접 clamp
        with torch.no_grad():
            pixel_delta = x_wm_vae - x_vae
            pixel_delta = torch.clamp(pixel_delta, -pixel_eps, pixel_eps)
            x_wm_vae_clamped = torch.clamp(x_vae + pixel_delta, 0, 1)
        # gradient는 clamp 전 x_wm_vae로 흘러야 하므로
        # straight-through estimator 적용
        x_wm_vae = x_wm_vae + (x_wm_vae_clamped - x_wm_vae).detach()

        x_wm = F.interpolate(x_wm_vae, size=(IMG_SIZE, IMG_SIZE),
                             mode='bilinear', align_corners=False)

        # feature 추출
        x_wm_norm    = norm_imagenet(x_wm)
        features_raw = encoder(x_wm_norm)[feat_layer]
        B, C, H, W   = features_raw.shape
        features_flat = features_raw.permute(0, 2, 3, 1).view(B, H * W, C)
        features_norm = features_flat / (
            torch.norm(features_flat, p=2, dim=-1, keepdim=True) + EPS_NORM
        )

        if loss_type == 'cosine':
            cos_sim = torch.matmul(features_norm, anchor.T)
            loss_wm = torch.mean(1 - cos_sim)
        elif loss_type == 'l2':
            anchor_exp = anchor.unsqueeze(0).expand(B, H * W, -1)
            loss_wm = F.mse_loss(features_flat, anchor_exp)

        loss_wm.backward()
        optimizer.step()

    # 최종
    with torch.no_grad():
        x_wm_final_vae = pipe.vae.decode(latent + delta).sample
        x_wm_final_vae = (x_wm_final_vae + 1) / 2
        pixel_delta = x_wm_final_vae - x_vae
        pixel_delta = torch.clamp(pixel_delta, -pixel_eps, pixel_eps)
        x_wm_final_vae = torch.clamp(x_vae + pixel_delta, 0, 1)
        x_wm_final = F.interpolate(x_wm_final_vae, size=(IMG_SIZE, IMG_SIZE),
                                   mode='bilinear', align_corners=False)

    final_psnr = compute_psnr(x_wm_final, x_orig)
    return x_wm_final.detach(), final_psnr


# ──────────────────────────────────────────────
# 메인 실험
# ──────────────────────────────────────────────

def run_experiment():
    # ── 예상 시간 및 설정 출력
    regen_mode = "diffusion (~1.5hr)" if USE_DIFFUSION_REGEN else "VAE-only (~40min)"
    print("=" * 55)
    print(f"  N_IMAGES={N_IMAGES}, N_STEPS={N_STEPS}, LR={LR}")
    print(f"  PSNR_TARGETS={PSNR_TARGETS}")
    print(f"  LATENT_SCALE={LATENT_SCALE}")
    print(f"  Regeneration: {regen_mode}")
    print("=" * 55)

    print("\nPSNR → latent epsilon 변환:")
    for pt in PSNR_TARGETS:
        print(f"  PSNR={pt}dB  →  eps_latent={psnr_to_latent_eps(pt):.5f}")
    print()

    # ── 모델 로드
    pipe, encoder, _ = load_models()

    # ── anchor 생성
    # anchor = generate_anchor(FEATURE_DIM, DEVICE)
    anchor = load_anchor(FEATURE_DIM, DEVICE)
    print(f"Anchor: shape={anchor.shape}, norm={anchor.norm().item():.4f}\n")

    # ── 이미지 로드
    images = load_images(IMAGE_DIR, N_IMAGES, IMG_SIZE)
    print(f"Images: {images.shape}\n")

    # ── 결과 저장 구조
    results = {
        loss_t: {
            psnr: {'before': [], 'after': [], 'actual_psnr': []}
            for psnr in PSNR_TARGETS
        }
        for loss_t in ['cosine', 'l2']
    }

    for img_idx, x in enumerate(tqdm(images, desc="Images")):
        print(f"\n=== Image {img_idx+1}/{len(images)} ===")

        for loss_type in ['cosine', 'l2']:
            for psnr_target in PSNR_TARGETS:
                eps = psnr_to_latent_eps(psnr_target)
                print(f"  [{loss_type}] psnr_target={psnr_target}dB  eps={eps:.5f}")

                # watermark 삽입
                x_wm, actual_psnr = embed_watermark(
                    pipe, encoder, x, anchor,
                    loss_type=loss_type,
                    psnr_target=psnr_target,
                    n_steps=N_STEPS,
                    feat_layer=FEAT_LAYER
                )

                # VAE regeneration
                x_regen = vae_regenerate(pipe, x_wm)
                x_regen = F.interpolate(x_regen, size=(IMG_SIZE, IMG_SIZE),
                                        mode='bilinear', align_corners=False)

                # cosine 측정
                cos_before = measure_cosine(encoder, x_wm,    anchor, FEAT_LAYER)
                cos_after  = measure_cosine(encoder, x_regen, anchor, FEAT_LAYER)

                results[loss_type][psnr_target]['before'].append(cos_before)
                results[loss_type][psnr_target]['after'].append(cos_after)
                results[loss_type][psnr_target]['actual_psnr'].append(actual_psnr)

                print(f"    actual PSNR: {actual_psnr:.2f}dB")
                print(f"    cos before:  {cos_before:.4f}")
                print(f"    cos after:   {cos_after:.4f}")

    save_results(results)
    plot_results(results)


# ──────────────────────────────────────────────
# 결과 저장
# ──────────────────────────────────────────────

def save_results(results):
    out = {}
    for loss_type in results:
        out[loss_type] = {}
        for psnr_target in results[loss_type]:
            d = results[loss_type][psnr_target]
            out[loss_type][str(psnr_target)] = {
                'before_mean':  float(np.mean(d['before'])),
                'before_std':   float(np.std(d['before'])),
                'after_mean':   float(np.mean(d['after'])),
                'after_std':    float(np.std(d['after'])),
                'actual_psnr':  float(np.mean(d['actual_psnr'])),
            }

    path = '/mnt/user-data/outputs/results_cos_vs_l2.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {path}")


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────

def plot_results(results):
    colors = {'cosine': '#2196F3', 'l2': '#F44336'}

    # ── 그래프 1: before / after cosine by PSNR
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, metric in enumerate(['before', 'after']):
        ax    = axes[ax_idx]
        title = 'Before Regeneration' if metric == 'before' else 'After Regeneration'
        for loss_type in ['cosine', 'l2']:
            means, stds = [], []
            for pt in PSNR_TARGETS:
                vals = results[loss_type][pt][metric]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(PSNR_TARGETS, means, yerr=stds,
                        label=loss_type, color=colors[loss_type],
                        marker='o', linewidth=2, capsize=4)
        ax.set_xlabel('PSNR Target (dB)', fontsize=12)
        ax.set_ylabel('cos(f(x), anchor)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.0)

    plt.suptitle(
        'Cosine vs L2 Loss — Anchor Alignment Efficiency\n'
        'under Same Latent Epsilon (PSNR) Constraint',
        fontsize=14
    )
    plt.tight_layout()
    p1 = '/mnt/user-data/outputs/results_cos_vs_l2.png'
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {p1}")
    plt.close()

    # ── 그래프 2: cosine drop (before - after)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    for loss_type in ['cosine', 'l2']:
        drops = []
        for pt in PSNR_TARGETS:
            before = np.mean(results[loss_type][pt]['before'])
            after  = np.mean(results[loss_type][pt]['after'])
            drops.append(before - after)
        ax2.plot(PSNR_TARGETS, drops,
                 label=loss_type, color=colors[loss_type],
                 marker='s', linewidth=2)
    ax2.set_xlabel('PSNR Target (dB)', fontsize=12)
    ax2.set_ylabel('Cosine Drop (before − after)', fontsize=12)
    ax2.set_title('Regeneration Robustness: Cosine Drop by PSNR', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = '/mnt/user-data/outputs/results_cos_vs_l2_drop.png'
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    print(f"Drop plot saved → {p2}")
    plt.close()

    # ── 그래프 3: actual PSNR 검증 (epsilon clamp가 제대로 동작하는지)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for loss_type in ['cosine', 'l2']:
        actual_psnrs = [
            np.mean(results[loss_type][pt]['actual_psnr'])
            for pt in PSNR_TARGETS
        ]
        ax3.plot(PSNR_TARGETS, actual_psnrs,
                 label=loss_type, color=colors[loss_type],
                 marker='o', linewidth=2)
    ax3.plot(PSNR_TARGETS, PSNR_TARGETS, 'k--', label='target (ideal)', linewidth=1)
    ax3.set_xlabel('PSNR Target (dB)', fontsize=12)
    ax3.set_ylabel('Actual PSNR (dB)', fontsize=12)
    ax3.set_title('Actual vs Target PSNR (epsilon clamp 검증)', fontsize=13)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = '/mnt/user-data/outputs/results_actual_psnr.png'
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    print(f"PSNR verification plot saved → {p3}")
    plt.close()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == '__main__':
    run_experiment()