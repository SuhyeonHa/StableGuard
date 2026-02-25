"""
Motivation Experiment: Feature map layer (0,1,2,3) 비교
- 간단한 워터마크 embedding (1-cos_sim loss + PSNR loss)
- L∞ epsilon은 locmark 방식: latent 최적화 후 최종 pixel space에서만 clamp
- Perturbation 시각화
- fg_mask(center 50%)로 inpainting 1회: FG 새로 생성, BG 유지
- watermarked vs after-inpaint heatmap을 FG/BG 영역별로 비교
- 마지막에 전체 이미지 평균 summary figure 저장

NOTE: L∞ epsilon 통일은 perturbation 상한만 보장.
      layer마다 optimizer가 실제 사용하는 perturbation 크기가 달라
      PSNR이 완전히 동일하게 나오지는 않음. 결과 figure에 실제 PSNR 표시.
"""

import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import timm
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from torchvision.utils import save_image
from glob import glob
import pickle
import argparse

# ────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE   = 256
VAE_SIZE     = 512
STEPS        = 50
LR           = 1.0
LAMBDA_P     = 0.015
# L∞ epsilon: 최종 pixel space clamp 기준 (locmark 동일 방식).
# 학습 중에는 latent delta에 제약 없음. 최종에만 pixel clamp 적용.
# 참고: ε=0.05 → 이론 max PSNR ≈ 26dB, ε=0.03 → ≈ 30dB
EPSILON      = 32/255
LAMBDA_CLEAN = 1.0
NUM_IMAGES   = 100
CROP_RATIO   = 0.5

TRAIN_DIR    = "/mnt/nas5/suhyeon/datasets/coco-2017/train2017"
# OUT_DIR      = "/mnt/nas5/suhyeon/projects/locmark_motiv_fig/p0.05"
OUT_DIR      = "/mnt/nas5/suhyeon/projects/locmark_motiv_fig/p0.015_seed19"
CACHE_DIR    = "/mnt/nas5/suhyeon/caches"
VEC_DIR      = os.path.join(OUT_DIR, "direction_vectors")

LAYERS       = [0, 1, 2, 3]
FEAT_DIMS    = {0: 96, 1: 192, 2: 384, 3: 768}
FEAT_SPATIAL = {0: 64, 1: 32,  2: 16,  3: 8}

SEED = 19

# ── RUN MODE ────────────────────────────────
# 'full'         : embed + inpaint + all figures (default)
# 'avg_only'     : load records.pkl → avg figures only (no model needed)
# 'summary_only' : load records.pkl → summary figure only (no model needed)
MODE = 'full'
RECORDS_PATH     = os.path.join(OUT_DIR, 'records.pkl')
DELTA_LISTS_PATH = os.path.join(OUT_DIR, 'delta_lists.pkl')

EPS = 1e-6

# heatmap cos_sim 고정 범위 (모든 figure 공통)
HMAP_VMIN = -0.3
HMAP_VMAX =  0.5
DIFF_VLIM =  0.4   # Δheatmap diverging colorbar: [-DIFF_VLIM, +DIFF_VLIM]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VEC_DIR, exist_ok=True)

# ────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def norm_imagenet(x):
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)

def denorm_imagenet(x):
    return (x * IMAGENET_STD.to(x.device)) + IMAGENET_MEAN.to(x.device)

def compute_psnr(pred, target):
    """pixel [0,1] tensor → scalar dB"""
    mse = F.mse_loss(pred.clamp(0,1), target.clamp(0,1)).item()
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))

def psnr_loss(pred, target):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + 1e-8))

def load_image(path, size=IMAGE_SIZE):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t

def tensor_to_pil(t):
    if t.dim() == 4:
        t = t.squeeze(0)
    t = t.clamp(0, 1).cpu()
    return transforms.ToPILImage()(t)

def make_center_crop_mask(size, ratio=0.5):
    """center ratio×ratio 영역 = 1 (foreground)"""
    mask = torch.zeros(1, 1, size, size)
    h = w = int(size * ratio)
    sy = (size - h) // 2
    sx = (size - w) // 2
    mask[:, :, sy:sy+h, sx:sx+w] = 1.0
    return mask

def generate_direction_vector(feat_dim, layer_idx, seed=42):
    path = os.path.join(VEC_DIR, f"dir_vec_layer{layer_idx}.pt")
    if os.path.exists(path):
        print(f"  [vec] load layer{layer_idx} from {path}")
        return torch.load(path, map_location=DEVICE)
    torch.manual_seed(seed + layer_idx)
    v = torch.randn(1, feat_dim)
    v = v - v.mean(dim=1, keepdim=True)
    v = torch.sign(v + 1e-6)
    v = v / torch.norm(v, p=2, dim=1, keepdim=True)
    torch.save(v.to(DEVICE), path)
    print(f"  [vec] generated & saved layer{layer_idx}: {path}")
    return v.to(DEVICE)

def get_fg_mask_feat(fg_mask_np, feat_h):
    """fg_mask_np [256,256] → bool [feat_h*feat_h]"""
    resized = np.array(
        Image.fromarray((fg_mask_np * 255).astype(np.uint8)).resize(
            (feat_h, feat_h), Image.NEAREST)
    ).flatten() > 127
    return resized

# ────────────────────────────────────────────
# MODEL INIT
# ────────────────────────────────────────────
if MODE == 'full':
    print("Loading models...")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-inpainting",
        safety_checker=None,
        cache_dir=CACHE_DIR,
    ).to(DEVICE)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.eval(); pipe.unet.eval(); pipe.text_encoder.eval()

    image_encoder = timm.create_model(
        'convnext_small.dinov3_lvd1689m',
        pretrained=True,
        features_only=True
    ).to(DEVICE)
    image_encoder.eval()
    for p in image_encoder.parameters():
        p.requires_grad = False

    avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1).to(DEVICE)

    print("Models loaded.")

# ────────────────────────────────────────────
# EMBED
# ────────────────────────────────────────────

def embed_watermark(image_01, direction_vec, feat_layer, save_path=None):
    """
    locmark 방식의 L∞ epsilon 적용:
      1. latent space에서 delta_m을 자유롭게 최적화 (학습 중 clamp 없음)
      2. 최종에만 pixel space에서 clamp(-ε, ε) 적용
         final_delta = clamp(rec_wm - orig_pixel, -ε, ε)
         final_wm    = clamp(orig_pixel + final_delta, 0, 1)

    save_path: 지정 시 optimized tensor (wm_img, delta) 저장 → 재실행 시 로드

    Returns: wm_img [1,3,256,256], delta [1,3,256,256], psnr (float)
    """
    image_01 = image_01.to(DEVICE)

    # ── 캐시 로드 ──
    if save_path is not None and os.path.exists(save_path):
        print(f"  [embed] load cached tensor: {save_path}")
        ckpt = torch.load(save_path, map_location=DEVICE, weights_only=False)
        return ckpt['wm_img'], ckpt['delta'], ckpt['psnr']

    # VAE encode (VAE_SIZE 해상도)
    image_vae = F.interpolate(image_01, size=(VAE_SIZE, VAE_SIZE),
                               mode="bilinear", align_corners=False)
    with torch.no_grad():
        latent = pipe.vae.encode(2 * image_vae - 1).latent_dist.sample()

    # latent space delta (학습 중 clamp 없음 — locmark 동일)
    delta_m = torch.zeros_like(latent, requires_grad=True)
    optimizer = optim.Adam([delta_m], lr=LR)

    # feature 계산용 image (IMAGE_SIZE)
    image_enc = F.interpolate(image_01, size=(IMAGE_SIZE, IMAGE_SIZE),
                               mode="bilinear", align_corners=False)

    for step in range(STEPS):
        optimizer.zero_grad()

        perturbed_latent = latent + delta_m
        wm_vae = pipe.vae.decode(perturbed_latent).sample
        wm_01  = (wm_vae + 1) / 2  # [0,1], VAE_SIZE

        wm_256 = F.interpolate(wm_01, size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode="bilinear", align_corners=False)
        wm_norm = norm_imagenet(wm_256)

        feats = image_encoder(wm_norm)[feat_layer]
        feats = avg_pool(feats)
        B, C, H, W = feats.shape
        feats = feats.permute(0,2,3,1).view(B, H*W, C)
        feats_norm = feats / (torch.norm(feats, p=2, dim=-1, keepdim=True) + EPS)
        cos_sim = torch.matmul(feats_norm, direction_vec.T)  # [B, H*W, 1]

        loss_wm   = torch.mean(1 - cos_sim)
        
        image_denorm = denorm_imagenet(image_enc)
        wm_denorm    = denorm_imagenet(wm_256).clamp(0, 1)
        loss_psnr    = -psnr_loss(wm_denorm, image_denorm)
        total_loss = LAMBDA_CLEAN * loss_wm + LAMBDA_P * loss_psnr

        total_loss.backward()
        optimizer.step()

        if step == 0 or (step+1) % 50 == 0:
            psnr_val = compute_psnr(wm_256.detach(), image_enc)
            print(f"    step {step+1}/{STEPS} | loss={total_loss.item():.4f} "
                  f"| cos_sim={cos_sim.mean().item():.4f} | PSNR={psnr_val:.2f}dB")

    # 최종: pixel space에서 L∞ clamp (locmark 방식)
    with torch.no_grad():
        latent_wm = latent + delta_m
        rec_wm    = pipe.vae.decode(latent_wm).sample
        rec_wm    = (rec_wm + 1) / 2  # VAE_SIZE

        final_delta = torch.clamp(rec_wm - image_vae, -EPSILON, EPSILON)
        final_wm    = torch.clamp(image_vae + final_delta, 0, 1)

        # IMAGE_SIZE로 리사이즈
        final_wm    = F.interpolate(final_wm,    size=(IMAGE_SIZE, IMAGE_SIZE),
                                    mode="bilinear", align_corners=False)
        final_delta = F.interpolate(final_delta, size=(IMAGE_SIZE, IMAGE_SIZE),
                                    mode="bilinear", align_corners=False)
        psnr_final  = compute_psnr(final_wm, image_enc)

    # ── 캐시 저장 ──
    if save_path is not None:
        torch.save({
            'wm_img': final_wm.cpu(),
            'delta':  final_delta.cpu(),
            'psnr':   psnr_final,
        }, save_path)
        print(f"  [embed] saved tensor: {save_path}")

    return final_wm.detach(), final_delta.detach(), psnr_final


# ────────────────────────────────────────────
# DECODE
# ────────────────────────────────────────────

def decode_heatmap(image_01, direction_vec, feat_layer):
    """Returns cos_sim map as numpy [H, W]"""
    image_01 = image_01.to(DEVICE)
    with torch.no_grad():
        img_norm = norm_imagenet(image_01)
        feats = image_encoder(img_norm)[feat_layer]
        feats = avg_pool(feats)
        B, C, H, W = feats.shape
        feats = feats.permute(0,2,3,1).view(B, H*W, C)
        feats_norm = feats / (torch.norm(feats, p=2, dim=-1, keepdim=True) + EPS)
        cos_sim = torch.matmul(feats_norm, direction_vec.T)
        cos_map = cos_sim.view(B, H, W).squeeze(0)
    return cos_map.cpu().numpy()


# ────────────────────────────────────────────
# INPAINT
# ────────────────────────────────────────────

def inpaint(image_01, mask_01, img_idx, prompt=""):
    """mask_01: 1인 부분 inpaint. Returns [1,3,256,256]"""
    image_pil = tensor_to_pil(image_01)
    mask_pil  = tensor_to_pil(mask_01.expand(-1,3,-1,-1))
    image_pil = image_pil.resize((VAE_SIZE, VAE_SIZE), Image.BILINEAR)
    mask_pil  = mask_pil.resize((VAE_SIZE, VAE_SIZE), Image.NEAREST)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + img_idx)
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            height=VAE_SIZE,
            width=VAE_SIZE,
            num_inference_steps=50,
            generator=generator,
        ).images[0]
    result_t = transforms.ToTensor()(result).unsqueeze(0)
    result_t = F.interpolate(result_t, size=(IMAGE_SIZE, IMAGE_SIZE),
                              mode="bilinear", align_corners=False)
    return result_t.to(DEVICE)


# ────────────────────────────────────────────
# VISUALIZATION UTILS
# ────────────────────────────────────────────

def cosmap_to_rgb(cosmap, vmin, vmax):
    norm = (cosmap - vmin) / (vmax - vmin + 1e-8)
    return cm.jet(norm)[:,:,:3]

def visualize_perturbation(delta_t):
    """delta [1,3,H,W] → [0,1] RGB numpy for vis"""
    d = delta_t.squeeze(0).cpu().numpy().transpose(1,2,0)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d


# ────────────────────────────────────────────
# PER-IMAGE FIGURE
# ────────────────────────────────────────────

def save_per_image_figure(img_idx, layer_idx,
                          orig, wm_img, regen_img, delta, psnr_val,
                          hmap_orig, hmap_wm, hmap_regen,
                          fg_mask_np):
    """
    Layout (3×3):
      Row 0: Original        | Watermarked          | After FG Inpaint
      Row 1: Original heatmap| Watermarked heatmap  | Inpainted heatmap
      Row 2: bar charts (FG/BG 영역별 분석)
    """
    feat_h      = hmap_wm.shape[0]
    fg_feat     = get_fg_mask_feat(fg_mask_np, feat_h)
    bg_feat     = ~fg_feat
    fg_feat_map = fg_feat.reshape(feat_h, feat_h).astype(float)

    wm_flat    = hmap_wm.flatten()
    regen_flat = hmap_regen.flatten()

    vmin, vmax = HMAP_VMIN, HMAP_VMAX

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f"Image {img_idx} | Layer {layer_idx} | "
        f"ε={EPSILON:.4f} | PSNR={psnr_val:.2f}dB\n"
        f"FG mask inpaint: FG=새로 생성, BG=유지",
        fontsize=13, fontweight='bold'
    )

    # ── Row 0: 이미지 3종 ──
    orig_np  = orig.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
    wm_np    = wm_img.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()
    regen_np = regen_img.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy()

    fg_boundary = np.array(
        Image.fromarray((fg_mask_np * 255).astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    ) / 255.0

    for ax, img_np, title in zip(
        axes[0],
        [orig_np, wm_np, regen_np],
        ["Original", f"Watermarked (PSNR={psnr_val:.1f}dB)", "After FG Inpaint"]
    ):
        ax.imshow(img_np); ax.set_title(title); ax.axis('off')
        ax.contour(fg_boundary, levels=[0.5], colors='red', linewidths=1.5)

    # ── Row 1: heatmap 3종 (original / watermarked / inpainted) ──
    for ax, hmap, title in zip(
        axes[1],
        [hmap_orig, hmap_wm, hmap_regen],
        ["Heatmap: Original", "Heatmap: Watermarked", "Heatmap: After FG Inpaint"]
    ):
        ax.imshow(cosmap_to_rgb(hmap, vmin, vmax))
        ax.set_title(title); ax.axis('off')
        ax.contour(fg_feat_map, levels=[0.5], colors='white', linewidths=1.5)
        plt.colorbar(
            cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='jet'),
            ax=ax, shrink=0.6, label='cos_sim',
            orientation='horizontal', pad=0.04
        )

    # ── Row 2: 영역별 bar chart ──
    labels_region = ['WM\n(FG)', 'WM\n(BG)', 'After\n(FG)', 'After\n(BG)']
    vals_region   = [
        wm_flat[fg_feat].mean(),    wm_flat[bg_feat].mean(),
        regen_flat[fg_feat].mean(), regen_flat[bg_feat].mean(),
    ]
    bar_colors = ['steelblue', 'lightsteelblue', 'tomato', 'lightsalmon']

    ax4 = axes[2,0]
    bars4 = ax4.bar(labels_region, vals_region, color=bar_colors, alpha=0.85)
    ax4.set_title("Mean cos_sim: WM vs After (FG/BG 영역별)")
    ax4.set_ylabel("Mean cos_sim")
    for bar, val in zip(bars4, vals_region):
        ax4.text(bar.get_x()+bar.get_width()/2, val+0.002, f'{val:.3f}',
                 ha='center', va='bottom', fontsize=9)

    # per-patch cos_sim scatter: FG/BG 색 구분
    ax5 = axes[2,1]
    pidx = np.arange(len(wm_flat))
    ax5.scatter(pidx[fg_feat], wm_flat[fg_feat],    c='steelblue', s=8,  alpha=0.7, label='WM FG')
    ax5.scatter(pidx[bg_feat], wm_flat[bg_feat],    c='lightsteelblue', s=8, alpha=0.7, label='WM BG')
    ax5.scatter(pidx[fg_feat], regen_flat[fg_feat], c='tomato',     s=8,  alpha=0.7, label='After FG')
    ax5.scatter(pidx[bg_feat], regen_flat[bg_feat], c='lightsalmon',s=8,  alpha=0.7, label='After BG')
    ax5.set_title(f"Per-patch cos_sim (n={len(wm_flat)})")
    ax5.set_xlabel("Patch index"); ax5.set_ylabel("cos_sim")
    ax5.legend(fontsize=7, markerscale=2)

    # Δcos_sim bar: FG/BG 영역별 평균 변화량
    ax6 = axes[2,2]
    delta_fg = (regen_flat[fg_feat] - wm_flat[fg_feat]).mean()
    delta_bg = (regen_flat[bg_feat] - wm_flat[bg_feat]).mean()
    bar_c = ['tomato' if delta_fg < 0 else 'steelblue',
             'tomato' if delta_bg < 0 else 'steelblue']
    ax6.bar(['FG region', 'BG region'], [delta_fg, delta_bg], color=bar_c, alpha=0.85)
    ax6.axhline(0, color='black', linewidth=1)
    ax6.set_title("Mean Δcos_sim (after − wm) by region")
    ax6.set_ylabel("Δcos_sim")
    for i, val in enumerate([delta_fg, delta_bg]):
        offset = 0.001 if val >= 0 else -0.003
        ax6.text(i, val+offset, f'{val:+.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"img{img_idx:02d}_layer{layer_idx}.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [fig] saved: {save_path}")


def _retention(hmap_wm, hmap_regen):
    wm_flat    = hmap_wm.flatten()
    regen_flat = hmap_regen.flatten()
    return regen_flat / (wm_flat + EPS), wm_flat, regen_flat


# ────────────────────────────────────────────
# AVERAGE SUMMARY FIGURE  (per layer)
# ────────────────────────────────────────────

def save_average_figure(layer_idx, all_records, fg_mask_np):
    """
    all_records: list of {hmap_orig, hmap_wm, hmap_regen, psnr}
    Row 0: Avg heatmap Original | Watermarked | After FG Inpaint  (colorbar 아래 가로)
    Row 1: FG/BG mean ± std bar | Δcos_sim ± std bar | per-image PSNR
    Row 2: per-image Δcos_sim 추이 | WM vs After scatter | distribution histogram
    """
    # figure용 records는 최대 5개로 제한
    MAX_FIG_IMAGES = 5
    fig_records = all_records[:MAX_FIG_IMAGES]

    avg_hmap_orig  = np.mean([r['hmap_orig']  for r in all_records], axis=0)
    avg_hmap_wm    = np.mean([r['hmap_wm']    for r in all_records], axis=0)
    avg_hmap_regen = np.mean([r['hmap_regen'] for r in all_records], axis=0)
    avg_psnr       = np.mean([r['psnr']       for r in all_records])

    feat_h      = avg_hmap_wm.shape[0]
    fg_feat     = get_fg_mask_feat(fg_mask_np, feat_h)
    bg_feat     = ~fg_feat
    fg_feat_map = fg_feat.reshape(feat_h, feat_h).astype(float)
    n           = len(fig_records)  # figure용 개수 (최대 5)
    n_all       = len(all_records)  # 통계용 전체 개수

    # 통계용: 전체 이미지
    all_orig  = np.concatenate([r['hmap_orig'].flatten()  for r in all_records])
    all_wm    = np.concatenate([r['hmap_wm'].flatten()    for r in all_records])
    all_regen = np.concatenate([r['hmap_regen'].flatten() for r in all_records])
    fg_tile   = np.tile(fg_feat, n_all)
    bg_tile   = np.tile(bg_feat, n_all)

    vmin, vmax = HMAP_VMIN, HMAP_VMAX

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    fig.suptitle(
        f"AVERAGE ({n_all} images) | Layer {layer_idx} | "
        f"ε={EPSILON:.4f} | Avg PSNR={avg_psnr:.2f}dB\n"
        f"FG mask inpaint: FG=새로 생성, BG=유지  (per-image plots: first {n})",
        fontsize=13, fontweight='bold'
    )

    # ── Row 0: 평균 heatmap 3종 ──
    for ax, hmap, title in zip(
        axes[0],
        [avg_hmap_orig, avg_hmap_wm, avg_hmap_regen],
        ["Avg Heatmap: Original", "Avg Heatmap: Watermarked", "Avg Heatmap: After FG Inpaint"]
    ):
        ax.imshow(cosmap_to_rgb(hmap, vmin, vmax))
        ax.set_title(title); ax.axis('off')
        ax.contour(fg_feat_map, levels=[0.5], colors='white', linewidths=1.5)
        plt.colorbar(
            cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='jet'),
            ax=ax, shrink=0.6, label='cos_sim',
            orientation='horizontal', pad=0.04
        )

    # ── Row 1: FG/BG mean ± std ──
    labels2  = ['Orig\n(FG)', 'Orig\n(BG)', 'WM\n(FG)', 'WM\n(BG)', 'After\n(FG)', 'After\n(BG)']
    means2   = [all_orig[fg_tile].mean(), all_orig[bg_tile].mean(),
                all_wm[fg_tile].mean(),   all_wm[bg_tile].mean(),
                all_regen[fg_tile].mean(),all_regen[bg_tile].mean()]
    stds2    = [all_orig[fg_tile].std(),  all_orig[bg_tile].std(),
                all_wm[fg_tile].std(),    all_wm[bg_tile].std(),
                all_regen[fg_tile].std(), all_regen[bg_tile].std()]
    bcolors2 = ['lightgray', 'whitesmoke', 'steelblue', 'lightsteelblue', 'tomato', 'lightsalmon']

    ax_r1 = axes[1,0]
    bars_r1 = ax_r1.bar(labels2, means2, color=bcolors2, alpha=0.9,
                        yerr=stds2, capsize=4, error_kw={'elinewidth':1.5},
                        edgecolor='gray', linewidth=0.5)
    ax_r1.set_title("Mean cos_sim ± std (FG/BG 영역별)")
    ax_r1.set_ylabel("cos_sim")
    ax_r1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    for bar, val in zip(bars_r1, means2):
        ax_r1.text(bar.get_x()+bar.get_width()/2, val+0.002, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=8)

    # Δcos_sim by region (wm→regen)
    delta_fg_vals = all_regen[fg_tile] - all_wm[fg_tile]
    delta_bg_vals = all_regen[bg_tile] - all_wm[bg_tile]
    ax_r2 = axes[1,1]
    bar_c = ['tomato' if delta_fg_vals.mean() < 0 else 'steelblue',
             'tomato' if delta_bg_vals.mean() < 0 else 'steelblue']
    ax_r2.bar(['FG region', 'BG region'],
              [delta_fg_vals.mean(), delta_bg_vals.mean()],
              yerr=[delta_fg_vals.std(), delta_bg_vals.std()],
              color=bar_c, alpha=0.85, capsize=5)
    ax_r2.axhline(0, color='black', linewidth=1)
    ax_r2.set_title("Mean Δcos_sim ± std\n(after inpaint − watermarked)")
    ax_r2.set_ylabel("Δcos_sim")
    for i, val in enumerate([delta_fg_vals.mean(), delta_bg_vals.mean()]):
        offset = 0.001 if val >= 0 else -0.003
        ax_r2.text(i, val+offset, f'{val:+.3f}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

    psnrs = [r['psnr'] for r in fig_records]
    ax_r3 = axes[1,2]
    ax_r3.bar(np.arange(n), psnrs, color='slategray', alpha=0.8)
    ax_r3.axhline(avg_psnr, color='red', linestyle='--', linewidth=1.5,
                  label=f'avg={avg_psnr:.2f}dB')
    ax_r3.set_title(f"PSNR per image (Layer {layer_idx})")
    ax_r3.set_xlabel("Image index"); ax_r3.set_ylabel("PSNR (dB)")
    ax_r3.legend(fontsize=9)

    # ── Row 2: per-image Δcos_sim 추이 / scatter / histogram ──
    per_img_delta_fg = [
        (r['hmap_regen'].flatten()[fg_feat] - r['hmap_wm'].flatten()[fg_feat]).mean()
        for r in fig_records
    ]
    per_img_delta_bg = [
        (r['hmap_regen'].flatten()[bg_feat] - r['hmap_wm'].flatten()[bg_feat]).mean()
        for r in fig_records
    ]
    x = np.arange(n)
    ax_r4 = axes[2,0]
    ax_r4.bar(x - 0.2, per_img_delta_fg, 0.4, label='FG region', color='tomato',        alpha=0.8)
    ax_r4.bar(x + 0.2, per_img_delta_bg, 0.4, label='BG region', color='lightsteelblue', alpha=0.8)
    ax_r4.axhline(0, color='black', linewidth=1)
    ax_r4.set_title("Per-image Δcos_sim (after − wm) by region")
    ax_r4.set_xlabel("Image index"); ax_r4.set_ylabel("Δcos_sim")
    ax_r4.legend(fontsize=9)

    ax_r5 = axes[2,1]
    ax_r5.scatter(all_wm[fg_tile], all_regen[fg_tile], c='tomato',        s=3, alpha=0.3, label='FG patch')
    ax_r5.scatter(all_wm[bg_tile], all_regen[bg_tile], c='lightsteelblue', s=3, alpha=0.3, label='BG patch')
    lim_min = min(all_wm.min(), all_regen.min())
    lim_max = max(all_wm.max(), all_regen.max())
    ax_r5.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, label='y=x')
    ax_r5.set_title("Patch cos_sim: WM vs After FG Inpaint")
    ax_r5.set_xlabel("WM cos_sim"); ax_r5.set_ylabel("After cos_sim")
    ax_r5.legend(fontsize=8, markerscale=3)

    ax_r6 = axes[2,2]
    bins = np.linspace(HMAP_VMIN, HMAP_VMAX, 40)
    ax_r6.hist(all_wm[fg_tile],    bins=bins, alpha=0.5, color='steelblue', label='WM FG',    density=True)
    ax_r6.hist(all_regen[fg_tile], bins=bins, alpha=0.5, color='tomato',    label='After FG', density=True)
    ax_r6.hist(all_wm[bg_tile],    bins=bins, alpha=0.3, color='gray',      label='WM BG',    density=True, linestyle='--', histtype='step', linewidth=1.5)
    ax_r6.hist(all_regen[bg_tile], bins=bins, alpha=0.3, color='black',     label='After BG', density=True, linestyle='--', histtype='step', linewidth=1.5)
    ax_r6.set_title("cos_sim distribution (FG/BG 영역별)")
    ax_r6.set_xlabel("cos_sim"); ax_r6.set_ylabel("Density")
    ax_r6.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"avg_layer{layer_idx}.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[avg fig] saved: {save_path}")


def _perturb_panels(axes_row0, axes_row1, delta_np, title_prefix):
    """공통 패널 렌더링: Row0/Row1에 perturbation 분석 내용 채우기."""
    H, W = delta_np.shape[1], delta_np.shape[2]

    mag_map = np.linalg.norm(delta_np, axis=0)  # [H, W]

    delta_vis = delta_np.transpose(1, 2, 0)
    delta_vis_norm = (delta_vis - delta_vis.min()) / (delta_vis.max() - delta_vis.min() + 1e-8)

    # Row 0-0: delta RGB
    axes_row0[0].imshow(delta_vis_norm)
    axes_row0[0].set_title(f"{title_prefix} Delta RGB (normalized)")
    axes_row0[0].axis('off')

    # Row 0-1: L2 magnitude map
    im1 = axes_row0[1].imshow(mag_map, cmap='hot')
    axes_row0[1].set_title(f"{title_prefix} L2 Magnitude")
    axes_row0[1].axis('off')
    plt.colorbar(im1, ax=axes_row0[1], shrink=0.6, orientation='horizontal', pad=0.04)

    # Row 0-2: R/G/B channel histogram
    for ch, col in zip(range(3), ['red', 'green', 'blue']):
        axes_row0[2].hist(delta_np[ch].flatten(), bins=60, color=col,
                          alpha=0.4, density=True, label=f'Ch{ch}')
    axes_row0[2].axvline(0, color='black', linewidth=1, linestyle='--')
    axes_row0[2].set_title("R/G/B channel distribution")
    axes_row0[2].set_xlabel("Delta value"); axes_row0[2].set_ylabel("Density")
    axes_row0[2].legend(fontsize=9)

    # Row 1-0: magnitude histogram
    axes_row1[0].hist(mag_map.flatten(), bins=60, color='slategray', alpha=0.7, density=True)
    axes_row1[0].set_title("L2 magnitude distribution")
    axes_row1[0].set_xlabel("L2 magnitude"); axes_row1[0].set_ylabel("Density")

    # Row 1-1: FFT log power
    fft_mag = np.fft.fftshift(np.fft.fft2(mag_map))
    fft_log = np.log1p(np.abs(fft_mag))
    im_fft  = axes_row1[1].imshow(fft_log, cmap='inferno')
    axes_row1[1].set_title("FFT of delta magnitude (log power)")
    axes_row1[1].axis('off')
    plt.colorbar(im_fft, ax=axes_row1[1], shrink=0.6, orientation='horizontal', pad=0.04)

    # Row 1-2: Cumulative radial energy
    cy, cx   = H // 2, W // 2
    Y, X     = np.ogrid[:H, :W]
    R_dist   = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    power    = np.abs(fft_mag)**2
    max_r    = min(cy, cx)
    radii    = np.arange(max_r)
    radial_e = np.array([power[R_dist == r].sum() for r in radii])
    cum_e    = np.cumsum(radial_e) / (radial_e.sum() + 1e-8)
    axes_row1[2].plot(radii, cum_e, color='darkorange', linewidth=2)
    axes_row1[2].axhline(0.9, color='gray', linestyle='--', linewidth=1, label='90% energy')
    axes_row1[2].set_title("Cumulative FFT energy (radial)")
    axes_row1[2].set_xlabel("Frequency radius (px)"); axes_row1[2].set_ylabel("Cumulative energy")
    axes_row1[2].legend(fontsize=9); axes_row1[2].grid(alpha=0.3)


def save_perturbation_figure(img_idx, layer_idx, orig, wm_img, regen_img, delta, fg_mask_np):
    """
    Per-image, per-layer perturbation + regen 변화 분석 figure. (4×3 grid)

    Row 0: delta_orig RGB    | delta_orig L2 mag       | R/G/B histogram
    Row 1: mag histogram     | FFT log power            | Cumulative FFT energy
    Row 2: delta_after RGB   | delta_after L2 mag       | survival map (delta_after/delta_orig)
    Row 3: delta_loss RGB    | delta_loss L2 mag        | FG/BG survival mean bar

    delta_orig  = wm   - orig   (심은 perturbation)
    delta_after = regen - orig  (inpaint 후 남은 perturbation)
    survival    = ||delta_after|| / (||delta_orig|| + eps)  pixel-level
    delta_loss  = delta_orig - delta_after  (사라진 perturbation)
    """
    orig_np  = orig.squeeze(0).cpu().numpy()        # [3, H, W]
    delta_np = delta.squeeze(0).cpu().numpy()        # [3, H, W]  delta_orig
    regen_np = regen_img.squeeze(0).cpu().numpy()    # [3, H, W]

    delta_after_np = regen_np - orig_np              # [3, H, W]
    delta_loss_np  = delta_np - delta_after_np       # [3, H, W]

    H, W = delta_np.shape[1], delta_np.shape[2]

    mag_orig  = np.linalg.norm(delta_np,       axis=0)  # [H, W]
    mag_after = np.linalg.norm(delta_after_np, axis=0)
    mag_loss  = np.linalg.norm(delta_loss_np,  axis=0)
    survival  = mag_after / (mag_orig + 1e-8)            # [H, W]  pixel-level

    # fg_mask를 image 해상도로 resize
    fg_img    = np.array(
        Image.fromarray((fg_mask_np * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    ) / 255.0
    fg_bool   = fg_img > 0.5

    def norm_vis(arr3hw):
        v = arr3hw.transpose(1, 2, 0)
        return (v - v.min()) / (v.max() - v.min() + 1e-8)

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    fig.suptitle(
        f"Perturbation & Survival Analysis | Image {img_idx} | Layer {layer_idx} | ε={EPSILON:.4f}",
        fontsize=13, fontweight='bold'
    )

    # ── Row 0 / Row 1: delta_orig (기존 _perturb_panels) ──
    _perturb_panels(axes[0], axes[1], delta_np, title_prefix="Orig")

    # ── Row 2: delta_after ──
    axes[2,0].imshow(norm_vis(delta_after_np))
    axes[2,0].set_title("delta_after RGB  (regen − orig, normalized)")
    axes[2,0].axis('off')
    axes[2,0].contour(fg_img, levels=[0.5], colors='red', linewidths=1.5)

    im_a = axes[2,1].imshow(mag_after, cmap='hot')
    axes[2,1].set_title("delta_after L2 magnitude")
    axes[2,1].axis('off')
    axes[2,1].contour(fg_img, levels=[0.5], colors='cyan', linewidths=1.5)
    plt.colorbar(im_a, ax=axes[2,1], shrink=0.6, orientation='horizontal', pad=0.04)

    # survival map: 0=완전소실(파랑), 1=완전보존(빨강)
    im_s = axes[2,2].imshow(survival, cmap='RdBu_r', vmin=0, vmax=2)
    axes[2,2].set_title("Survival map  ||delta_after|| / ||delta_orig||\n"
                         "Red=preserved, Blue=lost")
    axes[2,2].axis('off')
    axes[2,2].contour(fg_img, levels=[0.5], colors='white', linewidths=1.5)
    plt.colorbar(im_s, ax=axes[2,2], shrink=0.6, orientation='horizontal', pad=0.04)

    # ── Row 3: delta_loss ──
    axes[3,0].imshow(norm_vis(delta_loss_np))
    axes[3,0].set_title("delta_loss RGB  (delta_orig − delta_after, normalized)")
    axes[3,0].axis('off')
    axes[3,0].contour(fg_img, levels=[0.5], colors='red', linewidths=1.5)

    im_l = axes[3,1].imshow(mag_loss, cmap='hot')
    axes[3,1].set_title("delta_loss L2 magnitude  (erased perturbation)")
    axes[3,1].axis('off')
    axes[3,1].contour(fg_img, levels=[0.5], colors='cyan', linewidths=1.5)
    plt.colorbar(im_l, ax=axes[3,1], shrink=0.6, orientation='horizontal', pad=0.04)

    # FG/BG survival mean bar
    fg_surv = survival[fg_bool].mean()
    bg_surv = survival[~fg_bool].mean()
    bars = axes[3,2].bar(['FG region', 'BG region'], [fg_surv, bg_surv],
                          color=['tomato', 'steelblue'], alpha=0.85)
    axes[3,2].axhline(1.0, color='gray', linestyle='--', linewidth=1, label='survival=1')
    axes[3,2].set_title(f"Mean pixel survival by region\n"
                         f"FG={fg_surv:.3f}  BG={bg_surv:.3f}")
    axes[3,2].set_ylabel("Mean survival rate")
    axes[3,2].legend(fontsize=9)
    for bar, val in zip(bars, [fg_surv, bg_surv]):
        axes[3,2].text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    img_out_dir = os.path.join(OUT_DIR, f"L_{layer_idx}", f"img{img_idx:02d}")
    save_path   = os.path.join(img_out_dir, f"perturb_analysis_layer{layer_idx}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[perturb fig] saved: {save_path}")


def save_avg_perturbation_figure(layer_idx, delta_list):
    """
    Layer별 average perturbation 분석 figure. (3×3 grid)

    delta_list: list of [3,H,W] numpy arrays (pixel space, 이미지별 delta)

    Row 0: Avg delta RGB | Avg L2 magnitude | Std of L2 magnitude (이미지 간)
    Row 1: Avg magnitude histogram | FFT of avg magnitude | Cumulative FFT energy
    Row 2: Per-image mean L2 bar | Avg R/G/B channel histogram | 비어있음(axis off)
    """
    deltas  = np.stack(delta_list, axis=0)      # [N, 3, H, W]
    avg_delta = deltas.mean(axis=0)             # [3, H, W]
    std_mag   = np.linalg.norm(deltas, axis=1).std(axis=0)  # [H, W]
    H, W      = avg_delta.shape[1], avg_delta.shape[2]
    n         = len(delta_list)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(
        f"Avg Perturbation Analysis | Layer {layer_idx} | ε={EPSILON:.4f} | N={n} images",
        fontsize=13, fontweight='bold'
    )

    # Row 0/1: 공통 패널 (avg delta 기준)
    _perturb_panels(axes[0], axes[1], avg_delta, title_prefix="Avg")

    # Row 0-2 override: Std of magnitude map
    std_im = axes[0,2].imshow(std_mag, cmap='coolwarm')
    axes[0,2].set_title("Std of L2 magnitude across images")
    axes[0,2].axis('off')
    plt.colorbar(std_im, ax=axes[0,2], shrink=0.6, orientation='horizontal', pad=0.04)

    # Row 2-0: per-image mean L2 bar
    per_img_means = [np.linalg.norm(d, axis=0).mean() for d in delta_list]
    axes[2,0].bar(np.arange(n), per_img_means, color='slategray', alpha=0.8)
    axes[2,0].axhline(np.mean(per_img_means), color='red', linestyle='--',
                      linewidth=1.5, label=f'avg={np.mean(per_img_means):.4f}')
    axes[2,0].set_title("Per-image mean L2 magnitude")
    axes[2,0].set_xlabel("Image index"); axes[2,0].set_ylabel("Mean L2")
    axes[2,0].legend(fontsize=9)

    # Row 2-1: avg R/G/B histogram (avg delta 기준)
    for ch, col in zip(range(3), ['red', 'green', 'blue']):
        axes[2,1].hist(avg_delta[ch].flatten(), bins=60, color=col,
                       alpha=0.4, density=True, label=f'Ch{ch}')
    axes[2,1].axvline(0, color='black', linewidth=1, linestyle='--')
    axes[2,1].set_title("Avg delta R/G/B channel distribution")
    axes[2,1].set_xlabel("Delta value"); axes[2,1].set_ylabel("Density")
    axes[2,1].legend(fontsize=9)

    # Row 2-2: unused
    axes[2,2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"avg_perturb_layer{layer_idx}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[avg perturb fig] saved: {save_path}")


def save_summary_figure(records, fg_mask_np):
    """
    After inpaint 시점에서 FG/BG region의 mean cos_sim + gap line.

    fg_mean = mean_FG(hmap_regen)
    bg_mean = mean_BG(hmap_regen)
    gap     = bg_mean - fg_mean  → 클수록 BG가 FG보다 cos_sim 높음 (공간 구분 명확)

    error band = ±std across images (FG/BG only)
    """
    layer_list = sorted(records.keys())

    fg_means, fg_stds = [], []
    bg_means, bg_stds = [], []

    orig_means, orig_stds = [], []
    wm_means,   wm_stds   = [], []

    for layer_idx in layer_list:
        recs = records[layer_idx]
        if not recs:
            fg_means.append(0);   fg_stds.append(0)
            bg_means.append(0);   bg_stds.append(0)
            orig_means.append(0); orig_stds.append(0)
            wm_means.append(0);   wm_stds.append(0)
            continue

        feat_h  = recs[0]['hmap_wm'].shape[0]
        fg_feat = get_fg_mask_feat(fg_mask_np, feat_h)
        bg_feat = ~fg_feat

        per_img_fg, per_img_bg = [], []
        per_img_orig, per_img_wm = [], []
        for r in recs:
            regen_f = r['hmap_regen'].flatten()
            per_img_fg.append(regen_f[fg_feat].mean())
            per_img_bg.append(regen_f[bg_feat].mean())
            per_img_orig.append(r['hmap_orig'].mean())
            per_img_wm.append(r['hmap_wm'].mean())

        fg_means.append(np.mean(per_img_fg));     fg_stds.append(np.std(per_img_fg))
        bg_means.append(np.mean(per_img_bg));     bg_stds.append(np.std(per_img_bg))
        orig_means.append(np.mean(per_img_orig)); orig_stds.append(np.std(per_img_orig))
        wm_means.append(np.mean(per_img_wm));     wm_stds.append(np.std(per_img_wm))

    fg_means   = np.array(fg_means)
    bg_means   = np.array(bg_means)
    fg_stds    = np.array(fg_stds)
    bg_stds    = np.array(bg_stds)
    orig_means = np.array(orig_means)
    orig_stds  = np.array(orig_stds)
    wm_means   = np.array(wm_means)
    wm_stds    = np.array(wm_stds)
    gap_means  = bg_means - fg_means
    x          = np.arange(len(layer_list))
    labels     = [f"Layer {l}" for l in layer_list]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(
        f"cos_sim by Layer: Clean / Watermarked / After Inpaint (FG & BG)\n"
        f"epsilon={EPSILON:.4f}  |  error band = +-std",
        fontsize=12, fontweight='bold'
    )

    # Clean (orig)
    ax.plot(x, orig_means, color='gray', marker='D', linewidth=1.5,
            markersize=6, linestyle=':', label='Clean (all)', zorder=2)
    ax.fill_between(x, orig_means - orig_stds, orig_means + orig_stds,
                    color='gray', alpha=0.08)

    # Watermarked
    ax.plot(x, wm_means, color='mediumpurple', marker='D', linewidth=1.5,
            markersize=6, linestyle=':', label='Watermarked (all)', zorder=2)
    ax.fill_between(x, wm_means - wm_stds, wm_means + wm_stds,
                    color='mediumpurple', alpha=0.08)
    for i, val in enumerate(wm_means):
        ax.text(i, wm_means[i] + wm_stds[i] + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, color='mediumpurple')

    # Foreground Generation
    ax.plot(x, fg_means, color='tomato', marker='o', linewidth=2,
            markersize=7, label='Foreground Generation', zorder=3)
    ax.fill_between(x, fg_means - fg_stds, fg_means + fg_stds,
                    color='tomato', alpha=0.15)
    for i, val in enumerate(fg_means):
        ax.text(i, fg_means[i] - fg_stds[i] - 0.02, f'{val:.3f}',
                ha='center', va='top', fontsize=9, color='tomato', fontweight='bold')

    # Background Regeneration
    ax.plot(x, bg_means, color='steelblue', marker='s', linewidth=2,
            markersize=7, label='Background Regeneration', zorder=3)
    ax.fill_between(x, bg_means - bg_stds, bg_means + bg_stds,
                    color='steelblue', alpha=0.15)
    for i, val in enumerate(bg_means):
        ax.text(i, bg_means[i] + bg_stds[i] + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, color='steelblue', fontweight='bold')

    # BG - FG gap line
    ax.plot(x, gap_means, color='seagreen', marker='^', linewidth=2,
            markersize=7, linestyle='--', label='BG - FG Gap', zorder=4)
    for i, val in enumerate(gap_means):
        ax.text(i + 0.05, gap_means[i], f'{val:.3f}',
                ha='left', va='center', fontsize=9, color='seagreen', fontweight='bold')

    ax.axhline(0.0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cos_sim")
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "summary_all_layers.png")
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[summary fig] saved: {save_path}")


# ────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────

def save_records(records, path):
    with open(path, 'wb') as f:
        pickle.dump(records, f)
    print(f"[records] saved: {path}")


def load_records(path):
    with open(path, 'rb') as f:
        records = pickle.load(f)
    print(f"[records] loaded: {path}  "
          f"(layers={sorted(records.keys())}, "
          f"n_images={len(next(iter(records.values())))})")
    return records


def save_delta_lists(delta_lists, path):
    with open(path, 'wb') as f:
        pickle.dump(delta_lists, f)
    print(f"[delta_lists] saved: {path}")


def load_delta_lists(path):
    with open(path, 'rb') as f:
        delta_lists = pickle.load(f)
    print(f"[delta_lists] loaded: {path}  "
          f"(layers={sorted(delta_lists.keys())}, "
          f"n_images={len(next(iter(delta_lists.values())))})")
    return delta_lists


def main():
    fg_mask    = make_center_crop_mask(IMAGE_SIZE, CROP_RATIO)
    fg_mask_np = fg_mask.squeeze().numpy()

    # ── avg_only / summary_only: records 로드 후 figure만 생성 ──
    if MODE in ('avg_only', 'summary_only'):
        assert os.path.exists(RECORDS_PATH),             f"records.pkl not found: {RECORDS_PATH}\nRun with MODE='full' first."
        records = load_records(RECORDS_PATH)

        if MODE == 'avg_only':
            assert os.path.exists(DELTA_LISTS_PATH), \
                f"delta_lists.pkl not found: {DELTA_LISTS_PATH}\nRun with MODE='full' first."
            delta_lists = load_delta_lists(DELTA_LISTS_PATH)
            print("\n=== Saving average figures ===")
            for layer_idx in sorted(records.keys()):
                save_average_figure(layer_idx, records[layer_idx], fg_mask_np)
                save_avg_perturbation_figure(layer_idx, delta_lists[layer_idx])

        elif MODE == 'summary_only':
            print("\n=== Saving summary figure ===")
            save_summary_figure(records, fg_mask_np)

        print(f"\n=== Done! Results in: {OUT_DIR} ===")
        return

    # ── full: embed + inpaint + figures + records 저장 ──
    all_imgs = sorted(glob(os.path.join(TRAIN_DIR, "*.jpg")))
    random.seed(SEED)
    selected = random.sample(all_imgs, NUM_IMAGES)
    print(f"Selected {len(selected)} images")
    print(f"L∞ epsilon = {EPSILON} (pixel space, uniform across all layers)")

    dir_vecs = {layer: generate_direction_vector(FEAT_DIMS[layer], layer, seed=SEED) for layer in LAYERS}
    records     = {layer: [] for layer in LAYERS}
    delta_lists = {layer: [] for layer in LAYERS}  # avg perturb용

    for img_idx, img_path in enumerate(selected):
        print(f"\n=== Image {img_idx}/{NUM_IMAGES-1}: {os.path.basename(img_path)} ===")
        orig = load_image(img_path, IMAGE_SIZE).to(DEVICE)

        for layer_idx in LAYERS:
            print(f"\n  --- Layer {layer_idx} (feat_dim={FEAT_DIMS[layer_idx]}, "
                  f"spatial={FEAT_SPATIAL[layer_idx]}²) ---")
            dir_vec = dir_vecs[layer_idx]

            img_out_dir = os.path.join(OUT_DIR, f"L_{layer_idx}", f"img{img_idx:02d}")
            os.makedirs(img_out_dir, exist_ok=True)
            save_image(orig, os.path.join(img_out_dir, "orig.png"))

            wm_tensor_path = os.path.join(img_out_dir, f"wm_tensor_layer{layer_idx}.pt")
            print("  [embed] watermarking...")
            wm_img, delta, psnr_val = embed_watermark(
                orig, dir_vec, layer_idx, save_path=wm_tensor_path
            )
            print(f"  [embed] done. PSNR={psnr_val:.2f}dB | "
                  f"|δ|_∞={delta.abs().max().item():.4f} (EPSILON={EPSILON})")

            hmap_orig = decode_heatmap(orig, dir_vec, layer_idx)
            hmap_wm   = decode_heatmap(wm_img, dir_vec, layer_idx)

            regen_pt_path = os.path.join(img_out_dir, f"regen_tensor_layer{layer_idx}.pt")
            if os.path.exists(regen_pt_path):
                print("  [inpaint] load cached regen tensor...")
                regen_img = torch.load(regen_pt_path, map_location=DEVICE, weights_only=False)
            else:
                print("  [inpaint] FG mask inpaint (FG 새로 생성, BG 유지)...")
                regen_img = inpaint(wm_img, fg_mask.to(DEVICE), img_idx, prompt="")
                torch.save(regen_img.cpu(), regen_pt_path)
                print(f"  [inpaint] saved: {regen_pt_path}")
            hmap_regen = decode_heatmap(regen_img, dir_vec, layer_idx)

            save_image(wm_img,    os.path.join(img_out_dir, f"wm_layer{layer_idx}.png"))
            save_image(regen_img, os.path.join(img_out_dir, f"fg_inpaint_layer{layer_idx}.png"))
            np.save(os.path.join(img_out_dir, f"hmap_orig_layer{layer_idx}.npy"),   hmap_orig)
            np.save(os.path.join(img_out_dir, f"hmap_wm_layer{layer_idx}.npy"),    hmap_wm)
            np.save(os.path.join(img_out_dir, f"hmap_regen_layer{layer_idx}.npy"), hmap_regen)

            save_per_image_figure(img_idx, layer_idx,
                                  orig, wm_img, regen_img, delta, psnr_val,
                                  hmap_orig, hmap_wm, hmap_regen,
                                  fg_mask_np)
            save_perturbation_figure(img_idx, layer_idx, orig, wm_img, regen_img, delta, fg_mask_np)
            delta_lists[layer_idx].append(delta.squeeze(0).cpu().numpy())

            records[layer_idx].append({
                'hmap_orig':  hmap_orig,
                'hmap_wm':    hmap_wm,
                'hmap_regen': hmap_regen,
                'psnr':       psnr_val,
            })

            fg_idx = get_fg_mask_feat(fg_mask_np, hmap_wm.shape[0])
            bg_idx = ~fg_idx
            delta_fg = (hmap_regen.flatten()[fg_idx] - hmap_wm.flatten()[fg_idx]).mean()
            delta_bg = (hmap_regen.flatten()[bg_idx] - hmap_wm.flatten()[bg_idx]).mean()
            print(f"  [result] WM mean={hmap_wm.mean():.4f} | After mean={hmap_regen.mean():.4f}")
            print(f"           FG Δcos_sim={delta_fg:+.4f} | BG Δcos_sim={delta_bg:+.4f}")

    # records 저장
    save_records(records, RECORDS_PATH)
    save_delta_lists(delta_lists, DELTA_LISTS_PATH)

    # Average figures
    print("\n=== Saving average figures ===")
    for layer_idx in LAYERS:
        save_average_figure(layer_idx, records[layer_idx], fg_mask_np)
        save_avg_perturbation_figure(layer_idx, delta_lists[layer_idx])

    # Summary figure
    print("\n=== Saving summary figure ===")
    save_summary_figure(records, fg_mask_np)

    print(f"\n=== Done! Results in: {OUT_DIR} ===")


if __name__ == "__main__":
    main()