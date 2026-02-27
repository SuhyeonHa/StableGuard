"""
Motivation figure:
  - 왼쪽: Original image (크게) + 오른쪽 하단 inset으로 center crop mask
  - 오른쪽: 3×4 subfigures
      Row 0: Watermarked          — Layer 0, 1, 2, 3
      Row 1: After FG Inpaint     — Layer 0, 1, 2, 3
      Row 2: Heatmap: After Inpaint — Layer 0, 1, 2, 3

records.pkl + 이미지 파일에서 직접 로드.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────
OUT_DIR      = "/mnt/nas5/suhyeon/projects/locmark_motiv_fig/p0.03_seed19"
RECORDS_PATH = os.path.join(OUT_DIR, "records.pkl")
LAYERS       = [0, 1, 2, 3]
IMG_IDX      = 19        # 시각화할 이미지 인덱스
IMAGE_SIZE   = 256
CROP_RATIO   = 0.5
HMAP_VMIN    = -0.3
HMAP_VMAX    =  0.5
EPSILON      = 32 / 255
# ─────────────────────────────────────────────────────────────────


def cosmap_to_rgb(cosmap, vmin=HMAP_VMIN, vmax=HMAP_VMAX):
    norm = (cosmap - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)
    return cm.jet(norm)[:, :, :3]


def make_center_crop_mask(size, ratio=0.5):
    mask = np.zeros((size, size), dtype=np.float32)
    h = w = int(size * ratio)
    sy = (size - h) // 2
    sx = (size - w) // 2
    mask[sy:sy+h, sx:sx+w] = 1.0
    return mask


def load_img(path):
    return np.array(Image.open(path).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)) / 255.0


def main():
    assert os.path.exists(RECORDS_PATH), f"records.pkl not found: {RECORDS_PATH}"
    with open(RECORDS_PATH, 'rb') as f:
        records = pickle.load(f)
    print(f"Loaded records: layers={sorted(records.keys())}")

    fg_mask_np = make_center_crop_mask(IMAGE_SIZE, CROP_RATIO)

    # ── 이미지 & heatmap 로드 ──
    # orig는 Layer 0 기준 (모든 layer 공통)
    orig_path = os.path.join(OUT_DIR, f"L_0", f"img{IMG_IDX:02d}", "orig.png")
    orig_np   = load_img(orig_path)

    layer_data = {}
    for layer_idx in LAYERS:
        img_dir  = os.path.join(OUT_DIR, f"L_{layer_idx}", f"img{IMG_IDX:02d}")
        wm_path    = os.path.join(img_dir, f"wm_layer{layer_idx}.png")
        regen_path = os.path.join(img_dir, f"fg_inpaint_layer{layer_idx}.png")
        hmap_path  = os.path.join(img_dir, f"hmap_regen_layer{layer_idx}.npy")

        wm_np    = load_img(wm_path)    if os.path.exists(wm_path)    else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        regen_np = load_img(regen_path) if os.path.exists(regen_path) else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
        hmap     = np.load(hmap_path)   if os.path.exists(hmap_path)  else np.zeros((8, 8))

        # records에서 PSNR 가져오기
        recs     = records.get(layer_idx, [])
        psnr_val = recs[IMG_IDX]['psnr'] if IMG_IDX < len(recs) else float('nan')

        layer_data[layer_idx] = dict(wm=wm_np, regen=regen_np, hmap=hmap, psnr=psnr_val)

    # ── Figure layout ──
    # 전체를 1×2로 나누고 왼쪽=Original, 오른쪽=3×4 grid
    fig = plt.figure(figsize=(18, 10))
    gs_outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 3],
                                  left=0.04, right=0.97, top=0.93, bottom=0.04,
                                  wspace=0.06)

    # ── 왼쪽: Original image ──
    ax_orig = fig.add_subplot(gs_outer[0, 0])
    ax_orig.imshow(orig_np)
    ax_orig.set_title("Inputs", fontsize=13, fontweight='bold', pad=8)
    ax_orig.axis('off')

    # inset: 오른쪽 하단에 딱 붙임 (axes 좌표 기준)
    inset_ax = ax_orig.inset_axes([0.58, 0.0, 0.42, 0.42])  # [x, y, w, h]
    inset_ax.imshow(fg_mask_np, cmap='gray', vmin=0, vmax=1)
    inset_ax.axis('off')
    for spine in inset_ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1.5)
        spine.set_visible(True)

    # ── 오른쪽: 3×4 subfigures ──
    gs_right = gridspec.GridSpecFromSubplotSpec(
        3, 4, subplot_spec=gs_outer[0, 1],
        hspace=0.12, wspace=0.06
    )

    row_labels = ["Perturbed", "Inpainted", "Similarity Map"]
    # row별 마지막 axes 저장 (오른쪽에 label 붙이기용)
    last_axes = {}

    for col, layer_idx in enumerate(LAYERS):
        d = layer_data[layer_idx]

        # Row 0: Watermarked (Perturbed)
        ax = fig.add_subplot(gs_right[0, col])
        ax.imshow(d['wm'])
        ax.axis('off')
        title = f"Layer {layer_idx}" if np.isnan(d['psnr']) else f"Layer {layer_idx}  PSNR={d['psnr']:.1f}dB"
        ax.set_title(title, fontsize=12, fontweight='bold')
        last_axes[0] = ax

        # Row 1: After FG Inpaint (Inpainted)
        ax = fig.add_subplot(gs_right[1, col])
        ax.imshow(d['regen'])
        ax.axis('off')
        last_axes[1] = ax

        # Row 2: Heatmap after inpaint (Similarity Map)
        ax = fig.add_subplot(gs_right[2, col])
        hmap_rgb = cosmap_to_rgb(d['hmap'])
        ax.imshow(hmap_rgb, interpolation='nearest')
        ax.axis('off')
        last_axes[2] = ax

    # 오른쪽 row label: 시계 90도 방향 (rotation=-90)
    for row, label in enumerate(row_labels):
        last_axes[row].annotate(
            label, fontsize=11, fontweight='bold',
            xy=(1.04, 0.5), xycoords='axes fraction',
            ha='left', va='center', rotation=-90
        )

    # colorbar (heatmap 전용, figure 우측)
    cbar_ax = fig.add_axes([0.975, 0.04, 0.012, 0.27])
    sm = cm.ScalarMappable(norm=plt.Normalize(vmin=HMAP_VMIN, vmax=HMAP_VMAX), cmap='jet')
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='cos_sim')



    save_path = os.path.join(OUT_DIR, f"motiv_img{IMG_IDX:02d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()