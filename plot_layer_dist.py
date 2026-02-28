"""
avg_layer{0..3}.png 에서 그렸던 cos_sim distribution을
2×2 subfigure 한 장으로 합쳐서 저장하는 스크립트.

records.pkl을 로드해서 직접 데이터를 재계산함.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ── CONFIG (analysis_layers.py와 동일하게 맞춰줄 것) ──────────────
OUT_DIR    = "/mnt/nas5/suhyeon/projects/locmark_motiv_fig/p0.02_seed19"
RECORDS_PATH = os.path.join(OUT_DIR, "records.pkl")
LAYERS     = [0, 1, 2, 3]
IMAGE_SIZE = 256
CROP_RATIO = 0.5
HMAP_VMIN  = -0.3
HMAP_VMAX  =  0.5
INPAINT_MODE = 'fg_mask'   # 'fg_mask' | 'zero_mask'
# ─────────────────────────────────────────────────────────────────


def make_center_crop_mask(size, ratio=0.5):
    import torch
    mask = np.zeros((size, size), dtype=np.float32)
    h = w = int(size * ratio)
    sy = (size - h) // 2
    sx = (size - w) // 2
    mask[sy:sy+h, sx:sx+w] = 1.0
    return mask


def is_zero_mask(fg_mask_np):
    return fg_mask_np.sum() == 0


def get_fg_mask_feat(fg_mask_np, feat_h):
    resized = np.array(
        Image.fromarray((fg_mask_np * 255).astype(np.uint8)).resize(
            (feat_h, feat_h), Image.NEAREST)
    ).flatten() > 127
    if resized.sum() == 0:
        return np.ones(feat_h * feat_h, dtype=bool)
    return resized


def main():
    if INPAINT_MODE == 'zero_mask':
        fg_mask_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    else:
        fg_mask_np = make_center_crop_mask(IMAGE_SIZE, CROP_RATIO)

    zero_mode = is_zero_mask(fg_mask_np)

    assert os.path.exists(RECORDS_PATH), f"records.pkl not found: {RECORDS_PATH}"
    with open(RECORDS_PATH, 'rb') as f:
        records = pickle.load(f)
    
    bins = np.linspace(HMAP_VMIN, HMAP_VMAX, 40)

    # 하단 범례 공간 확보를 위해 figsize와 subplots_adjust 조정
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    for idx, layer_idx in enumerate(LAYERS):
        ax = axes[idx // 2, idx % 2]
        recs = records.get(layer_idx, [])

        if not recs:
            ax.set_title(f"Layer {layer_idx} (no data)")
            ax.axis('off')
            continue

        feat_h  = recs[0]['hmap_wm'].shape[0]
        fg_feat = get_fg_mask_feat(fg_mask_np, feat_h)
        bg_feat = ~fg_feat

        all_orig  = np.concatenate([r['hmap_orig'].flatten()  for r in recs])
        all_wm    = np.concatenate([r['hmap_wm'].flatten()    for r in recs])
        all_regen = np.concatenate([r['hmap_regen'].flatten() for r in recs])
        n_img = len(recs)
        fg_tile = np.tile(fg_feat, n_img)
        bg_tile = np.tile(bg_feat, n_img)

        if zero_mode:
            ax.hist(all_orig,  bins=bins, color='lightgray', label='Clean', 
                    density=True, histtype='step', linewidth=2.0, linestyle=':')
            ax.hist(all_wm,    bins=bins, color='darkgray', label='Perturbed', 
                    density=True, histtype='step', linewidth=1.5, linestyle='--')
            ax.hist(all_regen, bins=bins, color='tomato', label='After', 
                    density=True, alpha=0.6)
        else:
            ax.hist(all_orig, bins=bins, color='darkgray', label='Clean', 
                    density=True, histtype='step', linewidth=2.0, linestyle=':')
            ax.hist(all_wm, bins=bins, color='darkgray', label='Perturbed', 
                    density=True, histtype='step', linewidth=1.5, linestyle='--')
            ax.hist(all_regen[fg_tile], bins=bins, color='tomato', label='Foreground', 
                    density=True, alpha=0.55)
            ax.hist(all_regen[bg_tile], bins=bins, color='steelblue', label='Background', 
                    density=True, alpha=0.55)

        ax.set_title(f"Layer {layer_idx}", fontsize=28)
        ax.set_xlabel("Cosine Similarity", fontsize=24)
        ax.set_ylabel("Density", fontsize=28)
        ax.set_xlim(HMAP_VMIN, HMAP_VMAX)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=20)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=28, bbox_to_anchor=(0.5, 0.02))

    # 범례와 그래프가 겹치지 않도록 여백 조정
    plt.tight_layout(rect=[0, 0.16, 1, 0.93])
    plt.subplots_adjust(wspace=0.25, hspace=0.4)
    
    save_path = "fig_layer_dist.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()