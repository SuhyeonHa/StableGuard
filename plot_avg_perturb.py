"""
avg_perturb_layer{0..3}.png 에서 그렸던
  - Avg Delta RGB
  - FFT of delta magnitude (log power)
를 2×4 subfigure 한 장으로 합쳐서 저장하는 스크립트.

delta_lists.pkl을 로드해서 직접 계산함.
Layout:
  Row 0: Avg Delta RGB     — Layer 0, 1, 2, 3
  Row 1: FFT log power     — Layer 0, 1, 2, 3
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────
OUT_DIR          = "/mnt/nas5/suhyeon/projects/locmark_motiv_fig/p0.02_seed19"
DELTA_LISTS_PATH = os.path.join(OUT_DIR, "delta_lists.pkl")
LAYERS           = [0, 1, 2, 3]
EPSILON          = 32 / 255
# ─────────────────────────────────────────────────────────────────


def main():
    assert os.path.exists(DELTA_LISTS_PATH), f"delta_lists.pkl not found: {DELTA_LISTS_PATH}"
    with open(DELTA_LISTS_PATH, 'rb') as f:
        delta_lists = pickle.load(f)
    print(f"Loaded delta_lists: layers={sorted(delta_lists.keys())}, "
          f"n_images={len(next(iter(delta_lists.values())))}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15})
    fig.suptitle(
        f"Avg Perturbation by Layer  |  ε={EPSILON:.4f}",
        fontsize=14, fontweight='bold'
    )

    for col, layer_idx in enumerate(LAYERS):
        delta_list = delta_lists.get(layer_idx, [])
        if not delta_list:
            for row in range(2):
                axes[row, col].set_title(f"Layer {layer_idx} (no data)")
                axes[row, col].axis('off')
            continue

        deltas    = np.stack(delta_list, axis=0)   # [N, 3, H, W]
        avg_delta = deltas.mean(axis=0)             # [3, H, W]
        H, W      = avg_delta.shape[1], avg_delta.shape[2]

        # ── Row 0: Avg Delta RGB (FFT 크기와 동일하게 center crop) ──
        mag_map = np.linalg.norm(avg_delta, axis=0)  # [H, W]
        fft_size = mag_map.shape[0]  # FFT는 H×H square

        delta_vis = avg_delta.transpose(1, 2, 0)   # [H, W, 3]
        delta_vis_norm = (delta_vis - delta_vis.min()) / (delta_vis.max() - delta_vis.min() + 1e-8)
        # center crop to fft_size × fft_size
        cy, cx = H // 2, W // 2
        half = fft_size // 2
        delta_crop = delta_vis_norm[cy-half:cy+half, cx-half:cx+half]
        axes[0, col].imshow(delta_crop)
        axes[0, col].set_title(f"Layer {layer_idx}", fontsize=12)
        axes[0, col].axis('off')

        # ── Row 1: FFT of delta magnitude ──
        fft_mag = np.fft.fftshift(np.fft.fft2(mag_map))
        fft_log = np.log1p(np.abs(fft_mag))
        im = axes[1, col].imshow(fft_log, cmap='inferno')
        axes[1, col].set_title(f"Layer {layer_idx}", fontsize=12)
        axes[1, col].axis('off')
        # colorbar는 아래에서 일괄 처리

    # row labels
    for row, label in enumerate(["Avg Delta RGB", "FFT log power"]):
        axes[row, 0].set_ylabel(label, fontsize=11)

    # FFT colorbar: Row 1 전체 axes에 공유
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.35])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='inferno')
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.04, wspace=0.08, hspace=0.15)
    save_path = os.path.join(OUT_DIR, "avg_perturb_combined.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()