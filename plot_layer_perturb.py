import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CONFIG ────────────────────────────────────────────────────────
OUT_DIR          = "/mnt/nas5/suhyeon/projects/locmark_motiv_fig/p0.02_seed19"
DELTA_LISTS_PATH = os.path.join(OUT_DIR, "delta_lists.pkl")
LAYERS           = [0, 1, 2, 3]
# ─────────────────────────────────────────────────────────────────
# def main():
#     assert os.path.exists(DELTA_LISTS_PATH), f"delta_lists.pkl not found: {DELTA_LISTS_PATH}"
#     with open(DELTA_LISTS_PATH, 'rb') as f:
#         delta_lists = pickle.load(f)

#     fig = plt.figure(figsize=(16, 4))
    
#     # 1. 전체 레이아웃을 1x4로 나눔 (그룹 간 간격 wspace=0.4)
#     outer_gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.03)

#     for i, layer_idx in enumerate(LAYERS):
#         # 2. 각 그룹 내부를 1x2로 나눔 (그룹 내 간격 wspace=0.05)
#         inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[i], wspace=0.02)
        
#         ax_rgb = fig.add_subplot(inner_gs[0])
#         ax_fft = fig.add_subplot(inner_gs[1])

#         delta_list = delta_lists.get(layer_idx, [])
#         if not delta_list: continue

#         deltas    = np.stack(delta_list, axis=0)
#         avg_delta = deltas.mean(axis=0)
#         mag_map   = np.linalg.norm(avg_delta, axis=0)
#         H, W      = avg_delta.shape[1], avg_delta.shape[2]

#         # ── RGB Plot ──
#         delta_vis = avg_delta.transpose(1, 2, 0)
#         delta_vis_norm = (delta_vis - delta_vis.min()) / (delta_vis.max() - delta_vis.min() + 1e-8)
        
#         fft_size = mag_map.shape[0]
#         cy, cx = H // 2, W // 2
#         half = fft_size // 2
#         delta_crop = delta_vis_norm[cy-half:cy+half, cx-half:cx+half]
        
#         ax_rgb.imshow(delta_crop)
#         ax_rgb.axis('off')

#         # 타이틀을 두 이미지의 중앙에 배치 (x=1.025는 두 서브플롯 사이 정중앙 부근)
#         ax_rgb.set_title(f"Layer {layer_idx}", fontsize=20, pad=10, x=1.025)

#         # ── Frequency (FFT) Plot ──
#         fft_mag = np.fft.fftshift(np.fft.fft2(mag_map))
#         fft_log = np.log1p(np.abs(fft_mag))
#         im = ax_fft.imshow(fft_log, cmap='inferno')
#         ax_fft.axis('off')

#     # 여백 및 컬러바 설정
#     plt.subplots_adjust(left=0.02, right=0.91, top=0.80, bottom=0.02)
    
#     # 마지막 FFT 이미지의 위치를 기준으로 컬러바 정렬
#     pos = ax_fft.get_position()
#     cbar_ax = fig.add_axes([0.92, pos.y0, 0.01, pos.height])
#     cbar_ax.tick_params(labelsize=14)

#     sm = plt.cm.ScalarMappable(cmap='inferno')
#     sm.set_array([])
#     fig.colorbar(sm, cax=cbar_ax)

#     save_path = "fig_layer_perturb.png"
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Saved: {save_path}")

# if __name__ == "__main__":
#     main()

# 2*4
# """
# avg_perturb_layer{0..3}.png 에서 그렸던
#   - Avg Delta RGB
#   - FFT of delta magnitude (log power)
# 를 2×4 subfigure 한 장으로 합쳐서 저장하는 스크립트.

# delta_lists.pkl을 로드해서 직접 계산함.
# Layout:
#   Row 0: Avg Delta RGB     — Layer 0, 1, 2, 3
#   Row 1: FFT log power     — Layer 0, 1, 2, 3
# """

# import os
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# # ── CONFIG ────────────────────────────────────────────────────────
# OUT_DIR          = "/mnt/nas5/suhyeon/projects/locmark_motiv_fig/p0.02_seed19"
# DELTA_LISTS_PATH = os.path.join(OUT_DIR, "delta_lists.pkl")
# LAYERS           = [0, 1, 2, 3]
# EPSILON          = 32 / 255
# # ─────────────────────────────────────────────────────────────────


def main():
    assert os.path.exists(DELTA_LISTS_PATH), f"delta_lists.pkl not found: {DELTA_LISTS_PATH}"
    with open(DELTA_LISTS_PATH, 'rb') as f:
        delta_lists = pickle.load(f)
    print(f"Loaded delta_lists: layers={sorted(delta_lists.keys())}, "
          f"n_images={len(next(iter(delta_lists.values())))}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0})

    # ── Pass 1: 모든 레이어의 데이터 계산 (global vmin/vmax 산출용) ──
    layer_data = {}
    for col, layer_idx in enumerate(LAYERS):
        delta_list = delta_lists.get(layer_idx, [])
        if not delta_list:
            continue
        deltas    = np.stack(delta_list, axis=0)   # [N, 3, H, W]
        avg_delta = deltas.mean(axis=0)             # [3, H, W]
        H, W      = avg_delta.shape[1], avg_delta.shape[2]

        mag_map  = np.linalg.norm(avg_delta, axis=0)  # [H, W]
        fft_size = mag_map.shape[0]

        delta_vis      = avg_delta.transpose(1, 2, 0)
        delta_vis_norm = (delta_vis - delta_vis.min()) / (delta_vis.max() - delta_vis.min() + 1e-8)
        cy, cx     = H // 2, W // 2
        half       = fft_size // 2
        delta_crop = delta_vis_norm[cy-half:cy+half, cx-half:cx+half]

        fft_mag = np.fft.fftshift(np.fft.fft2(mag_map))
        fft_abs = np.abs(fft_mag)
        fcy, fcx = fft_abs.shape[0] // 2, fft_abs.shape[1] // 2
        fft_abs[fcy-2:fcy+2, fcx-2:fcx+2] = 0  # DC spike 제거
        fft_log = np.log1p(fft_abs)

        layer_data[col] = (layer_idx, delta_crop, fft_log)

    # 모든 레이어에 공통 적용할 실제 값 범위
    all_fft = [d[2] for d in layer_data.values()]
    vmin = min(f.min() for f in all_fft)
    vmax = max(f.max() for f in all_fft)

    # ── Pass 2: 플롯 ──
    im = None
    for col, layer_idx in enumerate(LAYERS):
        if col not in layer_data:
            for row in range(2):
                axes[row, col].set_title(f"Layer {layer_idx} (no data)")
                axes[row, col].axis('off')
            continue
        layer_idx, delta_crop, fft_log = layer_data[col]

        axes[0, col].imshow(delta_crop)
        axes[0, col].set_title(f"Layer {layer_idx}", fontsize=30, pad=10)
        axes[0, col].axis('off')

        im = axes[1, col].imshow(fft_log, cmap='magma', vmin=vmin, vmax=vmax)
        axes[1, col].axis('off')

    # row labels: axis('off')이므로 annotate로 왼쪽 바깥에 부착
    for row, label in zip([0, 1], ["RGB", "Frequency"]):
        axes[row, 0].annotate(
            label,
            xy=(-0.06, 0.5), xycoords='axes fraction',
            fontsize=30,
            ha='right', va='center',
            rotation=90,
            annotation_clip=False
        )

    # FFT colorbar: 실제 im의 norm(vmin~vmax) 기준
    cbar_ax = fig.add_axes([0.91, 0.06, 0.015, 0.35])  # [left, bottom, width, height]
    cbar_ax.tick_params(labelsize=15)
    fig.colorbar(im, cax=cbar_ax)

    plt.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.04, wspace=0.06, hspace=0.05)
    # save_path = os.path.join(OUT_DIR, "avg_perturb_combined.png")
    save_path = "fig_layer_perturb.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()