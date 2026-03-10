import matplotlib.pyplot as plt
import os
from PIL import Image

# 1. 이미지 경로 설정 (20개를 순서대로 입력하세요)
# 행 방향 우선: (WAM 5개) -> (OmniGuard 5개) -> (StableGuard 5개) -> (APT 5개)
image_paths = [
    # wam
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/wam/fig1_diff_p.png",
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/wam/fig1_diff_fr.png",
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/wam/fig1_diff_sp.png",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliceless/0221.png",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliced/0221.png",
    # omniguard
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/omniguard/fig1_diff_p.png",
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/omniguard/fig1_diff_fr.png",
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/omniguard/fig1_diff_sp.png",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliceless/0221.png",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliced/0221.png",
    # stableguard
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/stableguard/fig1_diff_p.png",
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/stableguard/fig1_diff_fr.png",
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/stableguard/fig1_diff_sp.png",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliceless/0221.png",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliced/0221.png",
    # ours
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/ours/fig1_diff_p.png",
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/ours/fig1_diff_fr.png",
    "/mnt/nas5/suhyeon/projects/locmark_fig1/other_methods/ours/fig1_diff_sp.png",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliceless_refiner/0221.png",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliced_refiner/0221.png",
]

row_labels = ["WAM", "OmniGuard", "StableGuard", "APT*"]
col_labels = ["Signal", "FR", "SP", "Mask (FR)", "Mask (SP)"]

rows, cols = 4, 5
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

# 간격 설정 (원본 코드 스타일 유지)
plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.08, right=0.98, top=0.98, bottom=0.05)

for r in range(rows):
    for c in range(cols):
        ax = axes[r, c]
        img_idx = r * cols + c
        
        try:
            img_path = image_paths[img_idx]
            img = Image.open(img_path)
            
            # 마스크나 특정 이미지 특성에 따른 cmap 설정
            ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
        except Exception as e:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
        
        # 테두리 및 축 설정 (원본 스타일)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor('black')
            s.set_linewidth(0.8)

        # 행 라벨 (왼쪽)
        if c == 0:
            ax.set_ylabel(row_labels[r], rotation=90, fontsize=24, labelpad=6)
        
        # 열 라벨 (하단)
        if r == rows - 1:
            ax.text(0.5, -0.06, col_labels[c], transform=ax.transAxes, 
                    ha='center', va='top', fontsize=24)

# 결과 저장
plt.savefig('fig_fig1_all.pdf', dpi=300, bbox_inches='tight')
# plt.show()