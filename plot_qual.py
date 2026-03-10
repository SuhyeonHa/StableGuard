import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib.lines import Line2D

# #################################################
# # 1. Fully-Generated Images
# #################################################

# # 1. Row 설정: 4행으로 변경 [tamper_model, image_name]
# row_configs = [
#     ["ldm", "0023.png"], # 23,32, 18, 66
#     ["brushnet", "0032.png"],
#     ["control", "0046.png"],
#     ["hdpainter", "0079.png"]
# ]

# # 2. 디렉토리 설정: 9열에 맞춰 리스트 조정 (앞의 9개 경로 사용 예시)
# directories = [
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliceless_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_bin_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliceless_refiner",
# ]

# row_labels = ["SD-Painter", "BrushNet", "ControlNet", "HD-Painter"]
# col_labels = ["Clean", "Perturbed", "Tampered", "GT", "WAM", "OmniGuard", "StableGuard", "APT", "APT*"]


# rows, cols = 4, 9
# fig, axes = plt.subplots(rows, cols, figsize=(20, 9))

# # 간격 설정 (wspace, hspace = 0.03)
# plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)

# # 이미지 로드 및 배치
# for r, (tamper_model, name) in enumerate(row_configs):
#     for c, base_dir in enumerate(directories):
#         ax = axes[r, c]
        
#         # 경로 생성 및 ldm 치환
#         target_dir = base_dir.replace('ldm', tamper_model)
#         img_path = os.path.join(target_dir, name)
        
#         try:
#             img = Image.open(img_path)
            
#             # 리사이즈 없이 원본 표시
#             ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
#         except:
#             ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
        
#         # 테두리 및 축 설정
#         ax.set_xticks([]); ax.set_yticks([])
#         for s in ax.spines.values():
#             s.set_visible(True)
#             s.set_edgecolor('black')
#             s.set_linewidth(0.8)

#         if c == 0:
#             ax.set_ylabel(row_labels[r], rotation=90, fontsize=24, labelpad=10)
        
#         if r == rows - 1:
#             ax.text(0.5, -0.05, col_labels[c], transform=ax.transAxes, 
#                     ha='center', va='top', fontsize=24)

# # 결과 저장
# plt.savefig('fig_qual_fr.pdf', dpi=300, bbox_inches='tight')

#################################################
# 2. Spliced Images
#################################################

# # 1. Row 설정: 4행으로 변경 [tamper_model, image_name]
# row_configs = [
#     ["ldm", "0071.png"], # 23,32, 18, 66
#     ["brushnet", "0038.png"],
#     ["control", "0034.png"],
#     ["hdpainter", "0073.png"]
# ]

# # 2. 디렉토리 설정: 9열에 맞춰 리스트 조정 (앞의 9개 경로 사용 예시)
# directories = [
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliced_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_bin_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliced_refiner",
# ]

# row_labels = ["SD-Painter", "BrushNet", "ControlNet", "HD-Painter"]
# col_labels = ["Clean", "Perturbed", "Tampered", "GT", "WAM", "OmniGuard", "StableGuard", "APT", "APT*"]


# rows, cols = 4, 9
# fig, axes = plt.subplots(rows, cols, figsize=(20, 9))

# # 간격 설정 (wspace, hspace = 0.03)
# plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)

# # 이미지 로드 및 배치
# for r, (tamper_model, name) in enumerate(row_configs):
#     for c, base_dir in enumerate(directories):
#         ax = axes[r, c]
        
#         # 경로 생성 및 ldm 치환
#         target_dir = base_dir.replace('ldm', tamper_model)
#         img_path = os.path.join(target_dir, name)
        
#         try:
#             img = Image.open(img_path)
            
#             # 리사이즈 없이 원본 표시
#             ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
#         except:
#             ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
        
#         # 테두리 및 축 설정
#         ax.set_xticks([]); ax.set_yticks([])
#         for s in ax.spines.values():
#             s.set_visible(True)
#             s.set_edgecolor('black')
#             s.set_linewidth(0.8)

#         if c == 0:
#             ax.set_ylabel(row_labels[r], rotation=90, fontsize=24, labelpad=10)
        
#         if r == rows - 1:
#             ax.text(0.5, -0.05, col_labels[c], transform=ax.transAxes, 
#                     ha='center', va='top', fontsize=24)

# # 결과 저장
# plt.savefig('fig_qual_sp.pdf', dpi=300, bbox_inches='tight')



# 5x 15
# row_configs = [
#     ["ldm", "0023.png"],
#     ["ldm", "0032.png"],
#     ["ldm", "0018.png"],
#     ["ldm", "0066.png"],
#     ["ldm", "0079.png"]
# ]

# # 2. 디렉토리 템플릿 (경로 내 'ldm' 부분이 tamper_model로 치환됨)
# directories = [
#     "/mnt/nas5/suhyeon/datasets/valAGE-Set/",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliced_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliceless_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/gt",
#     # --- Spliced Predictions ---
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_bin_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliced_refiner",
#     # --- Spliceless Predictions ---
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_bin_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliceless_refiner",
# ]

# rows, cols = 5, 15
# fig, axes = plt.subplots(rows, cols, figsize=(25, 8.5))

# # 기본 간격 설정 (wspace, hspace = 0.03)
# plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)

# # 이미지 로드 및 배치
# for r, (tamper_model, name) in enumerate(row_configs):
#     for c, base_dir in enumerate(directories):
#         ax = axes[r, c]
        
#         # tamper_model에 따라 경로 동적 변경 ('ldm' -> 지정 모델명)
#         target_dir = base_dir.replace('ldm', tamper_model)
#         img_path = os.path.join(target_dir, name)
        
#         try:
#             img = Image.open(img_path)
#             # 리사이즈 로직: 원본 이미지는 Bilinear, 마스크(GT 포함)는 Nearest
#             is_raw_image = "valAGE-Set/" in target_dir and "Mask" not in target_dir and "gt" not in target_dir
#             mode = Image.BILINEAR if is_raw_image else Image.NEAREST
#             img = img.resize((256, 256), resample=mode)
            
#             ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
#         except:
#             ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=8)
        
#         # 테두리 및 축 설정
#         ax.set_xticks([]); ax.set_yticks([])
#         for s in ax.spines.values():
#             s.set_visible(True); s.set_edgecolor('black'); s.set_linewidth(0.8)

# # 레이아웃 확정 후 미세 간격 조정 (세로선 공간 확보)
# fig.canvas.draw()
# # 이미지 폭의 약 2%만큼만 추가로 벌려 0.03 -> 0.05 효과 유도
# shift = 0.02 * axes[0, 0].get_position().width 

# for r in range(rows):
#     for c in range(5, 10): # 6~10열 이동
#         p = axes[r, c].get_position()
#         axes[r, c].set_position([p.x0 + shift, p.y0, p.width, p.height])
#     for c in range(10, 15): # 11~15열 추가 이동
#         p = axes[r, c].get_position()
#         axes[r, c].set_position([p.x0 + shift * 2, p.y0, p.width, p.height])

# # 세로선 그리기 (간격이 벌어진 정중앙에 배치)
# def draw_v_line(c_left, c_right):
#     p1 = axes[0, c_left].get_position()
#     p2 = axes[0, c_right].get_position()
#     mid_x = (p1.x1 + p2.x0) / 2
#     line = Line2D([mid_x, mid_x], [0.01, 0.99], color='black', linewidth=1.2, transform=fig.transFigure)
#     fig.add_artist(line)

# draw_v_line(4, 5)   # 5열과 6열 사이
# draw_v_line(9, 10)  # 10열과 11열 사이

# plt.savefig('combined_figure_v3.png', dpi=300, bbox_inches='tight')

#################################################
# 3. (Supp.) Segmentation Masks
#################################################

# # 1. Row 설정: 4행으로 변경 [tamper_model, image_name]
# row_configs = [
#     ["ldm", "0051.png"], # 23,32, 18, 66
#     ["brushnet", "0019.png"],
#     ["control", "0076.png"],
#     ["hdpainter", "0052.png"]
# ]

# # 2. 디렉토리 설정: 9열에 맞춰 리스트 조정 (앞의 9개 경로 사용 예시)
# directories = [
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliceless_segm_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/gt_segm",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliceless_segm",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliceless_segm",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliceless_segm",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_bin_mask_ldm_spliceless_segm",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliceless_segm_refiner",
# ]

# row_labels = ["SD-Painter", "BrushNet", "ControlNet", "HD-Painter"]
# col_labels = ["Clean", "Perturbed", "Tampered", "GT", "WAM", "OmniGuard", "StableGuard", "APT", "APT*"]


# rows, cols = 4, 9
# fig, axes = plt.subplots(rows, cols, figsize=(20, 9))

# # 간격 설정 (wspace, hspace = 0.03)
# plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)

# # 이미지 로드 및 배치
# for r, (tamper_model, name) in enumerate(row_configs):
#     for c, base_dir in enumerate(directories):
#         ax = axes[r, c]
        
#         # 경로 생성 및 ldm 치환
#         target_dir = base_dir.replace('ldm', tamper_model)
#         img_path = os.path.join(target_dir, name)
        
#         try:
#             img = Image.open(img_path)
            
#             # 리사이즈 없이 원본 표시
#             ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
#         except:
#             ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
        
#         # 테두리 및 축 설정
#         ax.set_xticks([]); ax.set_yticks([])
#         for s in ax.spines.values():
#             s.set_visible(True)
#             s.set_edgecolor('black')
#             s.set_linewidth(0.8)

#         if c == 0:
#             ax.set_ylabel(row_labels[r], rotation=90, fontsize=24, labelpad=10)
        
#         if r == rows - 1:
#             ax.text(0.5, -0.05, col_labels[c], transform=ax.transAxes, 
#                     ha='center', va='top', fontsize=24)

# # 결과 저장
# plt.savefig('fig_supp_qual_segm.pdf', dpi=300, bbox_inches='tight')

#################################################
# 4. (Supp.) Inverse Masks
#################################################

# 1. Row 설정: 4행으로 변경 [tamper_model, image_name]
row_configs = [
    ["ldm", "0098.png"], # 23,32, 18, 66
    ["brushnet", "0005.png"],
    ["control", "0020.png"],
    ["hdpainter", "0030.png"]
]

# 2. 디렉토리 설정: 9열에 맞춰 리스트 조정 (앞의 9개 경로 사용 예시)
directories = [
    "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliceless_inverse_images",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/gt_inverse",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliceless_inverse",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliceless_inverse",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliceless_inverse",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_bin_mask_ldm_spliceless_inverse",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliceless_inverse_refiner",
]

row_labels = ["SD-Painter", "BrushNet", "ControlNet", "HD-Painter"]
col_labels = ["Clean", "Perturbed", "Tampered", "GT", "WAM", "OmniGuard", "StableGuard", "APT", "APT*"]


rows, cols = 4, 9
fig, axes = plt.subplots(rows, cols, figsize=(20, 9))

# 간격 설정 (wspace, hspace = 0.03)
plt.subplots_adjust(wspace=0.03, hspace=0.03, left=0.01, right=0.99, top=0.99, bottom=0.01)

# 이미지 로드 및 배치
for r, (tamper_model, name) in enumerate(row_configs):
    for c, base_dir in enumerate(directories):
        ax = axes[r, c]
        
        # 경로 생성 및 ldm 치환
        target_dir = base_dir.replace('ldm', tamper_model)
        img_path = os.path.join(target_dir, name)
        
        try:
            img = Image.open(img_path)
            
            # 리사이즈 없이 원본 표시
            ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
        except:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10)
        
        # 테두리 및 축 설정
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor('black')
            s.set_linewidth(0.8)

        if c == 0:
            ax.set_ylabel(row_labels[r], rotation=90, fontsize=24, labelpad=10)
        
        if r == rows - 1:
            ax.text(0.5, -0.05, col_labels[c], transform=ax.transAxes, 
                    ha='center', va='top', fontsize=24)

# 결과 저장
plt.savefig('fig_supp_qual_inverse.png', dpi=300, bbox_inches='tight')