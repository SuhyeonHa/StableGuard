import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps

sp = '#4682B4' # 파란색 계열
fr = '#C44E52' # 빨간색 계열

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 1. 4개의 디렉토리 경로 (사용자 환경에 맞게 수정)
directories = [
    "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/ldm_spliced_images",
    #
    "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/ldm_spliceless_images",    
]

# 2. 불러올 공통 파일 이름
file_name = "0066.png"
bottom_titles = ["Original", "Spliced", "Mask", "Fully Regenerated"]

rows, cols = 2, 2
fig, axes = plt.subplots(rows, cols, figsize=(6, 6.5))
axes_flat = axes.flatten()

plt.subplots_adjust(wspace=0.01, hspace=0.15, left=0.01, right=0.99, top=0.98, bottom=0.1)

# 테두리 인덱스
sp_indices = [1]
fr_indices = [3]

for i, dir_path in enumerate(directories):
    ax = axes_flat[i]
    try:
        img = Image.open(os.path.join(dir_path, file_name))

        if img.mode == 'L':
            img = ImageOps.invert(img)

        ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
    except:
        ax.text(0.5, -0.1, 'N/A', ha='center', va='center')
    
    # 테두리 설정
    edge_color, linewidth = 'black', 2.0
    if i in sp_indices: edge_color = sp
    elif i in fr_indices: edge_color = fr

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(linewidth)

    # 4개 이미지 모두 하단 타이틀 추가
    ax.text(0.5, -0.03, bottom_titles[i], transform=ax.transAxes, 
            ha='center', va='top', fontsize=20, fontweight='normal')

plt.savefig('fig1_qual_4.png', dpi=300, bbox_inches='tight')

##################################################################################
# # 12 images
# import matplotlib.pyplot as plt
# import os
# from PIL import Image

# sp = '#4682B4' # 파란색 계열
# fr = '#C44E52' # 빨간색 계열

# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# # 1. 12개의 디렉토리 경로 (사용자 환경에 맞게 수정)
# directories = [
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/ldm_spliced_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliced",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliced_refiner",
#     #
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/ldm_spliceless_images",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/pred_mask_ldm_spliceless",
#     "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_mask_ldm_spliceless_refiner",    
# ]

# # 2. 불러올 공통 파일 이름
# file_name = "0066.png"
# bottom_titles = ["Inputs", "SP/FR Tamper", "WAM", "OmniGuard", "StableGuard", "Ours*"]

# rows, cols = 2, 6
# fig, axes = plt.subplots(rows, cols, figsize=(18, 6.2))
# axes_flat = axes.flatten()

# plt.subplots_adjust(wspace=0.01, hspace=0.03, left=0.01, right=0.99, top=0.98, bottom=0.05)

# # 테두리 인덱스
# sp_indices = [1, 2, 3, 4, 5]
# fr_indices = [7, 8, 9, 10, 11]

# for i, dir_path in enumerate(directories):
#     ax = axes_flat[i]
#     try:
#         img = Image.open(os.path.join(dir_path, file_name))
#         ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
#     except:
#         ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
    
#     # 테두리 설정
#     edge_color, linewidth = 'black', 2.0
#     if i in sp_indices:
#         edge_color, linewidth = sp, 2.0
#     elif i in fr_indices:
#         edge_color, linewidth = fr, 2.0

#     ax.set_xticks([]); ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_edgecolor(edge_color)
#         spine.set_linewidth(linewidth)

#     # 하단 타이틀 추가 (두 번째 row: 6~11번 인덱스)
#     if 6 <= i <= 11:
#         title_idx = i - 6  # 리스트의 0~5번 인덱스에 접근
#         ax.text(0.5, -0.05, bottom_titles[title_idx], transform=ax.transAxes, 
#                 ha='center', va='top', fontsize=26)

# plt.savefig('fig1_qual.png', dpi=300, bbox_inches='tight')