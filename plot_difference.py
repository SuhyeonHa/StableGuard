import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 설정값 ---
DIR_ORIG = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images"
DIR_PERT = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images"
# 4개의 파일 이름 설정
FILE_NAMES = ["0070.png", "0102.png", "0382.png", "0415.png"] #75

def plot_perturbations_horizontal(dir_orig, dir_pert, filenames):
    # 3행 4열 Subplot 설정 (가로로 긴 형태)
    # 3 rows: Original, Perturbed, Difference
    # 4 columns: Each filename
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    
    row_titles = ["Original", "Perturbed", "Difference"]

    for col in range(4):
        fname = filenames[col]
        path_orig = os.path.join(dir_orig, fname)
        path_pert = os.path.join(dir_pert, fname)
        
        if not (os.path.exists(path_orig) and os.path.exists(path_pert)):
            print(f"Error: Missing files for {fname}")
            continue

        img_orig_pil = Image.open(path_orig).convert('RGB')
        img_pert_pil = Image.open(path_pert).convert('RGB')
        
        orig_np = np.array(img_orig_pil, dtype=np.float32)
        pert_np = np.array(img_pert_pil, dtype=np.float32)
        
        # Difference 계산 및 스케일 조정 (* 5.0)
        diff = np.abs(pert_np - orig_np) * 10.0
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        orig_np = orig_np.astype(np.uint8)
        pert_np = pert_np.astype(np.uint8)
        
        imgs_to_plot = [orig_np, pert_np, diff]

        for row in range(3):
            ax = axes[row, col]
            ax.imshow(imgs_to_plot[row], aspect='auto')
            
            # 첫 번째 열에만 행 제목 표시
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=36, labelpad=10)

            ax.set_xticks([])
            ax.set_yticks([])
            # 테두리 설정
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor('black')
                s.set_linewidth(1.0)

    # 간격 조정
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.02, 
                        wspace=0.03, hspace=0.03)

    save_path = "fig_diff.pdf"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")

plot_perturbations_horizontal(DIR_ORIG, DIR_PERT, FILE_NAMES)