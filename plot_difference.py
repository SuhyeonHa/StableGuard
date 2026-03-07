import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 설정값 ---
DIR_ORIG = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images"
DIR_PERT = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images"
FILE_NAMES = ["0070.png", "0075.png"]
# FILE_NAMES = ["0144.png", "0266.png"]

def plot_perturbations(dir_orig, dir_pert, filenames):
    # 2x3 Subplot 설정. 제목을 위해 figsize 높이를 조금 확보합니다.
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    
    col_titles = ["Original", "Perturbed", "Difference"]

    for row in range(2):
        fname = filenames[row]
        path_orig = os.path.join(dir_orig, fname)
        path_pert = os.path.join(dir_pert, fname)
        
        # 파일 존재 여부 확인
        if not (os.path.exists(path_orig) and os.path.exists(path_pert)):
            print(f"Error: Missing files for {fname}")
            continue

        # 1. 이미지 로드 (RGB 변환)
        # diff 계산을 위해 uint8을 float로 미리 변환
        img_orig_pil = Image.open(path_orig).convert('RGB')
        img_pert_pil = Image.open(path_pert).convert('RGB')
        
        orig_np = np.array(img_orig_pil, dtype=np.float32)
        pert_np = np.array(img_pert_pil, dtype=np.float32)
        
        # 이미지 크기가 맞는지 확인 (예외 처리)
        if orig_np.shape != pert_np.shape:
            print(f"Error: Shape mismatch for {fname}")
            continue

        # 2. Difference 계산: |Perturb - Original| * 10
        # uint8 오버플로우 방지를 위해 float 상태에서 계산
        diff = np.abs(pert_np - orig_np) * 5.0
        
        # 시각화를 위해 0~255 사이로 자르고 uint8로 다시 변환
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        # 출력을 위해 원본 데이터도 다시 uint8로 인식 (시각화 목적)
        orig_np = orig_np.astype(np.uint8)
        pert_np = pert_np.astype(np.uint8)
        
        imgs_to_plot = [orig_np, pert_np, diff]

        # 3. 그리기 루프
        for col in range(3):
            ax = axes[row, col]
            
            # aspect='auto'로 이미지 비율을 깨고 박스에 꽉 채웁니다 (hspace=0 효과용)
            ax.imshow(imgs_to_plot[col], aspect='auto')
            
            # 첫 번째 행에만 타이틀 추가
            if row == 0:
                # pad=30을 주어 Row 1과 간격을 넓힙니다.
                ax.set_title(col_titles[col], fontsize=36, pad=10)

            ax.axis('off')

    # 간격 조정
    plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.02, 
                        wspace=0.02, hspace=0.02)

    save_path = "fig_diff.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    # plt.show()
    print(f"Figure saved to: {save_path}")

# 함수 실행 (경로가 실제로 존재해야 합니다)
plot_perturbations(DIR_ORIG, DIR_PERT, FILE_NAMES)