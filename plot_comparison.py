import os
import matplotlib.pyplot as plt
from PIL import Image

def save_2x3_batch_plots(dirs, save_dir, start=140, end=500):
    """
    dirs: 6개의 경로 리스트 [R0C0, R0C1, R0C2, R1C0, R1C1, R1C2]
    save_dir: 결과 저장 경로
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 0000.png ~ 0500.png 루프
    for i in range(start, end + 1):
        filename = f"{i:04d}.png"
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        success = False
        for idx, d in enumerate(dirs):
            ax = axes[idx // 3, idx % 3]
            img_path = os.path.join(d, filename)
            
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img, aspect='auto')
                success = True
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
            
            ax.axis('off')
            
        if success:
            # 간격 최소화 및 상단 여백 확보
            plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.95, bottom=0.01)
            save_path = os.path.join(save_dir, f"plot_{filename}")
            plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        
        plt.close(fig)
        if i % 50 == 0:
            print(f"Progress: {i}/{end}")

# --- 실행 설정 ---
SOURCE_DIRS = [
    "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/control_spliced_images",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_control_spliced",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/pred_mask_control_spliceless",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_bin_mask_control_spliced_refiner",
    "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/pred_bin_mask_control_spliceless_refiner"
]
TARGET_SAVE_DIR = "/mnt/nas5/suhyeon/projects/locmark_fig1/control_refiner"

save_2x3_batch_plots(SOURCE_DIRS, TARGET_SAVE_DIR)