import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from fg_bg_perlin import PerlinNoiseGenerator
from skimage.metrics import structural_similarity as ssim_func

# ==========================================
# 2. Configuration
# ==========================================
BASE_DIR = "/mnt/nas5/suhyeon/projects/locmark_analysis/perlin_analysis"
CSV_PATH = os.path.join(BASE_DIR, "perlin_fg_bg_results.csv")
NPY_DIR = os.path.join(BASE_DIR, "heatmaps")

IMG_SIZE = 512
TARGET_PSNR = 30.0
VIS_SCALING = 1.0
NUM_AVG = 100

# ==========================================
# 3. Data Processing Functions
# ==========================================
def get_dataset_statistics():
    df = pd.read_csv(CSV_PATH)
    avg_d = df['D_BG_Survival'].mean()
    avg_f = df['F_BG_Survival'].mean()
    return avg_d, avg_f

def compute_average_heatmap(band_char):
    files = sorted(glob.glob(os.path.join(NPY_DIR, f"*_{band_char}_heatmap.npy")))[:NUM_AVG]
    if not files:
        raise FileNotFoundError(f"No heatmaps found for {band_char} in {NPY_DIR}")
    
    avg_map = np.mean([np.load(f) for f in files], axis=0)
    # 나중을 위해 평균 맵 저장
    # np.save(f"avg_heatmap_{band_char}_n{NUM_AVG}.npy", avg_map)
    return avg_map

# ==========================================
# 4. Main Visualization Logic
# ==========================================
def main():
    # 데이터 로드
    avg_d_stat, avg_f_stat = get_dataset_statistics()
    heatmap_d = compute_average_heatmap("D")
    heatmap_f = compute_average_heatmap("F")

    # 노이즈 생성 (PSNR 30dB 타겟)
    generator = PerlinNoiseGenerator(IMG_SIZE, device="cpu")
    target_std = np.sqrt(10**(-TARGET_PSNR / 10.0))

    def get_scaled_noise(scale):
        raw_noise = generator.generate(scale)
        # PSNR 30dB에 맞게 정규화
        return raw_noise * (target_std / (np.std(raw_noise) + 1e-8))

    noise_d = get_scaled_noise(64)  # Scale D
    noise_f = get_scaled_noise(256) # Scale F

    # Plot 그리기
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    plt.subplots_adjust(wspace=0.15, hspace=0.3)

    # (a) Pixel-level (Scale F, 2px Grid)
    axes[0, 0].imshow(np.clip(0.5 + noise_f * VIS_SCALING, 0, 1))
    axes[0, 0].set_title("(a) Pixel-level Noise (2px Grid)", fontsize=20)
    axes[0, 0].axis('off')

    im_f = axes[1, 0].imshow(heatmap_f, cmap='magma', vmin=0, vmax=1)
    axes[1, 0].set_title(f"Avg Heatmap (F)\nBG Avg Correlation: {avg_f_stat:.4f}", fontsize=20)
    axes[1, 0].axis('off')

    # (b) Texture-level (Scale D, 8px Grid)
    axes[0, 1].imshow(np.clip(0.5 + noise_d * VIS_SCALING, 0, 1))
    axes[0, 1].set_title("(b) Texture-level Noise (8px Grid)", fontsize=20)
    axes[0, 1].axis('off')

    im_d = axes[1, 1].imshow(heatmap_d, cmap='magma', vmin=0, vmax=1)
    axes[1, 1].set_title(f"Avg Heatmap (D)\nBG Avg Correlation: {avg_d_stat:.4f}", fontsize=20)
    axes[1, 1].axis('off')
    plt.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(f"Granularity Analysis (Avg over {NUM_AVG} images)", fontsize=20, y=0.97)
    
    output_name = "comparison_perlin_d_f_final.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_name}")

if __name__ == "__main__":
    main()
