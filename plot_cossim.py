import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 파일명과 라벨, 색상 매핑
# (파일명, 라벨, 색상)
root_dir = "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_full/20260104-075451"

files_info = [
    # --- Blue Tones (Watermarked / Valid Regions) ---
    (os.path.join(root_dir, "logits_cover.npz"),          "Watermarked",              "blue"),
    # (os.path.join(root_dir, "logits_zero_mask.npz"),      "Zero Mask (Watermarked)",  "dodgerblue"),
    # (os.path.join(root_dir, "logits_spliced_in.npz"),     "Spliced (Watermarked)",    "cyan"),
    # (os.path.join(root_dir, "logits_spliceless_in.npz"),  "Spliceless (Watermarked)", "teal"),
    
    # --- Red Tones (Manipulated / Clean Regions) ---
    (os.path.join(root_dir, "logits_clean.npz"),          "Clean",                    "red"),
    (os.path.join(root_dir, "logits_spliced_out.npz"),    "Spliced (Manipulated)",    "orange"),
    (os.path.join(root_dir, "logits_spliceless_out.npz"), "Spliceless (Manipulated)", "magenta"),
]

print("Plotting distribution...")
plt.figure(figsize=(7, 7))

for filename, label, color in files_info:
    if os.path.exists(filename):
        try:
            # key='logits'로 데이터 로드
            data = np.load(filename)['logits']
            
            # 데이터가 비어있지 않은지 확인
            if len(data) > 0:
                sns.kdeplot(data, fill=True, label=label, color=color, alpha=0.15, linewidth=2)
            else:
                print(f"Warning: {filename} is empty.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    else:
        print(f"File not found: {filename}")

plt.title("Logits Distribution Comparison", fontsize=20)
plt.xlabel("Cosine Similarity", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.legend(fontsize=15)
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=15)

# x축 범위 설정 (데이터 분포에 따라 조절 필요)
plt.xlim(-0.3, 0.3) 

# 결과 저장
save_path = "./logits_dist_combined.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Plot saved to: {save_path}")
# plt.show()