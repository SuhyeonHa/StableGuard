import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = "/mnt/nas5/suhyeon/projects/eval_spliceless"

EXPERIMENTS = {
    "ours_full": "ours_full/20260104-075451",
    "ours_ones": "ours_ones/20260104-083833",
    "ours_random": "ours_random/20260104-080115",
    "ours_random_mean": "ours_random_mean/20260104-080404", 
    "ours_random_quan": "ours_random_quan/20260104-080730"
}

PRED_FOLDER_NAME = "pred_mask_zero_mask" 

def evaluate_watermark_retention(exp_name, exp_path):
    mask_dir = os.path.join(BASE_DIR, exp_path, PRED_FOLDER_NAME)
    
    if not os.path.exists(mask_dir):
        print(f"[Warning] Directory not found: {mask_dir}")
        return None

    files = sorted(glob.glob(os.path.join(mask_dir, "*.png")) + 
                   glob.glob(os.path.join(mask_dir, "*.jpg")))
    
    if len(files) == 0:
        print(f"[Warning] No images found in {mask_dir}")
        return None

    confidences = []
    accuracies = []

    print(f"Processing {exp_name} ({len(files)} images)...")
    
    for f in tqdm(files, leave=False):
        try:
            # 1. Load Prediction (GrayScale)
            # save_image는 0-1 값을 0-255로 저장하므로 로드 후 정규화
            img = Image.open(f).convert('L')
            pred_prob = np.array(img).astype(np.float32) / 255.0
            
            # 2. Metric 1: Mean Confidence (Signal Strength)
            # 전체 픽셀의 평균 확률값 (GT가 All-One이므로 높을수록 좋음)
            mean_conf = np.mean(pred_prob)
            confidences.append(mean_conf)
            
            # 3. Metric 2: Pixel Accuracy / TPR (Detectability)
            # Threshold 0.5 기준, 1로 예측된 픽셀 비율
            # GT가 All-One이므로, Accuracy = Recall(TPR)과 동일
            acc = np.mean(pred_prob > 0.5)
            accuracies.append(acc)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    return {
        "Experiment": exp_name,
        "Mean Confidence": np.mean(confidences),
        "Pixel Accuracy (TPR)": np.mean(accuracies),
        "Std Confidence": np.std(confidences)
    }

# ==========================================
# 2. Main Execution
# ==========================================
if __name__ == "__main__":
    results = []

    for exp_name, rel_path in EXPERIMENTS.items():
        res = evaluate_watermark_retention(exp_name, rel_path)
        if res:
            results.append(res)

    # 결과 출력 및 저장
    if results:
        df = pd.DataFrame(results)
        
        # 보기 좋게 포맷팅
        pd.options.display.float_format = '{:.4f}'.format
        
        print("\n" + "="*50)
        print("Quantitative Analysis: Zero-Mask Inpainting Retention")
        print("="*50)
        print(df)
        print("="*50)

        # CSV 저장
        save_csv_path = "watermark_retention_summary.csv"
        df.to_csv(save_csv_path, index=False)
        print(f"\n[Info] Results saved to {save_csv_path}")
    else:
        print("\n[Error] No results to show. Check your paths.")