import cv2
import numpy as np
import torch
import os

def analyze_logit_stats(gt_dir, logit_dir, num_images=100):
    mask_pixels = []
    non_mask_pixels = []

    for i in range(1, num_images + 1):
        filename = f"{i:04d}.png"
        gt_path = os.path.join(gt_dir, filename)
        logit_path = os.path.join(logit_dir, filename)

        if not os.path.exists(gt_path) or not os.path.exists(logit_path):
            continue

        # 이미지 로드 (Grayscale)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        logit = cv2.imread(logit_path, cv2.IMREAD_GRAYSCALE)

        gt_resized = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)

        # PyTorch Tensor 변환 및 정규화 (0~1)
        gt_tensor = torch.tensor(gt_resized, dtype=torch.float32) / 255.0
        logit_tensor = torch.tensor(logit, dtype=torch.float32) / 255.0

        # 이진화 (Threshold)
        binary_mask = (gt_tensor > 0.5)

        # Mask / Non-mask 영역 픽셀 추출
        mask_pixels.append(logit_tensor[binary_mask])
        non_mask_pixels.append(logit_tensor[~binary_mask])

    # 리스트 병합
    if not mask_pixels or not non_mask_pixels:
        return None

    all_mask_pixels = torch.cat(mask_pixels)
    all_non_mask_pixels = torch.cat(non_mask_pixels)

    # 통계 계산
    stats = {
        "Mask": {
            "Mean": all_mask_pixels.mean().item(),
            "Std": all_mask_pixels.std().item(),
            "Min": all_mask_pixels.min().item(),
            "Max": all_mask_pixels.max().item()
        },
        "Non-Mask": {
            "Mean": all_non_mask_pixels.mean().item(),
            "Std": all_non_mask_pixels.std().item(),
            "Min": all_non_mask_pixels.min().item(),
            "Max": all_non_mask_pixels.max().item()
        }
    }
    
    return stats

# 실행 예시
gt_directory = "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_psnr/hinge-hard-noise-target-0.1/20260222-122031/gt"
logit_directories = ["/mnt/nas5/suhyeon/projects/eval_spliceless/ours_psnr/hinge-hard-noise-target-0.1/20260222-122031/pred_mask_ldm_spliced",
                     "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_psnr/hinge-hard-noise-target-0.1/20260222-122031/pred_mask_ldm_spliceless"]

for logit_dir in logit_directories:
    print(f"--- Analyzing: {logit_dir} ---")
    stats = analyze_logit_stats(gt_directory, logit_dir)
    if stats:
        for region, stat in stats.items():
            print(f"[{region}] Mean: {stat['Mean']:.4f}, Std: {stat['Std']:.4f}, Min: {stat['Min']:.4f}, Max: {stat['Max']:.4f}")
    else:
        print("데이터를 찾을 수 없습니다.")