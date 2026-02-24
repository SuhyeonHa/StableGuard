import cv2
import torch
import os

def analyze_tensor_stats_per_image(gt_dir, tensor_dir, num_images=100):
    mask_pixels = []
    non_mask_pixels = []

    for i in range(1, num_images + 1):
        base_name = f"{i:04d}"
        gt_path = os.path.join(gt_dir, f"{base_name}.png")
        pt_path = os.path.join(tensor_dir, f"{base_name}.pt")

        if not os.path.exists(gt_path) or not os.path.exists(pt_path):
            continue

        # 개별 텐서 로드 및 2D 형태로 변환 (예: [1, 1, 32, 32] -> [32, 32])
        tensor_val = torch.load(pt_path, map_location='cpu').squeeze()
        H, W = tensor_val.shape

        # GT 로드 및 텐서 해상도(32x32)에 맞춰 리사이즈
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        binary_mask = torch.tensor(gt_resized, dtype=torch.float32) / 255.0 > 0.5

        # 픽셀 분류
        mask_pixels.append(tensor_val[binary_mask])
        non_mask_pixels.append(tensor_val[~binary_mask])

    if not mask_pixels or not non_mask_pixels:
        return None

    all_mask_pixels = torch.cat(mask_pixels)
    all_non_mask_pixels = torch.cat(non_mask_pixels)

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
tensor_directories = [
    "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_psnr/hinge-hard-noise-target-0.1/20260222-122031/cossim_ldm_spliced",
    "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_psnr/hinge-hard-noise-target-0.1/20260222-122031/cossim_ldm_spliceless"
]

for tensor_dir in tensor_directories:
    print(f"--- Analyzing: {tensor_dir} ---")
    stats = analyze_tensor_stats_per_image(gt_directory, tensor_dir)
    if stats:
        for region, stat in stats.items():
            print(f"[{region}] Mean: {stat['Mean']:.4f}, Std: {stat['Std']:.4f}, Min: {stat['Min']:.4f}, Max: {stat['Max']:.4f}")
    else:
        print("데이터를 찾을 수 없습니다.")