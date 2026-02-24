import cv2
import torch
import os
import timm
import torchvision.transforms as T

# ==========================================
# 1. 저장된 텐서(.pt) 분석 함수
# ==========================================
def analyze_saved_tensors(gt_dir, tensor_dir, num_images=100):
    mask_pixels, non_mask_pixels = [], []

    for i in range(1, num_images + 1):
        base_name = f"{i:04d}"
        gt_path = os.path.join(gt_dir, f"{base_name}.png")
        pt_path = os.path.join(tensor_dir, f"{base_name}.pt")

        if not os.path.exists(gt_path) or not os.path.exists(pt_path):
            continue

        tensor_val = torch.load(pt_path, map_location='cpu').squeeze()
        H, W = tensor_val.shape

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        binary_mask = torch.tensor(gt_resized, dtype=torch.float32) / 255.0 > 0.5

        mask_pixels.append(tensor_val[binary_mask])
        non_mask_pixels.append(tensor_val[~binary_mask])

    return compute_stats(mask_pixels, non_mask_pixels)

# ==========================================
# 2. Clean 이미지 실시간 분석 함수 (Feature 추출)
# ==========================================
def analyze_clean_images(gt_dir, img_dir, model, direction_vectors, device, num_images=100):
    norm_imagenet = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = T.ToTensor()
    
    mask_pixels, non_mask_pixels = [], []

    for i in range(1, num_images + 1):
        base_name = f"{i:04d}"
        gt_path = os.path.join(gt_dir, f"{base_name}.png")
        img_path = os.path.join(img_dir, f"{base_name}.png") # 확장자 필요시 수정

        if not os.path.exists(gt_path) or not os.path.exists(img_path):
            continue

        # 원본 이미지 로드
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 모델 추론
        with torch.no_grad():
            norm_img = norm_imagenet(img_tensor)
            features = model(norm_img)[1] 
            B, C, H, W = features.shape
            
            features_flat = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            epsilon = 1e-6
            features_norm = features_flat / (torch.norm(features_flat, p=2, dim=-1, keepdim=True) + epsilon)
            
            dot_products = torch.matmul(features_norm, direction_vectors.T)
            cossim_map = dot_products[0, :, 0].view(H, W).cpu()

        # GT 리사이즈 및 마스크 적용
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_resized = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        binary_mask = torch.tensor(gt_resized, dtype=torch.float32) / 255.0 > 0.5

        mask_pixels.append(cossim_map[binary_mask])
        non_mask_pixels.append(cossim_map[~binary_mask])

    return compute_stats(mask_pixels, non_mask_pixels)

# ==========================================
# 3. 통계 계산 유틸리티
# ==========================================
def compute_stats(mask_pixels, non_mask_pixels):
    if not mask_pixels or not non_mask_pixels:
        return None

    all_mask = torch.cat(mask_pixels)
    all_non_mask = torch.cat(non_mask_pixels)

    return {
        "Mask": {"Mean": all_mask.mean().item(), "Std": all_mask.std().item()},
        "Non-Mask": {"Mean": all_non_mask.mean().item(), "Std": all_non_mask.std().item()}
    }

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [수정 필요] 각 디렉토리 경로를 정확히 입력하세요.
    gt_dir = "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_psnr/hinge-hard-noise-target-0.1/20260222-122031/gt"
    clean_img_dir = "CLEAN_IMAGE_DIRECTORY_경로_입력" 
    optimized_dir = "OPTIMIZED_PT_DIRECTORY_경로_입력"
    spliced_dir = "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_psnr/hinge-hard-noise-target-0.1/20260222-122031/cossim_ldm_spliced"
    spliceless_dir = "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_psnr/hinge-hard-noise-target-0.1/20260222-122031/cossim_ldm_spliceless"

    # 모델 초기화 (Clean 이미지용)
    feature_dim = 192
    direction_vectors = torch.load(f'/mnt/nas5/suhyeon/projects/freq-loc/ablation_full_{feature_dim}.pt').to(device)
    
    image_encoder = timm.create_model('convnext_small.dinov3_lvd1689m', pretrained=True, features_only=True).to(device)
    image_encoder.eval()

    # 분석 실행
    print("--- Analyzing: Clean Images ---")
    stats_clean = analyze_clean_images(gt_dir, clean_img_dir, image_encoder, direction_vectors, device)
    if stats_clean:
        print(f"[Mask] Mean: {stats_clean['Mask']['Mean']:.4f}, Std: {stats_clean['Mask']['Std']:.4f}")
        print(f"[Non-Mask] Mean: {stats_clean['Non-Mask']['Mean']:.4f}, Std: {stats_clean['Non-Mask']['Std']:.4f}")

    saved_configs = [
        ("Optimized", optimized_dir),
        ("Spliced", spliced_dir),
        ("Spliceless", spliceless_dir)
    ]

    for name, path in saved_configs:
        print(f"\n--- Analyzing: {name} Tensors ---")
        stats = analyze_saved_tensors(gt_dir, path)
        if stats:
            print(f"[Mask] Mean: {stats['Mask']['Mean']:.4f}, Std: {stats['Mask']['Std']:.4f}")
            print(f"[Non-Mask] Mean: {stats['Non-Mask']['Mean']:.4f}, Std: {stats['Non-Mask']['Std']:.4f}")
        else:
            print(f"{name} 데이터를 찾을 수 없습니다. 경로를 확인하세요: {path}")