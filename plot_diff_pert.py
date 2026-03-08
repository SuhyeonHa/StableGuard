import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

def save_diff_images(dir_cl, dir_cover, dir_spliced, dir_spliceless, dir_mask, filename):
    path_cl = os.path.join(dir_cl, filename)
    path_cv = os.path.join(dir_cover, filename)
    path_sp = os.path.join(dir_spliced, filename)
    path_sl = os.path.join(dir_spliceless, filename)
    path_ms = os.path.join(dir_mask, filename)

    transform = T.ToTensor()
    
    # 이미지 로드
    img_cl = transform(Image.open(path_cl).convert('RGB')).unsqueeze(0)
    img_cv = transform(Image.open(path_cv).convert('RGB')).unsqueeze(0)
    # img_sp = transform(Image.open(path_sp).convert('RGB'))
    # img_sl = transform(Image.open(path_sl).convert('RGB'))
    img_sp = transform(Image.open(path_sp).convert('RGB')).unsqueeze(0)
    img_sl = transform(Image.open(path_sl).convert('RGB')).unsqueeze(0)
    
    # 마스크 로드 (L: Grayscale)
    mask = transform(Image.open(path_ms).convert('L')).unsqueeze(0) # [1, 1, H, W]

    # --- 이미지 보간 (512 -> 256) ---
    def resample(x):
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    img_cl = resample(img_cl)
    img_cv = resample(img_cv)
    img_sp = resample(img_sp)
    img_sl = resample(img_sl)
    
    # 마스크도 동일한 크기로 보간 (256, 256)
    mask = F.interpolate(mask, size=(256, 256), mode="nearest").squeeze(0) # [1, 256, 256]

    # --- 차이 계산 ---
    diff_pt = torch.abs(img_cl - img_cv) * 10.0
    diff_sp = torch.abs(img_cl - img_sp) * 10.0
    diff_sl = torch.abs(img_cl - img_sl) * 10.0

    # --- 마스킹 적용 (배경 영역만 남기기) ---
    # mask가 1인 곳이 배경, 0인 곳이 객체라고 가정할 때 diff * mask 수행
    diff_pt = diff_pt
    diff_sp = diff_sp * mask
    diff_sl = diff_sl * mask

    # 결과 저장
    to_pil = T.ToPILImage()
    to_pil(torch.clamp(diff_pt, 0, 1)).save(f"fig1_diff_p.png")
    to_pil(torch.clamp(diff_sp, 0, 1)).save(f"fig1_diff_sp.png")
    to_pil(torch.clamp(diff_sl, 0, 1)).save(f"fig1_diff_fr.png")

    print(f"Saved masked difference images for {filename}")

# 사용 예시
# DIR_CL = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images"
# DIR_CV = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images"
# DIR_SP = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliced_images"
# DIR_SL = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliceless_images"

# wam
# DIR_CL = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images"
# DIR_CV = "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/cover_images"
# DIR_SP = "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/ldm_spliced_images"
# DIR_SL = "/mnt/nas5/suhyeon/projects/locmark_table_1/wam/ldm_spliceless_images"
# DIR_MS = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt"

#omniguard
DIR_CL = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images"
DIR_CV = "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/cover_images"
DIR_SP = "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/ldm_spliced_images"
DIR_SL = "/mnt/nas5/suhyeon/projects/locmark_table_1/omniguard/ldm_spliceless_images"
DIR_MS = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt"

#stableguard
# DIR_CL = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images"
# DIR_CV = "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/cover_images"
# DIR_SP = "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/ldm_spliced_images"
# DIR_SL = "/mnt/nas5/suhyeon/projects/locmark_table_1/stableguard/ldm_spliceless_images"
# DIR_MS = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt"

#ours
# DIR_CL = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/cover_images"
# DIR_CV = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/cover_images"
# DIR_SP = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliced_images"
# DIR_SL = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/ldm_spliceless_images"
# DIR_MS = "/mnt/nas5/suhyeon/projects/locmark_table_1/clean/gt"

FILE = "0221.png"
save_diff_images(DIR_CL, DIR_CV, DIR_SP, DIR_SL, DIR_MS, FILE)