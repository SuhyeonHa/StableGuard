import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from radialProfile import azimuthalAverage
from diffusers import StableDiffusionInpaintPipeline
from torchvision import transforms

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "methods": {
        "WAM": "/mnt/nas5/suhyeon/projects/eval_spliceless/wam/256_valAGE_sd_1.2_wm_wofilter/cover_images",
        "StableGuard": "/mnt/nas5/suhyeon/projects/eval_spliceless/stableguard/256_valAGE_sd_1.2_wm_wofilter/cover_images",
        "OmniGuard": "/mnt/nas5/suhyeon/projects/eval_spliceless/omniguard/all_512_eval_256/cover_images",
        "Ours": "/mnt/nas5/suhyeon/projects/eval_spliceless/ours_full/20260104-075451/cover_images"
    },
    "orig_path": "/mnt/nas5/suhyeon/datasets/valAGE-Set",
    "save_dir": "./watermark_fft_analysis",
    "img_size": (256, 256),
    "num_images": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_dir": "/mnt/nas5/suhyeon/caches"
}

# ==========================================
# 3. Watermark Inpainting Analyzer
# ==========================================
class WatermarkInpaintingAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(self.cfg['save_dir'], exist_ok=True)
        
        # Inpainting 모델 로드
        print("Loading Inpainting Pipeline...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if cfg['device'] == 'cuda' else torch.float32,
            cache_dir=cfg['cache_dir'],
            safety_checker=None
        ).to(cfg['device'])
        self.pipe.set_progress_bar_config(disable=True)

    def get_mask(self):
        mask = Image.new("L", self.cfg['img_size'], 0)
        return mask

    def run_analysis(self):
        mask_pil = self.get_mask()
        
        for name, path in self.cfg['methods'].items():
            print(f"\nAnalyzing Method: {name}")
            wm_files = sorted(glob.glob(os.path.join(path, "*.*")))[:self.cfg['num_images']]
            
            mag_in_total = None  # 인페인팅 전 워터마크 매그니튜드
            mag_out_total = None # 인페인팅 후 워터마크 매그니튜드
            count = 0

            for wm_p in tqdm(wm_files, desc=f"[{name}] Inpainting & FFT"):
                fname = os.path.basename(wm_p)
                orig_p = os.path.join(self.cfg['orig_path'], fname)
                if not os.path.exists(orig_p): continue
                
                try:
                    # 1. 이미지 로드 및 텐서화
                    img_orig = Image.open(orig_p).convert("RGB").resize(self.cfg['img_size'], Image.LANCZOS)
                    img_wm = Image.open(wm_p).convert("RGB").resize(self.cfg['img_size'], Image.LANCZOS)
                    
                    t_orig = np.array(img_orig).astype(np.float32) / 255.0
                    t_wm = np.array(img_wm).astype(np.float32) / 255.0
                    
                    # 2. 인페인팅 수행
                    inpainted_img = self.pipe(prompt="", image=img_wm, mask_image=mask_pil).images[0]
                    t_inpainted = np.array(inpainted_img.resize(self.cfg['img_size'])).astype(np.float32) / 255.0
                    
                    # 3. 워터마크 추출 (요청에 따라 모두 단순 차분 수행)
                    w_in = np.mean(t_wm - t_orig, axis=-1)       # 원본 워터마크
                    w_out = np.mean(t_inpainted - t_orig, axis=-1) # 인페인팅 후 잔류 워터마크
                    
                    # 4. FFT 매그니튜드 누적
                    mag_in = np.abs(np.fft.fftshift(np.fft.fft2(w_in)))
                    mag_out = np.abs(np.fft.fftshift(np.fft.fft2(w_out)))
                    
                    if mag_in_total is None:
                        mag_in_total = np.zeros_like(mag_in)
                        mag_out_total = np.zeros_like(mag_out)
                    
                    mag_in_total += mag_in
                    mag_out_total += mag_out
                    count += 1
                except Exception as e:
                    print(f" Error {fname}: {e}")

            if count > 0:
                self.plot_evolution(name, mag_in_total/count, mag_out_total/count)

    def plot_evolution(self, method_name, avg_mag_in, avg_mag_out):
        """인페인팅 전후 주파수 변화 시각화"""
        prof_in = azimuthalAverage(avg_mag_in)[1:]
        prof_out = azimuthalAverage(avg_mag_out)[1:]
        
        # Area-under-curve = 1 정규화 (분포 비교용)
        prof_in /= (np.sum(prof_in) + 1e-8)
        prof_out /= (np.sum(prof_out) + 1e-8)
        
        plt.figure(figsize=(10, 6))
        freqs = np.arange(1, len(prof_in) + 1)
        
        plt.plot(freqs, prof_in, label="Input Watermark (Before Inp)", lw=2, color='blue')
        plt.plot(freqs, prof_out, label="Residual Watermark (After Inp)", lw=2, color='red', ls='--')
        
        plt.title(f"Watermark Frequency Evolution: {method_name}", fontsize=14)
        plt.xlabel("Frequency (Distance from DC)", fontsize=12)
        plt.ylabel("Relative Energy Density", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.cfg['save_dir'], f"evolution_{method_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Result saved for {method_name}")

if __name__ == "__main__":
    WatermarkInpaintingAnalyzer(CONFIG).run_analysis()