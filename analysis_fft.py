import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from radialProfile import azimuthalAverage
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL

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
# 2. Watermark Inpainting Analyzer
# ==========================================
class WatermarkInpaintingAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(self.cfg['save_dir'], exist_ok=True)

        # Inpainting 파이프라인 로드
        print("Loading Inpainting Pipeline...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if cfg['device'] == 'cuda' else torch.float32,
            cache_dir=cfg['cache_dir'],
            safety_checker=None
        ).to(cfg['device'])
        self.pipe.set_progress_bar_config(disable=True)

        # VAE reconstruction 기준선용 VAE (inpainting pipeline에서 추출)
        # 동일한 VAE를 사용해야 inpainting과 동일한 reconstruction 특성을 가짐
        self.vae = self.pipe.vae
        self.vae.eval()

        # PIL → torch tensor 변환용 ([-1, 1] 범위)
        self.to_tensor = lambda pil: torch.from_numpy(
            np.array(pil).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0)

        # torch tensor → numpy [0, 1] 변환용
        self.to_numpy = lambda t: (
            (t.squeeze(0).permute(1, 2, 0).cpu().float().numpy() + 1.0) / 2.0
        ).clip(0, 1)

    def vae_reconstruct(self, pil_img):
        """
        원본 이미지를 VAE encode → decode하여 reconstruction 기준선 생성.
        t_recon = VAE(t_orig)
        VAE reconstruction loss (고주파 손실 등)를 기준선으로 분리하기 위함.
        """
        x = self.to_tensor(pil_img).to(
            self.cfg['device'],
            dtype=torch.float16 if self.cfg['device'] == 'cuda' else torch.float32
        )
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mean
            recon = self.vae.decode(latent).sample
        return self.to_numpy(recon)

    def get_full_mask(self):
        """전체 이미지를 inpainting 대상으로 하는 mask (zero-mask inpainting)"""
        return Image.new("L", self.cfg['img_size'], 0)

    def run_analysis(self):
        mask_pil = self.get_full_mask()

        for name, path in self.cfg['methods'].items():
            print(f"\nAnalyzing Method: {name}")
            wm_files = sorted(glob.glob(os.path.join(path, "*.*")))[:self.cfg['num_images']]

            mag_in_total  = None  # 워터마크 perturbation: t_wm - t_orig
            mag_out_total = None  # 잔류 워터마크 (VAE loss 제거): t_inpainted - t_recon
            count = 0

            for wm_p in tqdm(wm_files, desc=f"[{name}] Processing"):
                fname = os.path.basename(wm_p)
                orig_p = os.path.join(self.cfg['orig_path'], fname)
                if not os.path.exists(orig_p):
                    continue

                try:
                    # 1. 이미지 로드
                    img_orig = Image.open(orig_p).convert("RGB").resize(self.cfg['img_size'], Image.LANCZOS)
                    img_wm   = Image.open(wm_p).convert("RGB").resize(self.cfg['img_size'], Image.LANCZOS)

                    t_orig = np.array(img_orig).astype(np.float32) / 255.0
                    t_wm   = np.array(img_wm).astype(np.float32)   / 255.0

                    # 2. VAE reconstruction 기준선 생성 (선택 A)
                    #    t_recon = VAE(t_orig)
                    #    이를 통해 inpainting 결과에서 VAE reconstruction loss를 분리
                    t_recon = self.vae_reconstruct(img_orig)

                    # 3. Inpainting 수행
                    inpainted_img = self.pipe(
                        prompt="", image=img_wm, mask_image=mask_pil
                    ).images[0]
                    t_inpainted = np.array(
                        inpainted_img.resize(self.cfg['img_size'])
                    ).astype(np.float32) / 255.0

                    # 4. 워터마크 신호 추출
                    #    w_in:  원본 워터마크 perturbation
                    #    w_out: inpainting 후 잔류 신호에서 VAE reconstruction loss 제거
                    #           = (t_inpainted - t_orig) - (t_recon - t_orig)
                    #           = t_inpainted - t_recon
                    w_in  = np.mean(t_wm - t_orig,        axis=-1)
                    w_out = np.mean(t_inpainted - t_recon, axis=-1)

                    # 5. FFT magnitude 누적
                    mag_in  = np.abs(np.fft.fftshift(np.fft.fft2(w_in)))
                    mag_out = np.abs(np.fft.fftshift(np.fft.fft2(w_out)))

                    if mag_in_total is None:
                        mag_in_total  = np.zeros_like(mag_in)
                        mag_out_total = np.zeros_like(mag_out)

                    mag_in_total  += mag_in
                    mag_out_total += mag_out
                    count += 1

                except Exception as e:
                    print(f"  Error [{fname}]: {e}")

            if count > 0:
                print(f"  Processed {count} images.")
                self.plot_evolution(name, mag_in_total / count, mag_out_total / count)

    def plot_evolution(self, method_name, avg_mag_in, avg_mag_out):
        """inpainting 전후 워터마크 주파수 분포 시각화"""
        prof_in  = azimuthalAverage(avg_mag_in)[1:]
        prof_out = azimuthalAverage(avg_mag_out)[1:]

        # AUC=1 정규화: 절대 크기가 아닌 주파수 분포(shape) 비교
        prof_in  /= (np.sum(prof_in)  + 1e-8)
        prof_out /= (np.sum(prof_out) + 1e-8)

        plt.figure(figsize=(10, 6))
        freqs = np.arange(1, len(prof_in) + 1)

        plt.plot(freqs, prof_in,  label="Input Watermark (Before Inpainting)",   lw=2, color='blue')
        plt.plot(freqs, prof_out, label="Residual Watermark (After Inpainting)",  lw=2, color='red', ls='--')

        plt.title(f"Watermark Frequency Distribution: {method_name}", fontsize=14)
        plt.xlabel("Frequency (Distance from DC)", fontsize=12)
        plt.ylabel("Relative Energy Density (AUC=1)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.cfg['save_dir'], f"evolution_{method_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


if __name__ == "__main__":
    WatermarkInpaintingAnalyzer(CONFIG).run_analysis()