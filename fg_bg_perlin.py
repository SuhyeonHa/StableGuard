import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "src_path": "/mnt/nas5/suhyeon/datasets/valAGE-Set",
    "save_dir": "/mnt/nas5/suhyeon/projects/locmark_analysis/perlin_analysis",
    "save_csv": "perlin_fg_bg_results.csv",
    "num_images": 100,
    "img_size": 512,
    "mask_size": 256,
    "target_psnr": 30.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_dir": "/mnt/nas5/suhyeon/caches",
    "save_samples": True,
    # Scales and corresponding Grid Sizes (512 / Scale)
    "scales": {
        "A": 8,   # 64px grid
        "B": 16,  # 32px grid
        "C": 32,  # 16px grid
        "D": 64,  # 8px grid
        "E": 128, # 4px grid
        "F": 256  # 2px grid
    },
    "vis_scaling": 1.0
}

# ==========================================
# 2. Perlin Noise Generator
# ==========================================
class PerlinNoiseGenerator:
    def __init__(self, size, device):
        self.size = size
        self.device = device

    def generate(self, scale):
        grid_size = scale
        angles = 2 * np.pi * torch.rand((grid_size + 1, grid_size + 1), device=self.device)
        grads = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        
        y, x = torch.meshgrid(torch.linspace(0, grid_size, self.size, device=self.device),
                              torch.linspace(0, grid_size, self.size, device=self.device), indexing='ij')
        
        x0, y0 = x.long(), y.long()
        x1, y1 = torch.clamp(x0 + 1, max=grid_size), torch.clamp(y0 + 1, max=grid_size)
        dx, dy = x - x0.float(), y - y0.float()
        
        def dot(grad, dx, dy): return grad[..., 0] * dx + grad[..., 1] * dy
        n00, n10 = dot(grads[y0, x0], dx, dy), dot(grads[y0, x1], dx - 1, dy)
        n01, n11 = dot(grads[y1, x0], dx, dy - 1), dot(grads[y1, x1], dx - 1, dy - 1)
        
        def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
        wx, wy = fade(dx), fade(dy)
        
        nx0, nx1 = n00 + wx * (n10 - n00), n01 + wx * (n11 - n01)
        noise = nx0 + wy * (nx1 - nx0)
        return noise.unsqueeze(0).repeat(3, 1, 1).cpu().numpy().transpose(1, 2, 0)

    def generate_bands(self, target_psnr, cfg):
        target_std = np.sqrt(10**(-target_psnr / 10.0))
        bands = {}
        for name, scale in cfg['scales'].items():
            noise = self.generate(scale)
            bands[name] = noise * (target_std / (np.std(noise) + 1e-8))
        return bands

# ==========================================
# 3. Perlin Analysis (Grid-Matched & FG/BG Separate)
# ==========================================
class PerlinAnalysis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['device']
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            cache_dir=cfg['cache_dir'], safety_checker=None
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.gen = PerlinNoiseGenerator(cfg['img_size'], cfg['device'])
        
        self.vis_dir = os.path.join(self.cfg['save_dir'], "vis")
        self.npy_dir = os.path.join(self.cfg['save_dir'], "heatmaps")
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.npy_dir, exist_ok=True)

    def compute_local_correlation(self, img1, img2, win_size):
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        _, _, h, w = t1.shape

        def pool(x):
            # 짝수 win_size 대응을 위한 슬라이싱 추가
            out = F.avg_pool2d(x, win_size, stride=1, padding=win_size // 2)
            return out[:, :, :h, :w]

        mu1, mu2 = pool(t1), pool(t2)
        s1 = torch.sqrt(torch.clamp(pool(t1*t1) - mu1*mu1, min=1e-10))
        s2 = torch.sqrt(torch.clamp(pool(t2*t2) - mu2*mu2, min=1e-10))
        corr = (pool(t1*t2) - mu1*mu2) / (s1 * s2 + 1e-8)
        
        return torch.clamp(corr.mean(dim=1), 0, 1).squeeze().detach().cpu().numpy().astype(np.float64)

    def run(self):
        paths = sorted(glob.glob(os.path.join(self.cfg['src_path'], "*.*")))[:self.cfg['num_images']]
        mask_pil = Image.new("L", (self.cfg['img_size'], self.cfg['img_size']), 0)
        draw = ImageDraw.Draw(mask_pil)
        offset = (self.cfg['img_size'] - self.cfg['mask_size']) // 2
        draw.rectangle([offset, offset, offset + self.cfg['mask_size'], offset + self.cfg['mask_size']], fill=255)
        
        fg_mask = np.array(mask_pil) >= 128
        bg_mask = ~fg_mask
        
        results = []
        for p in tqdm(paths, desc="Perlin Grid-Matched Analysis"):
            try:
                img_orig = np.array(Image.open(p).convert("RGB").resize((self.cfg['img_size'], self.cfg['img_size']))).astype(np.float32) / 255.0
                bands = self.gen.generate_bands(self.cfg['target_psnr'], self.cfg)
                row = {"Image": os.path.basename(p)}
                
                for name, delta in bands.items():
                    win_size = self.cfg['img_size'] // self.cfg['scales'][name]
                    img_noisy_pil = Image.fromarray((np.clip(img_orig + delta, 0, 1) * 255).astype(np.uint8))
                    out_np = np.array(self.pipe(prompt="", image=img_noisy_pil, mask_image=mask_pil).images[0]).astype(np.float32) / 255.0
                    recovered = out_np - img_orig
                    
                    corr_map = self.compute_local_correlation(delta, recovered, win_size)
                    
                    # 1. 히트맵 저장 (.npy)
                    base_name = os.path.splitext(os.path.basename(p))[0]
                    np.save(os.path.join(self.npy_dir, f"{base_name}_{name}_heatmap.npy"), corr_map)
                    
                    # 2. FG/BG 개별 결과 저장
                    row[f"{name}_BG_Survival"] = np.mean(corr_map[bg_mask])
                    row[f"{name}_FG_Survival"] = np.mean(corr_map[fg_mask])
                    
                    if len(results) < 5: 
                        self.save_recovery_plot(delta, recovered, corr_map, name, os.path.basename(p), len(results)+1, bg_mask, fg_mask)
                results.append(row)
            except Exception as e: print(f"Error {p}: {e}")

        if results:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(self.cfg['save_dir'], self.cfg['save_csv']), index=False)
            self.print_summary(df)

    def save_recovery_plot(self, delta, recovered, corr_map, band, fname, idx, bg_mask, fg_mask):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        v = self.cfg['vis_scaling']
        grid_px = self.cfg['img_size'] // self.cfg['scales'][band]
        
        axes[0].imshow(np.clip(0.5 + delta * v, 0, 1)); axes[0].set_title(f"Input ({band})"); axes[0].axis('off')
        axes[1].imshow(np.clip(0.5 + recovered * v, 0, 1)); axes[1].set_title("Recovered (Inp-Orig)"); axes[1].axis('off')
        
        im = axes[2].imshow(corr_map, cmap='magma', vmin=0, vmax=1)
        bg_avg = np.mean(corr_map[bg_mask])
        fg_avg = np.mean(corr_map[fg_mask])
        title = f"Local Correlation (r)\nBG Avg: {bg_avg:.4f} / FG Avg: {fg_avg:.4f}\nGrid: {grid_px}px"
        
        axes[2].set_title(title); axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f"recovery_{idx:02d}_{band}_{fname}.png"), dpi=200, bbox_inches='tight')
        plt.close()

    def print_summary(self, df):
        print("\n" + "="*75 + f"\nPerlin Grid-Matched Analysis Summary ({self.cfg['target_psnr']}dB)\n" + "-"*75)
        for b in sorted(self.cfg['scales'].keys()):
            grid = self.cfg['img_size'] // self.cfg['scales'][b]
            bg_val = df[f"{b}_BG_Survival"].mean()
            fg_val = df[f"{b}_FG_Survival"].mean()
            print(f"{b} (Grid {grid:3d}px) | BG (Robustness): {bg_val:.4f} | FG (Erasure): {fg_val:.4f}")
        print("="*75)

if __name__ == "__main__":
    PerlinAnalysis(CONFIG).run()