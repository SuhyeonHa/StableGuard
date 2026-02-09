import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "src_path": "/mnt/nas5/suhyeon/datasets/valAGE-Set", 
    "save_dir": "./texture_modulation_visuals",
    "num_images": 5, # 시각화 확인을 위해 5장 진행
    "img_size": 512,
    "mask_size": 256,
    "anchor_dim": 4, # VAE Latent 채널 수
    "perturb_budget": 3.0, # 시각적/통계적 차이를 명확히 보기 위해 높게 설정
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_dir": "/mnt/nas5/suhyeon/caches"
}

# ==========================================
# 2. Texture Modulation & Persistence Logic
# ==========================================
class TextureModulator:
    def __init__(self, dim, device):
        self.anchor = torch.randint(0, 2, (dim,), device=device).float() * 2 - 1
        self.anchor = F.normalize(self.anchor, p=2, dim=0)

    def get_random_perturbation(self, latent_shape, budget):
        delta = torch.randn(latent_shape, device=CONFIG['device'])
        return F.normalize(delta, p=2) * budget

    def get_aligned_perturbation(self, latent, budget):
        delta = self.anchor.view(1, -1, 1, 1).expand_as(latent)
        return F.normalize(delta, p=2) * budget

    def get_bg_cosine_similarity(self, latent, bg_mask):
        masked_latent = latent * bg_mask
        bg_sum = masked_latent.sum(dim=(2, 3))
        bg_count = bg_mask.sum(dim=(2, 3)) + 1e-6
        avg_v = bg_sum / bg_count
        sim = F.cosine_similarity(avg_v, self.anchor.unsqueeze(0))
        return sim.mean().item()

# ==========================================
# 3. Persistence Analysis & Plotting Pipeline
# ==========================================
class PersistenceVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['device']
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            cache_dir=cfg['cache_dir'],
            safety_checker=None
        ).to(self.device)
        self.vae = self.pipe.vae
        self.modulator = TextureModulator(cfg['anchor_dim'], self.device)
        os.makedirs(cfg['save_dir'], exist_ok=True)

    def create_masks(self):
        mask_pil = Image.new("L", (self.cfg['img_size'], self.cfg['img_size']), 0)
        draw = ImageDraw.Draw(mask_pil)
        offset = (self.cfg['img_size'] - self.cfg['mask_size']) // 2
        draw.rectangle([offset, offset, offset + self.cfg['mask_size'], offset + self.cfg['mask_size']], fill=255)
        
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask_torch = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)
        bg_mask_latent = 1.0 - F.interpolate(mask_torch, size=(self.cfg['img_size']//8, self.cfg['img_size']//8), mode='nearest')
        return mask_pil, bg_mask_latent

    def tensor_to_pil(self, tensor):
        img = (tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    def run(self):
        paths = sorted(glob.glob(os.path.join(self.cfg['src_path'], "*.*")))[:self.cfg['num_images']]
        mask_pil, bg_mask_latent = self.create_masks()
        results = []

        for idx, p in enumerate(tqdm(paths, desc="Visual Modulation Analysis")):
            try:
                img_pil = Image.open(p).convert("RGB").resize((self.cfg['img_size'], self.cfg['img_size']))
                img_tensor = (torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float().to(self.device) / 127.5 - 1).unsqueeze(0)
                
                with torch.no_grad():
                    latent = self.vae.encode(img_tensor.to(self.pipe.dtype)).latent_dist.sample()

                d_rand = self.modulator.get_random_perturbation(latent.shape, self.cfg['perturb_budget']).to(self.pipe.dtype)
                d_align = self.modulator.get_aligned_perturbation(latent, self.cfg['perturb_budget']).to(self.pipe.dtype)

                case_imgs = {"Original": img_pil}
                case_sims = {}

                for name, v_mod in [("Random", latent + d_rand), ("Aligned", latent + d_align)]:
                    with torch.no_grad():
                        img_mod_pil = self.tensor_to_pil(self.vae.decode(v_mod).sample)
                    
                    # 배경 재생성 수행
                    out_pil = self.pipe(prompt="", image=img_mod_pil, mask_image=mask_pil).images[0]
                    
                    case_imgs[f"{name}_WM"] = img_mod_pil
                    case_imgs[f"{name}_Inpainted"] = out_pil
                    
                    # 생존 여부 측정
                    out_tensor = (torch.from_numpy(np.array(out_pil)).permute(2, 0, 1).float().to(self.device) / 127.5 - 1).unsqueeze(0)
                    with torch.no_grad():
                        v_rec = self.vae.encode(out_tensor.to(self.pipe.dtype)).latent_dist.mode()
                    case_sims[name] = self.modulator.get_bg_cosine_similarity(v_rec, bg_mask_latent)

                results.append({"Image": os.path.basename(p), "Random_Survival": case_sims["Random"], "Aligned_Survival": case_sims["Aligned"]})

                # 이미지 결과 Plot 저장
                self.plot_sample(case_imgs, case_sims, idx, os.path.basename(p))

            except Exception as e:
                print(f"Error {p}: {e}")

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.cfg['save_dir'], "survival_results_final.csv"), index=False)
        self.print_summary(df)
        self.plot_summary(df)

    def plot_sample(self, imgs, sims, idx, fname):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        # Row 1: Random Additive
        axes[0, 0].imshow(imgs["Original"]); axes[0, 0].set_title("Original Image"); axes[0, 0].axis('off')
        axes[0, 1].imshow(imgs["Random_WM"]); axes[0, 1].set_title(f"Random WM (Budget {self.cfg['perturb_budget']})"); axes[0, 1].axis('off')
        axes[0, 2].imshow(imgs["Random_Inpainted"]); axes[0, 2].set_title(f"After Regeneration (Sim: {sims['Random']:.4f})"); axes[0, 2].axis('off')
        # Row 2: Anchor-Aligned
        axes[1, 0].imshow(imgs["Original"]); axes[1, 0].set_title("Original Image"); axes[1, 0].axis('off')
        axes[1, 1].imshow(imgs["Aligned_WM"]); axes[1, 1].set_title(f"Aligned WM (Budget {self.cfg['perturb_budget']})"); axes[1, 1].axis('off')
        axes[1, 2].imshow(imgs["Aligned_Inpainted"]); axes[1, 2].set_title(f"After Regeneration (Sim: {sims['Aligned']:.4f})"); axes[1, 2].axis('off')
        
        plt.suptitle(f"Sample {idx}: {fname} Comparison", fontsize=16)
        plt.tight_layout(); plt.savefig(os.path.join(self.cfg['save_dir'], f"sample_{idx}_comparison.png")); plt.close()

    def plot_summary(self, df):
        plt.figure(figsize=(8, 6))
        means = [df['Random_Survival'].mean(), df['Aligned_Survival'].mean()]
        stds = [df['Random_Survival'].std(), df['Aligned_Survival'].std()]
        plt.bar(['Random Additive', 'Anchor-Aligned'], means, yerr=stds, color=['gray', 'blue'], alpha=0.7, capsize=10)
        plt.ylabel('Cosine Similarity (Background Survival)'); plt.title('Persistence Comparison'); plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(self.cfg['save_dir'], "survival_summary_plot.png")); plt.close()

    def print_summary(self, df):
        print("\n" + "="*60)
        print("Survival Analysis (Background Region, High Budget)")
        print("-" * 60)
        print(f"Random Survival (Avg CosSim):  {df['Random_Survival'].mean():.4f}")
        print(f"Aligned Survival (Avg CosSim): {df['Aligned_Survival'].mean():.4f}")
        print("="*60)

if __name__ == "__main__":
    PersistenceVisualizer(CONFIG).run()