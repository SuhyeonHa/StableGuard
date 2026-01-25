import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline
import timm
from skimage.metrics import structural_similarity as ssim_metric

# ==========================================
# 1. Configuration
# ==========================================
CONFIG = {
    "src_path": "/mnt/nas5/suhyeon/datasets/valAGE-Set",
    "save_dir": "/mnt/nas5/suhyeon/projects/locmark_analysis",
    "save_csv": "fg_bg_analysis_results.csv",
    "num_images": 100,
    "img_size": 512,
    "mask_size": 256,
    "model_name": "convnext_small.dinov3_lvd1689m",
    "layer_idx": 1,
    "num_inference_steps": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "cache_dir": "/mnt/nas5/suhyeon/caches",
    "save_samples": True  # Option to save generated mask and inpainted samples
}

torch.manual_seed(CONFIG['seed'])

# ==========================================
# 2. Feature Extractor
# ==========================================
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_name, layer_idx=1, device='cuda'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True).to(device)
        self.model.eval()
        self.layer_idx = layer_idx
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, x):
        x = (x - self.mean) / self.std
        with torch.no_grad():
            features = self.model(x)
        return features[self.layer_idx]

# ==========================================
# 3. Multi-Metric Selectivity Analysis
# ==========================================
class SelectivityAnalysis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['device']
        self.extractor = FeatureExtractor(cfg['model_name'], cfg['layer_idx'], self.device)
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            cache_dir=cfg['cache_dir']
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        # Create subdirectories for masks and inpainted results only
        if self.cfg.get('save_samples', False):
            self.sample_dirs = {
                "mask": os.path.join(self.cfg['save_dir'], "mask"),
                "inpainted": os.path.join(self.cfg['save_dir'], "inpainted")
            }
            for d in self.sample_dirs.values():
                os.makedirs(d, exist_ok=True)

    def create_mask(self):
        mask_pil = Image.new("L", (self.cfg['img_size'], self.cfg['img_size']), 0)
        draw = ImageDraw.Draw(mask_pil)
        offset = (self.cfg['img_size'] - self.cfg['mask_size']) // 2
        draw.rectangle([offset, offset, offset + self.cfg['mask_size'], offset + self.cfg['mask_size']], fill=255)
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        return mask_pil, mask_np

    def compute_ssim_map(self, img_orig_np, img_inp_np, win_size=7):
        """Compute SSIM map between two images"""
        _, ssim_map = ssim_metric(
            img_orig_np, img_inp_np,
            data_range=1.0,
            channel_axis=2,
            win_size=win_size,
            full=True
        )
        return ssim_map.mean(axis=2)

    def run(self):
        paths = sorted(glob.glob(os.path.join(self.cfg['src_path'], "*.*")))[:self.cfg['num_images']]
        mask_pil, mask_np = self.create_mask()
        
        # Save common mask once
        if self.cfg.get('save_samples', False):
            mask_path = os.path.join(self.sample_dirs['mask'], "mask.png")
            mask_pil.save(mask_path)
            print(f"[Saved] Common mask saved to {mask_path}")

        results = []
        accum_maps = {'L1': None, 'L2': None, 'Cosine': None, 'SSIM': None}
        count = 0

        for p in tqdm(paths, desc="Multi-Metric Analysis"):
            try:
                img_orig_pil = Image.open(p).convert("RGB").resize((self.cfg['img_size'], self.cfg['img_size']))
                img_orig_np = np.array(img_orig_pil).astype(np.float32) / 255.0

                img_inp_pil = self.pipe(prompt="", image=img_orig_pil, mask_image=mask_pil,
                                        num_inference_steps=self.cfg['num_inference_steps']).images[0]
                img_inp_np = np.array(img_inp_pil).astype(np.float32) / 255.0

                # Save only inpainted samples
                if self.cfg.get('save_samples', False):
                    fname = os.path.basename(p)
                    img_inp_pil.save(os.path.join(self.sample_dirs['inpainted'], f"inp_{fname}"))

                # Feature extraction
                t_orig = torch.from_numpy(img_orig_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
                t_inp = torch.from_numpy(img_inp_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
                f_orig, f_inp = self.extractor(t_orig), self.extractor(t_inp)

                # Compute metrics
                l1_map = torch.norm(f_orig - f_inp, p=1, dim=1)[0].cpu().numpy()
                l2_map = torch.norm(f_orig - f_inp, p=2, dim=1)[0].cpu().numpy()

                f_orig_norm = F.normalize(f_orig, p=2, dim=1)
                f_inp_norm = F.normalize(f_inp, p=2, dim=1)
                cos_sim = (f_orig_norm * f_inp_norm).sum(dim=1)[0].cpu().numpy()
                cos_map = 1 - cos_sim 

                ssim_map = 1 - self.compute_ssim_map(img_orig_np, img_inp_np)
                ssim_map_resized = np.array(Image.fromarray(ssim_map).resize((l2_map.shape[1], l2_map.shape[0]), Image.BILINEAR))

                if accum_maps['L1'] is None:
                    for key in accum_maps:
                        accum_maps[key] = np.zeros_like(l2_map)

                accum_maps['L1'] += l1_map
                accum_maps['L2'] += l2_map
                accum_maps['Cosine'] += cos_map
                accum_maps['SSIM'] += ssim_map_resized
                count += 1

                # Region statistics
                h, w = l2_map.shape
                mask_res = np.array(mask_pil.resize((w, h), Image.NEAREST)) / 255.0
                fg_mask = mask_res > 0.5
                bg_mask = mask_res <= 0.5

                results.append({
                    "Image": os.path.basename(p), "Region": "Foreground",
                    "L1": l1_map[fg_mask].mean(), "L2": l2_map[fg_mask].mean(),
                    "Cosine": cos_map[fg_mask].mean(), "SSIM": ssim_map_resized[fg_mask].mean()
                })
                results.append({
                    "Image": os.path.basename(p), "Region": "Background",
                    "L1": l1_map[bg_mask].mean(), "L2": l2_map[bg_mask].mean(),
                    "Cosine": cos_map[bg_mask].mean(), "SSIM": ssim_map_resized[bg_mask].mean()
                })

            except Exception as e:
                print(f"Skipping {p} due to error: {e}")

        if count > 0:
            avg_maps = {k: v / count for k, v in accum_maps.items()}
            df = pd.DataFrame(results)

            self.plot_metric(df, avg_maps['L1'], 'L1', 'Feature $L_1$ Distance', 'magma')
            self.plot_metric(df, avg_maps['L2'], 'L2', 'Feature $L_2$ Distance', 'magma')
            self.plot_metric(df, avg_maps['Cosine'], 'Cosine', 'Feature Cosine Distance', 'magma')
            self.plot_metric(df, avg_maps['SSIM'], 'SSIM', '1 - SSIM (Dissimilarity)', 'magma')

            csv_path = os.path.join(self.cfg['save_dir'], self.cfg['save_csv'])
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
            self.print_summary(df)
        else:
            print("No images processed successfully.")

    def plot_metric(self, df, avg_map, metric_name, metric_label, cmap):
        _, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.boxplot(x='Region', y=metric_name, data=df, ax=axes[0], palette='coolwarm')
        sns.stripplot(x='Region', y=metric_name, data=df, ax=axes[0], color='black', alpha=0.3, size=3)
        axes[0].set_title(f"{metric_label} by Region ($N={self.cfg['num_images']}$)")
        im = axes[1].imshow(avg_map, cmap=cmap)
        axes[1].set_title(f"Average {metric_name} Heatmap ($N={self.cfg['num_images']}$)")
        plt.colorbar(im, ax=axes[1], label=f"Average {metric_name}")
        plt.tight_layout()
        save_path = os.path.join(self.cfg['save_dir'], f"metric_{metric_name}_analysis.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def print_summary(self, df):
        print("\n" + "=" * 70)
        print(f"=== Selectivity Analysis Summary (N={self.cfg['num_images']}) ===")
        print("=" * 70)
        for metric in ['L1', 'L2', 'Cosine', 'SSIM']:
            df_fg, df_bg = df[df['Region'] == 'Foreground'], df[df['Region'] == 'Background']
            fg_mean, bg_mean = df_fg[metric].mean(), df_bg[metric].mean()
            print(f"\n{metric}:")
            print(f"  Foreground: {fg_mean:.4f} ± {df_fg[metric].std():.4f}")
            print(f"  Background: {bg_mean:.4f} ± {df_bg[metric].std():.4f}")
            print(f"  Gap: {abs(fg_mean - bg_mean):.4f}")
        print("\n" + "=" * 70)

if __name__ == "__main__":
    analyzer = SelectivityAnalysis(CONFIG)
    analyzer.run()