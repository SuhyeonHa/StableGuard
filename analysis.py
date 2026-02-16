import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline
import timm

# ==========================================
# 1. Configuration
# ==========================================
ARGS = {
    "src_path": "/mnt/nas5/suhyeon/datasets/valAGE-Set",
    "save_path": "./dot_product_experiment.png",
    "num_images": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "img_size": 512,
    "feat_layer_idx": 3,
    "cache_dir": "/mnt/nas5/suhyeon/caches"
}

torch.manual_seed(ARGS['seed'])
np.random.seed(ARGS['seed'])

# ==========================================
# 2. Models
# ==========================================
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_name, layer_idx=1, device='cuda'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, features_only=True).to(device)
        self.model.eval()
        self.layer_idx = layer_idx
        self.device = device
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def preprocess(self, img_tensor):
        return (img_tensor - self.mean) / self.std

    def forward(self, img_tensor):
        x = self.preprocess(img_tensor)
        features = self.model(x)
        return features[self.layer_idx]

class DotProductExperiment:
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        
        self.extractor = FeatureExtractor(
            model_name='hrnet_w32',
            layer_idx=args['feat_layer_idx'],
            device=self.device
        )
        
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            safety_checker=None,
            cache_dir=args['cache_dir']
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def load_images(self):
        exts = ['*.jpg', '*.png', '*.jpeg']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self.args['src_path'], ext)))
        return sorted(files)[:self.args['num_images']]

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        return img.resize((self.args['img_size'], self.args['img_size']))

    def get_feature_vector(self, img_pil):
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat_map = self.extractor(img_tensor)
            feat_vec = feat_map.mean(dim=[2, 3]) 
        return feat_vec

    def generate_vectors_step_by_step(self, dim):
        # 1. Gaussian (Mean != 0, Magnitude Variance O)
        v1 = torch.randn(1, dim, device=self.device)
        
        # 2. Zero-Mean (Mean == 0, Shift Invariant)
        v2 = v1 - v1.mean(dim=1, keepdim=True)
        
        # 3. Sign (Dense Energy, Max L1 Norm)
        v3 = torch.sign(v2 + 1e-6)
        
        return {
            "1. Gaussian": v1,
            "2. Zero-Mean": v2,
            "3. Sign (Ours)": v3
        }

    def run(self):
        image_paths = self.load_images()
        results = []
        
        print("[Info] Starting Dot Product Analysis...")
        for img_path in tqdm(image_paths):
            try:
                orig_pil = self.preprocess_image(img_path)
                f_orig = self.get_feature_vector(orig_pil)
                dim = f_orig.shape[1]

                mask_pil = Image.new("L", orig_pil.size, 255) 
                regen_pil = self.pipe(prompt="", image=orig_pil, mask_image=mask_pil, num_inference_steps=20).images[0]

                f_regen = self.get_feature_vector(regen_pil)
                
                # Attack Vector (Delta)
                delta = f_regen - f_orig

                # Measure Dot Product
                vectors = self.generate_vectors_step_by_step(dim)
                for step_name, vec in vectors.items():
                    # ========================================================
                    # [핵심 변경] Cosine Similarity -> Dot Product
                    # 내적 = Proj_u(delta) * ||u||
                    # ========================================================
                    dot_prod = torch.sum(delta * vec, dim=1).item()
                    
                    results.append({
                        "Image": os.path.basename(img_path),
                        "Step": step_name,
                        "Dot Product": dot_prod
                    })

            except Exception as e:
                print(f"Error: {e}")
                continue

        return pd.DataFrame(results)

    def plot_results(self, df):
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        # Boxplot으로 분포 비교 (Variance 차이 확인)
        sns.boxplot(x="Step", y="Dot Product", data=df, width=0.5, palette="coolwarm", showfliers=False)
        sns.stripplot(x="Step", y="Dot Product", data=df, color="black", alpha=0.3, size=3)
        
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Projection')
        
        plt.title(f"Dot Product Distribution: Delta vs Anchor (Layer {self.args['feat_layer_idx']})", fontsize=15, fontweight='bold')
        plt.ylabel("Dot Product Value (Projection Energy)", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.args['save_path'], dpi=300)
        print(f"[Success] Plot saved to {self.args['save_path']}")

if __name__ == "__main__":
    experiment = DotProductExperiment(ARGS)
    df_results = experiment.run()
    experiment.plot_results(df_results)
    
    # 분산(Variance)과 평균(Mean) 비교
    print("\n=== Summary Statistics (Mean / Std) ===")
    print(df_results.groupby("Step")["Dot Product"].agg(['mean', 'std', 'min', 'max']))