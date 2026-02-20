"""
Compute anchor (direction) vectors for LocMark using COCO train2017 images.

Goals:
  1. Compute the mean feature direction across the calibration set.
     ConvNeXt-v2 (GRN) tends to cluster features in a narrow hypersphere region,
     so the "DC direction" of the feature space must be subtracted first.
  2. Generate an anchor vector that is orthogonal to this mean direction,
     ensuring base cosine similarity starts near 0 for all images.

Usage:
  python compute_anchor.py --feat_layer 0 --num_images 5000 --save_path /mnt/nas5/suhyeon/projects/freq-loc/anchor_ortho_128.pt
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ────────────────────────────────────────────────
# Hyperparameters (same as LocMark)
# ────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 256   # same as args.image_size in LocMark
EPSILON       = 1e-6

FEATURE_DIM_MAP = {0: 128, 1: 256, 2: 512, 3: 1024}  # ConvNeXt-v2 stage dims

# ────────────────────────────────────────────────
# Eval dataset filenames to exclude
# (valAGE-Set 폴더 이름 목록을 직접 읽어 제외)
# ────────────────────────────────────────────────
EVAL_DATASET_DIR = '/mnt/nas5/suhyeon/datasets/valAGE-Set'


def norm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] tensor with ImageNet stats (in-place safe)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_coco_paths(coco_dir: str, eval_dir: str, num_images: int) -> list[str]:
    """
    Return up to `num_images` COCO train2017 image paths,
    excluding filenames that appear in the eval dataset.
    """
    # Collect eval filenames (stem-only for flexible matching)
    eval_stems = set()
    if os.path.isdir(eval_dir):
        for fn in os.listdir(eval_dir):
            eval_stems.add(os.path.splitext(fn)[0])
        print(f"[INFO] Excluding {len(eval_stems)} eval images from calibration.")

    all_files = sorted([
        f for f in os.listdir(coco_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    selected = []
    for fn in all_files:
        stem = os.path.splitext(fn)[0]
        if stem not in eval_stems:
            selected.append(os.path.join(coco_dir, fn))
        if len(selected) >= num_images:
            break

    print(f"[INFO] Selected {len(selected)} calibration images from COCO train2017.")
    return selected


@torch.no_grad()
def compute_mean_direction(
    image_paths: list[str],
    encoder: torch.nn.Module,
    feat_layer: int,
    image_size: int,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    Extract features from all calibration images and compute the mean
    unit-direction vector across all spatial patches and all images.

    Returns:
        mean_dir: (feature_dim,) unit vector — the "DC direction" of the feature space.
    """
    transform = load_transform(image_size)
    feature_dim = FEATURE_DIM_MAP[feat_layer]

    accumulator = torch.zeros(feature_dim, device=device)
    total_patches = 0

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(transform(img))
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")
                continue

        if not imgs:
            continue

        batch = torch.stack(imgs, dim=0).to(device)  # [B, 3, H, W]
        batch = norm_imagenet(batch)

        features_list = encoder(batch)
        feat = features_list[feat_layer]  # [B, C, H, W]

        # Same processing as LocMark embed / decode
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).view(B, H * W, C)  # [B, P, C]

        feat_norm = feat / (torch.norm(feat, p=2, dim=-1, keepdim=True) + EPSILON)  # [B, P, C]

        # Accumulate sum over all patches and images
        accumulator += feat_norm.view(-1, C).sum(dim=0)
        total_patches += B * H * W

    mean_dir = accumulator / total_patches          # average direction
    mean_dir = F.normalize(mean_dir, dim=0)         # unit vector
    print(f"[INFO] Mean direction computed from {total_patches:,} patches.")
    return mean_dir


def make_orthogonal_anchor(mean_dir: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """
    Generate a random anchor vector and project out the mean_dir component
    so that the anchor is orthogonal to the DC direction of the feature space.

    This ensures base cosine similarity starts near 0 for any image.

    Returns:
        anchor: (1, feature_dim) unit vector
    """
    torch.manual_seed(seed)
    feature_dim = mean_dir.shape[0]

    # Random init (same strategy as generate_universal_vectors)
    anchor = torch.randn(feature_dim, device=mean_dir.device)
    anchor = anchor - anchor.mean()           # zero-mean
    anchor = torch.sign(anchor + 1e-6)        # sign quantization

    # [핵심] Project out the mean direction component
    # anchor_ortho = anchor - (anchor · mean_dir) * mean_dir
    proj = (anchor @ mean_dir) * mean_dir
    anchor = anchor - proj

    # Re-normalize
    anchor = F.normalize(anchor, dim=0)

    # Verify orthogonality
    cos_with_mean = (anchor @ mean_dir).item()
    print(f"[INFO] Anchor · mean_dir (should be ~0): {cos_with_mean:.6f}")

    return anchor.unsqueeze(0)  # [1, feature_dim]


@torch.no_grad()
def evaluate_anchor(
    anchor: torch.Tensor,
    image_paths: list[str],
    encoder: torch.nn.Module,
    feat_layer: int,
    image_size: int,
    device: torch.device,
    num_eval: int = 200,
    batch_size: int = 16,
) -> None:
    """
    Quick sanity check: compute cosine similarity distribution
    between random images and the anchor vector.
    Target: mean CS close to 0, std should be small.
    """
    transform = load_transform(image_size)
    eval_paths = image_paths[:num_eval]
    all_cs = []

    for i in tqdm(range(0, len(eval_paths), batch_size), desc="Evaluating anchor"):
        batch_paths = eval_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(transform(img))
            except Exception:
                continue

        if not imgs:
            continue

        batch = torch.stack(imgs, dim=0).to(device)
        batch = norm_imagenet(batch)

        features_list = encoder(batch)
        feat = features_list[feat_layer]

        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).view(B, H * W, C)
        feat_mean = feat.mean(dim=1, keepdim=True)
        feat = feat - feat_mean
        feat_norm = feat / (torch.norm(feat, p=2, dim=-1, keepdim=True) + EPSILON)

        cs = torch.matmul(feat_norm, anchor.T).squeeze(-1)  # [B, P]
        all_cs.append(cs.cpu())

    all_cs = torch.cat(all_cs, dim=0).view(-1)
    print(f"\n[Anchor Evaluation on {len(eval_paths)} images]")
    print(f"  Base CS  — mean: {all_cs.mean():.4f} | std: {all_cs.std():.4f} "
          f"| min: {all_cs.min():.4f} | max: {all_cs.max():.4f}")
    print("  (Ideal: mean ≈ 0.0, narrow std)")


def main():
    parser = argparse.ArgumentParser(description="Compute orthogonal anchor vector for LocMark")
    parser.add_argument('--coco_dir',    type=str,
                        default='/mnt/nas5/suhyeon/datasets/coco-2017/train2017')
    parser.add_argument('--eval_dir',    type=str,
                        default=EVAL_DATASET_DIR)
    parser.add_argument('--save_path',   type=str,
                        default='/mnt/nas5/suhyeon/projects/freq-loc/anchor_cvnt2_128.pt')
    parser.add_argument('--feat_layer',  type=int,  default=0,
                        help='ConvNeXt-v2 stage index (0→128, 1→256)')
    parser.add_argument('--num_images',  type=int,  default=100,
                        help='Number of COCO calibration images')
    parser.add_argument('--batch_size',  type=int,  default=16)
    parser.add_argument('--image_size',  type=int,  default=256)
    parser.add_argument('--seed',        type=int,  default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] feat_layer: {args.feat_layer}  feature_dim: {FEATURE_DIM_MAP[args.feat_layer]}")

    # ── 1. Load encoder ───────────────────────────────────────────────────────
    print("[INFO] Loading ConvNeXt-v2...")
    encoder = timm.create_model(
        'convnextv2_base.fcmae_ft_in22k_in1k',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # ── 2. Get calibration image paths ────────────────────────────────────────
    image_paths = get_coco_paths(args.coco_dir, args.eval_dir, args.num_images)

    # ── 3. Compute mean feature direction (DC direction) ─────────────────────
    mean_dir = compute_mean_direction(
        image_paths, encoder, args.feat_layer,
        args.image_size, device, args.batch_size,
    )

    # ── 4. Build orthogonal anchor ────────────────────────────────────────────
    anchor = make_orthogonal_anchor(mean_dir, seed=args.seed)
    print(f"[INFO] Anchor shape: {anchor.shape}")

    # ── 5. Sanity check ───────────────────────────────────────────────────────
    evaluate_anchor(
        anchor, image_paths, encoder, args.feat_layer,
        args.image_size, device, num_eval=300, batch_size=args.batch_size,
    )

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(anchor, args.save_path)
    print(f"\n[INFO] Anchor saved to: {args.save_path}")
    print("[INFO] Update LocMark.__init__ to load this file:")
    print(f"       self.direction_vectors = torch.load('{args.save_path}').to(self.args.device)")

    # ── 7. Save mean_dir as well (useful for debugging / future runs) ─────────
    mean_dir_path = args.save_path.replace('.pt', '_mean_dir.pt')
    torch.save(mean_dir, mean_dir_path)
    print(f"[INFO] Mean direction saved to: {mean_dir_path}")


if __name__ == '__main__':
    main()