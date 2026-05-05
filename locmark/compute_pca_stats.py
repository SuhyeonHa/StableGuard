"""
Compute PCA whitening statistics from ConvNeXt features of natural images.

Saves pca_stats_{feat_dim}.pt to the anchor vectors directory.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m locmark.compute_pca_stats \
        --src_image_path /mnt/nas5/suhyeon/datasets/mirflickr \
        --n 1000 --feat_layer 1
"""
import argparse
import os

import torch
import timm
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image


ANCHOR_DIR = '/mnt/nas5/suhyeon/projects/apt_rebuttal/ours/anchor_vectors'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_image_path', required=True)
    parser.add_argument('--n',          type=int, default=25000, help='Number of images')
    parser.add_argument('--feat_layer', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    feat_dim = {0: 96, 1: 192, 2: 384, 3: 768}[args.feat_layer]
    device   = torch.device(args.device)

    encoder = timm.create_model(
        'convnext_small.dinov3_lvd1689m',
        pretrained=True, features_only=True
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    exts = {'.jpg', '.jpeg', '.png'}
    files = sorted([
        os.path.join(r, f)
        for r, _, fs in os.walk(args.src_image_path)
        for f in fs if os.path.splitext(f)[1].lower() in exts
    ])[:args.n]
    print(f'Found {len(files)} images → extracting features...')

    all_features = []
    with torch.no_grad():
        for path in tqdm(files):
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                continue
            x = tf(img).unsqueeze(0).to(device)
            feat = encoder(x)[args.feat_layer]       # (1, C, H, W)
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(-1, C).cpu()  # (H*W, C)
            all_features.append(feat)

    all_features = torch.cat(all_features, dim=0).float()  # (N_patches, C)
    print(f'Total patches: {all_features.shape[0]}  dim: {all_features.shape[1]}')

    # ── PCA whitening statistics ─────────────────────────────────────────────
    mean = all_features.mean(dim=0)                     # (C,)
    centered = all_features - mean                      # (N, C)

    # SVD on centered features
    # U: (N, C), S: (C,), Vh: (C, C)  [thin SVD]
    print('Computing SVD...')
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # eigenvalues of covariance = S^2 / (N-1)
    explained_std = S / (len(all_features) - 1) ** 0.5  # (C,) — std per PC

    components = Vh  # (C, C): rows are principal components

    save_path = os.path.join(ANCHOR_DIR, f'pca_stats_{feat_dim}.pt')
    torch.save({
        'mean':          mean,           # (C,)
        'components':    components,     # (C, C) rows = eigenvectors
        'explained_std': explained_std,  # (C,) sqrt(eigenvalue)
    }, save_path)

    print(f'\nPCA stats saved → {save_path}')
    print(f'  mean L2 norm       : {mean.norm():.4f}')
    print(f'  explained_std min  : {explained_std.min():.6f}')
    print(f'  explained_std max  : {explained_std.max():.6f}')
    print(f'  explained_std mean : {explained_std.mean():.6f}')


if __name__ == '__main__':
    main()
