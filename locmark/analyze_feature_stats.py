"""
Analyze DINOv3-ConvNeXt feature statistics to verify DC bias and channel dominance.
Supports the theoretical motivation for zero-mean centering and Rademacher quantization.

Usage:
    python -m locmark.analyze_feature_stats \
        --src_image_path /mnt/nas5/suhyeon/datasets/valAGE-Set \
        --end_idx 100
"""
import argparse
import torch
import timm
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_image_path', default='/mnt/nas5/suhyeon/datasets/valAGE-Set')
    parser.add_argument('--end_idx', type=int, default=100)
    parser.add_argument('--feat_layer', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    encoder = timm.create_model(
        'convnext_small.dinov3_lvd1689m',
        pretrained=True,
        features_only=True
    ).to(args.device).eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_files = sorted(Path(args.src_image_path).glob('*.png'))[:args.end_idx]
    if not image_files:
        image_files = sorted(Path(args.src_image_path).glob('*.jpg'))[:args.end_idx]

    features_all = []
    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(args.device)
            feats = encoder(x)[args.feat_layer]          # (1, C, H, W)
            feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])  # (H*W, C)
            features_all.append(feats.cpu())

    features_all = torch.cat(features_all, dim=0)   # (N*H*W, C)
    N, feature_dim = features_all.shape

    mean_vec = features_all.mean(dim=0)              # (C,)
    std_per_channel = features_all.std(dim=0)        # (C,)

    # DC Bias: expected norm of sample mean if features were truly zero-mean isotropic
    # Each channel mean ~ N(0, sigma^2/N), so mean_vec norm ~ sigma * sqrt(d/N)
    sigma_avg = std_per_channel.mean().item()
    expected_norm_if_zero_mean = sigma_avg * (feature_dim / N) ** 0.5
    actual_norm = mean_vec.norm().item()
    dc_ratio = actual_norm / (expected_norm_if_zero_mean + 1e-10)

    # Channel Dominance: ratio of max to min std
    std_max = std_per_channel.max().item()
    std_min = std_per_channel.min().item()
    std_ratio = std_max / (std_min + 1e-10)

    # Expected cosine similarity between a raw (non-zero-mean) feature and a Rademacher anchor
    # A Rademacher anchor a has a_i = ±1/√d. E[f·a] = mean_vec · a ≈ 0 if zero-mean, but
    # for raw features: the mean bias contribution is mean_vec · a (random). We estimate the
    # typical magnitude of this bias as ||mean_vec|| / sqrt(d) (projection of mean onto random dir).
    mean_bias_on_random_anchor = actual_norm / (feature_dim ** 0.5)

    print(f"=== DINOv3-ConvNeXt Feature Statistics (layer={args.feat_layer}, dim={feature_dim}) ===")
    print(f"Samples: {N:,} spatial vectors from {len(image_files)} images")
    print()
    print(f"[DC Bias]")
    print(f"  Mean vector L2 norm    : {actual_norm:.4f}")
    print(f"  Expected if zero-mean  : {expected_norm_if_zero_mean:.4f}  (CLT baseline: sigma*sqrt(d/N))")
    print(f"  Actual / Expected ratio: {dc_ratio:.1f}×  (>>1 = significant DC bias)")
    print(f"  Bias on random anchor  : {mean_bias_on_random_anchor:.4f}  (accidental cos_sim from DC alone)")
    print()
    print(f"[Channel Dominance]")
    print(f"  Channel std range      : {std_min:.4f} ~ {std_max:.4f}  (ratio: {std_ratio:.1f}×)")
    print(f"  Channel std mean       : {sigma_avg:.4f}")
    print(f"  Channel std CV         : {(std_per_channel.std() / std_per_channel.mean()):.4f}  (0 = uniform)")
    print()
    print("Interpretation:")
    print(f"  - DC ratio {dc_ratio:.0f}× >> 1: features are strongly biased toward a mean direction")
    print(f"    → zero-mean centering removes this bias from the anchor")
    print(f"  - Std ratio {std_ratio:.0f}×: dominant channels have {std_ratio:.0f}× more variance than weak ones")
    print(f"    → Rademacher equalization prevents dominant channels from causing false alignment")


if __name__ == '__main__':
    main()
