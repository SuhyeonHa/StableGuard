"""
Generate anchor .pt files for ablation study on anchor vector design.
Saves to the same directory as the production anchor, with distinct names.

Production anchor (unchanged): ablation_full_{dim}.pt
New ablation anchors:
  ablation_gaussian_{dim}.pt   — randn → L2 normalize
  ablation_zeromean_{dim}.pt   — randn → zero-mean → L2 normalize
  ablation_quantized_{dim}.pt  — randn → sign → L2 normalize

Usage:
    python -m locmark.generate_ablation_anchors --feature_dim 192 --seed 42
"""
import argparse
import torch


# ANCHOR_DIR = '/mnt/nas5/suhyeon/projects/freq-loc'
ANCHOR_DIR = '/mnt/nas5/suhyeon/projects/apt_rebuttal/ours/anchor_vectors'


def generate(feature_dim, anchor_type, seed):
    torch.manual_seed(seed)
    vecs = torch.randn(1, feature_dim)
    if anchor_type in ('zeromean', 'rademacher'):
        vecs = vecs - vecs.mean(dim=1, keepdim=True)
    if anchor_type in ('quantized', 'rademacher'):
        vecs = torch.sign(vecs + 1e-6)
    vecs = vecs / torch.norm(vecs, p=2, dim=1, keepdim=True)
    return vecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', type=int, default=192)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for anchor_type in ('gaussian', 'zeromean', 'quantized'):
        vec = generate(args.feature_dim, anchor_type, args.seed)
        save_path = f'{ANCHOR_DIR}/ablation_{anchor_type}_{args.feature_dim}.pt'
        torch.save(vec, save_path)
        print(f"Saved {anchor_type}: {save_path}  shape={vec.shape}  norm={vec.norm():.4f}")

    print("\nProduction anchor (not modified):")
    print(f"  {ANCHOR_DIR}/ablation_full_{args.feature_dim}.pt")


if __name__ == '__main__':
    main()
