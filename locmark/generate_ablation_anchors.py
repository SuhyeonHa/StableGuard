"""
Generate anchor .pt files for ablation study on anchor vector design.
Saves to the same directory as the production anchor, with distinct names.

Scans seeds to find one where the Rademacher vector has negative mean,
then generates all four variants from that seed.

Production anchor (unchanged): ablation_full_{dim}.pt
New ablation anchors (all from same seed):
  ablation_gaussian_{dim}.pt   — randn → L2 normalize
  ablation_zeromean_{dim}.pt   — randn → zero-mean → L2 normalize
  ablation_quantized_{dim}.pt  — randn → sign → L2 normalize
  ablation_rademacher_{dim}.pt — randn → zero-mean → sign → L2 normalize

Usage:
    # auto-find seed where rademacher mean < 0
    python -m locmark.generate_ablation_anchors --feature_dim 192

    # use a specific seed
    python -m locmark.generate_ablation_anchors --feature_dim 192 --seed 7
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


def find_negative_mean_seed(feature_dim, max_trials=200):
    """Return first seed where the Rademacher vector has negative mean."""
    print('Scanning seeds for negative Rademacher mean...')
    for seed in range(max_trials):
        vec = generate(feature_dim, 'rademacher', seed)
        m = vec.mean().item()
        if m < 0:
            print(f'  seed={seed}  rademacher mean={m:.6f}  ✓')
            return seed
        print(f'  seed={seed}  rademacher mean={m:.6f}')
    raise RuntimeError(f'No seed with negative Rademacher mean found in {max_trials} trials')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', type=int, default=192)
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed seed. If omitted, auto-searches for negative-mean Rademacher.')
    args = parser.parse_args()

    if args.seed is None:
        seed = find_negative_mean_seed(args.feature_dim)
    else:
        seed = args.seed
    print(f'\nUsing seed={seed}\n')

    print(f"{'type':<12} {'mean':>10} {'std':>8} {'L2':>6}  path")
    print('-' * 80)
    for anchor_type in ('gaussian', 'zeromean', 'quantized', 'rademacher'):
        vec = generate(args.feature_dim, anchor_type, seed)
        save_path = f'{ANCHOR_DIR}/ablation_{anchor_type}_{args.feature_dim}.pt'
        torch.save(vec, save_path)
        print(f"{anchor_type:<12} {vec.mean().item():10.6f} {vec.std().item():8.4f} {vec.norm().item():6.4f}  {save_path}")

    print(f'\nProduction anchor (not modified):')
    print(f'  {ANCHOR_DIR}/ablation_full_{args.feature_dim}.pt')


if __name__ == '__main__':
    main()
