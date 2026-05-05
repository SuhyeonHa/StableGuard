"""
Generate anchor .pt files for ablation study on anchor vector design.
Saves to the same directory as the production anchor, with distinct names.

Production anchor (unchanged): ablation_full_{dim}.pt
Ablation anchors:
  ablation_gaussian_{dim}.pt   — randn → L2 normalize
  ablation_zeromean_{dim}.pt   — randn → zero-mean → L2 normalize
  ablation_quantized_{dim}.pt  — randn → sign → L2 normalize
  ablation_rademacher_{dim}.pt — randn → zero-mean → sign → L2 normalize
  ablation_pca_{dim}.pt        — randn → L2 normalize (Gaussian in whitened space)

Usage:
    # generate a specific anchor type (auto-find negative-mean seed for rademacher)
    python -m locmark.generate_ablation_anchors --feature_dim 192 --anchor_type rademacher

    # use a specific seed
    python -m locmark.generate_ablation_anchors --feature_dim 192 --anchor_type pca --seed 1

    # generate all types at once
    python -m locmark.generate_ablation_anchors --feature_dim 192 --anchor_type all --seed 1
"""
import argparse
import torch


# ANCHOR_DIR = '/mnt/nas5/suhyeon/projects/freq-loc'
ANCHOR_DIR = '/mnt/nas5/suhyeon/projects/apt_rebuttal/ours/anchor_vectors'


def generate(feature_dim, anchor_type, seed):
    torch.manual_seed(seed)
    vecs = torch.randn(1, feature_dim)
    if anchor_type in ('pca', 'centering'):
        # isotropic/centered space — plain Gaussian is sufficient
        vecs = vecs / torch.norm(vecs, p=2, dim=1, keepdim=True)
        return vecs
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


ALL_TYPES = ('gaussian', 'zeromean', 'quantized', 'rademacher', 'pca', 'centering')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', type=int, default=192)
    parser.add_argument('--anchor_type', default='rademacher',
                        choices=ALL_TYPES + ('all',),
                        help='Anchor type to generate. Use "all" to generate all types.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed seed. If omitted and anchor_type includes rademacher, '
                             'auto-searches for negative-mean seed.')
    args = parser.parse_args()

    types_to_generate = ALL_TYPES if args.anchor_type == 'all' else (args.anchor_type,)

    if args.seed is None:
        if 'rademacher' in types_to_generate:
            seed = find_negative_mean_seed(args.feature_dim)
        else:
            seed = 0
    else:
        seed = args.seed
    print(f'\nUsing seed={seed}\n')

    print(f"{'type':<12} {'mean':>10} {'std':>8} {'L2':>6}  path")
    print('-' * 80)
    for anchor_type in types_to_generate:
        vec = generate(args.feature_dim, anchor_type, seed)
        save_path = f'{ANCHOR_DIR}/ablation_{anchor_type}_{args.feature_dim}.pt'
        torch.save(vec, save_path)
        print(f"{anchor_type:<12} {vec.mean().item():10.6f} {vec.std().item():8.4f} {vec.norm().item():6.4f}  {save_path}")


if __name__ == '__main__':
    main()
