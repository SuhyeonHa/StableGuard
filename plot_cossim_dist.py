"""
Plot cosine similarity distributions from dist_results.json.

Usage:
    python plot_cossim_dist.py --json /path/to/dist_results.json [--out /path/to/output.png]

If --out is omitted, saves cossim_dist.png next to the json file.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


KEYS   = ['clean_cossim', 'inp_inside_cossim', 'inp_outside_cossim', 'wm_cossim']
LABELS = ['Clean', 'Foreground', 'Background', 'Perturbed']
COLORS = ['#ff7f0e', '#d62728', '#2ca02c', '#1f77b4']


def plot_dist(json_path: str, out_path: str) -> None:
    with open(json_path) as f:
        summary = json.load(f)

    per_image = summary['per_image']
    data = {k: np.array([v[k] for v in per_image.values()]) for k in KEYS}

    fig, ax = plt.subplots(figsize=(4, 3))

    for key, label, color in zip(KEYS, LABELS, COLORS):
        vals = data[key]

        ax.hist(vals, bins=20, density=True, alpha=0.25, color=color)

        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 300)
        ax.plot(xs, kde(xs), color=color, lw=2,
                label=f'{label} (μ={vals.mean():.3f})')
                # label=f'{label} (μ={vals.mean():.3f}, σ={vals.std():.3f})')
                # label=f'{label}')
        ax.axvline(vals.mean(), color=color, lw=1, linestyle='--', alpha=0.8)

    ax.set_xlabel('Cosine Similarity', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    # ax.set_title('Cossim Distribution')
    ax.legend(fontsize=10, loc='upper left', columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    json_file = "/mnt/nas5/suhyeon/projects/locmark_table_1/ours/hinge-hard-noise-500/20260226-045858/dist_results.json"
    out_file = "fig_cossim_dist.pdf"
    plot_dist(json_file, out_file)


if __name__ == '__main__':
    main()
