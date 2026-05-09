"""
Offline analysis: scatter plots for R3 C3 rebuttal.

  Plot 1: BG complexity (normalized Shannon entropy) vs BG anchor alignment (cos_sim)
  Plot 2: FG-BG semantic similarity (CLIP cosine sim) vs FG anchor alignment

Two modes:
  --mode analyze   Run full analysis (CLIP + entropy) and save results to
                   {save_path}/decoupling_data.json. Also generates the plot.
  --mode plot      Skip analysis; load existing decoupling_data.json and re-plot only.

Usage:
    # Full run
    python -m locmark.analyze_decoupling \
        --save_path /mnt/nas5/suhyeon/projects/apt_rebuttal/ours \
        --mode analyze --device cuda

    # Plot only (tweak figure without re-running CLIP)
    python -m locmark.analyze_decoupling \
        --save_path /mnt/nas5/suhyeon/projects/apt_rebuttal/ours \
        --mode plot
"""
import argparse
import json
import os

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import open_clip


def extract_region_as_pil(tensor, mask, fill_value=0.5):
    """Keep mask=1 pixels, fill mask=0 pixels with neutral gray, return PIL."""
    region = tensor * mask + fill_value * (1 - mask)
    return to_pil_image(region.squeeze(0).clamp(0, 1).cpu())


def normalized_entropy(gray_img: np.ndarray, mask_np: np.ndarray) -> float:
    """Normalized Shannon entropy over grayscale pixels selected by mask."""
    pixels = gray_img[mask_np > 0.5].flatten()
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    hist = hist[hist > 0].astype(float)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log(hist))) / np.log(256)


def run_analysis(args, save_path, data_path):
    device = torch.device(args.device)

    model_size = args.model_size
    mask_side  = model_size // 2
    y0 = x0    = model_size // 4
    center_mask = torch.zeros(1, 1, model_size, model_size)
    center_mask[0, 0, y0:y0 + mask_side, x0:x0 + mask_side] = 1.0
    fg_mask = center_mask
    bg_mask = 1 - center_mask

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    clip_model = clip_model.to(device).eval()

    with open(os.path.join(save_path, 'dist_results.json')) as f:
        per_image = json.load(f)['per_image']

    tensors_dir    = os.path.join(save_path, 'dist_tensors')
    sample_img_dir = os.path.join(save_path, 'complexity_samples')
    os.makedirs(sample_img_dir, exist_ok=True)

    bg_complexity_list = []
    fg_complexity_list = []
    bg_cossim_list     = []
    fg_clip_sim_list   = []
    fg_cossim_list     = []

    for i, img_name in enumerate(tqdm(sorted(per_image.keys()), desc='analyze')):
        stats = per_image[img_name]
        stem  = os.path.splitext(img_name)[0]

        tensors = torch.load(
            os.path.join(tensors_dir, stem + '.pt'),
            map_location='cpu', weights_only=True,
        )
        wm_tensor        = tensors['wm']
        inpainted_tensor = tensors['inpainted']

        # ── BG complexity: normalized Shannon entropy ────────────────────────
        bg_mask_np = bg_mask[0, 0].cpu().numpy()
        fg_mask_np = fg_mask[0, 0].cpu().numpy()

        # use clean image if provided, else fall back to wm_tensor
        if args.clean_path is not None:
            clean_candidates = [
                os.path.join(args.clean_path, stem + ext)
                for ext in ('.png', '.jpg', '.jpeg')
            ]
            clean_file = next((p for p in clean_candidates if os.path.exists(p)), None)
            if clean_file is not None:
                bgr = cv2.imread(clean_file)
                bgr = cv2.resize(bgr, (model_size, model_size))
                complexity_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            else:
                complexity_gray = cv2.cvtColor(
                    (wm_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                    cv2.COLOR_RGB2GRAY,
                )
        else:
            complexity_gray = cv2.cvtColor(
                (wm_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            )

        bg_complexity = normalized_entropy(complexity_gray, bg_mask_np)
        fg_complexity = normalized_entropy(complexity_gray, fg_mask_np)

        if i < 5:
            if args.clean_path is not None and clean_file is not None:
                sample_bgr = cv2.imread(clean_file)
                sample_bgr = cv2.resize(sample_bgr, (model_size, model_size))
                sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
            else:
                sample_rgb = (wm_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            fig_s, ax_s = plt.subplots(figsize=(3, 3))
            ax_s.imshow(sample_rgb)
            ax_s.set_title(f'entropy={bg_complexity:.3f}', fontsize=10)
            ax_s.axis('off')
            fig_s.tight_layout()
            fig_s.savefig(
                os.path.join(sample_img_dir, f'{stem}_entropy{bg_complexity:.3f}.png'),
                dpi=100, bbox_inches='tight',
            )
            plt.close(fig_s)

        bg_complexity_list.append(bg_complexity)
        fg_complexity_list.append(fg_complexity)
        bg_cossim_list.append(stats['inp_outside_cossim'])

        # ── FG-BG CLIP semantic similarity ───────────────────────────────────
        fg_pil = extract_region_as_pil(inpainted_tensor, fg_mask)
        bg_pil = extract_region_as_pil(wm_tensor, bg_mask)

        with torch.no_grad():
            fg_feat = F.normalize(
                clip_model.encode_image(clip_preprocess(fg_pil).unsqueeze(0).to(device)), dim=-1)
            bg_feat = F.normalize(
                clip_model.encode_image(clip_preprocess(bg_pil).unsqueeze(0).to(device)), dim=-1)

        fg_clip_sim_list.append((fg_feat * bg_feat).sum().item())
        fg_cossim_list.append(stats['inp_inside_cossim'])

    data = {
        'bg_complexity': bg_complexity_list,
        'fg_complexity': fg_complexity_list,
        'bg_cossim':     bg_cossim_list,
        'fg_clip_sim':   fg_clip_sim_list,
        'fg_cossim':     fg_cossim_list,
    }
    with open(data_path, 'w') as f:
        json.dump(data, f)
    print(f'Analysis saved → {data_path}')
    return data


def run_plot(data, output_fig):
    from matplotlib.ticker import FormatStrFormatter, LinearLocator, MaxNLocator

    bg_complexity_list = data['bg_complexity']
    fg_complexity_list = data.get('fg_complexity')
    bg_cossim_list     = data['bg_cossim']
    fg_clip_sim_list   = data['fg_clip_sim']
    fg_cossim_list     = data['fg_cossim']

    r1, p1 = pearsonr(bg_complexity_list, bg_cossim_list)
    r2, p2 = pearsonr(fg_clip_sim_list, fg_cossim_list)
    r3, p3 = pearsonr(fg_clip_sim_list, bg_cossim_list)
    print(f'Plot 1  BG complexity  vs BG cos_sim : r={r1:.3f}  p={p1:.3f}')
    if fg_complexity_list is not None:
        r_fg_comp, p_fg_comp = pearsonr(fg_complexity_list, fg_cossim_list)
        print(f'Plot 1  FG complexity  vs FG cos_sim : r={r_fg_comp:.3f}  p={p_fg_comp:.3f}')
    else:
        print('Plot 1  FG complexity missing. Re-run with --mode analyze to save fg_complexity.')
    print(f'Plot 2  FG-BG CLIP sim vs FG cos_sim : r={r2:.3f}  p={p2:.3f}')
    print(f'Plot 2  FG-BG CLIP sim vs BG cos_sim : r={r3:.3f}  p={p3:.3f}')

    margin = 0.01
    y_min = min(min(bg_cossim_list), min(fg_cossim_list)) - margin
    y_max = max(max(bg_cossim_list), max(fg_cossim_list)) + margin

    output_root, output_ext = os.path.splitext(output_fig)
    if output_ext == '':
        output_ext = '.png'
    complexity_fig = output_root + '_complexity' + output_ext
    semantic_fig = output_root + '_semantic' + output_ext

    def save_figure(fig, path, dpi=150):
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved → {path}')

        pdf_path = os.path.splitext(path)[0] + '.pdf'
        if os.path.abspath(pdf_path) != os.path.abspath(path):
            fig.savefig(pdf_path, bbox_inches='tight')
            print(f'Saved → {pdf_path}')

    def plot_complexity(ax, show_ylabel=True):
        ax.scatter(bg_complexity_list, bg_cossim_list,
                   s=70, alpha=0.75, edgecolors='none', color='steelblue', label='Background')
        if fg_complexity_list is not None:
            ax.scatter(fg_complexity_list, fg_cossim_list,
                       s=70, alpha=0.75, edgecolors='none', color='darkorange', label='Foreground')
        ax.set_xlabel('Background\nComplexity', fontsize=29, fontweight='bold')
        if show_ylabel:
            ax.set_ylabel('Cos. Sim.', fontsize=29, fontweight='bold')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='both', labelsize=23)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.grid(True, axis='y', alpha=0.3)

    def plot_semantic(ax, show_ylabel=True):
        ax.scatter(fg_clip_sim_list, bg_cossim_list,
                   s=70, alpha=0.75, edgecolors='none', color='steelblue', label='Background')
        ax.scatter(fg_clip_sim_list, fg_cossim_list,
                   s=70, alpha=0.75, edgecolors='none', color='darkorange', label='Foreground')
        ax.set_xlabel('FG-BG Semantic\nSimilarity', fontsize=29, fontweight='bold')
        if show_ylabel:
            ax.set_ylabel('Cos. Sim.\nw/ Anchor', fontsize=29, fontweight='bold')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(LinearLocator(5))
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='both', labelsize=23)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.grid(True, axis='y', alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    plot_complexity(axes[0], show_ylabel=True)
    plot_semantic(axes[1], show_ylabel=False)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=2,
        frameon=True,
        prop={'size': 26, 'weight': 'bold'},
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.0,
        handletextpad=0.5,
        markerscale=1.8,
    )
    for text in legend.get_texts():
        text.set_fontweight('bold')

    plt.tight_layout(rect=(0, 0, 1, 0.88))
    fig.subplots_adjust(wspace=0.25)
    save_figure(fig, output_fig)
    plt.close(fig)

    # fig_c, ax_c = plt.subplots(figsize=(5, 4))
    # plot_complexity(ax_c)
    # fig_c.tight_layout()
    # save_figure(fig_c, complexity_fig)
    # plt.close(fig_c)

    # fig_s, ax_s = plt.subplots(figsize=(5, 4))
    # plot_semantic(ax_s)
    # fig_s.tight_layout()
    # save_figure(fig_s, semantic_fig)
    # plt.close(fig_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', required=True,
                        help='Directory with dist_results.json and dist_tensors/')
    parser.add_argument('--mode', choices=['analyze', 'plot'], default='analyze',
                        help='"analyze": run full analysis and save; "plot": load saved data and re-plot')
    parser.add_argument('--clean_path', default=None,
                        help='Directory of clean images for complexity measurement')
    parser.add_argument('--output_fig', default=None)
    parser.add_argument('--model_size', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.output_fig is None:
        args.output_fig = os.path.join(args.save_path, 'scatter_decoupling.png')

    data_path = os.path.join(args.save_path, 'decoupling_data.json')

    if args.mode == 'analyze':
        data = run_analysis(args, args.save_path, data_path)
    else:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'No saved data at {data_path}. Run with --mode analyze first.')
        with open(data_path) as f:
            data = json.load(f)
        print(f'Loaded analysis from {data_path}')

    run_plot(data, args.output_fig)


if __name__ == '__main__':
    main()
