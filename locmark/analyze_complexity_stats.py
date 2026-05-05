"""
Compute normalized Shannon entropy statistics for an image folder.

Entropy is computed on the full image (grayscale).

Usage:
    python -m locmark.analyze_complexity_stats \
        --folder '/mnt/nas3/suhyeon/datasets/ImageNet2012_val' \
        [--n 100] \
        [--model_size 256]
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def normalized_entropy(gray_img: np.ndarray) -> float:
    """Normalized Shannon entropy over all pixels. Returns value in [0, 1]."""
    pixels = gray_img.flatten()
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    hist = hist[hist > 0].astype(float)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log(hist))) / np.log(256)


def folder_stats(folder: str, n: int, model_size: int) -> dict:
    exts = {'.png', '.jpg', '.jpeg'}
    all_files = sorted([
        os.path.join(root, fname)
        for root, _, fnames in os.walk(folder)
        for fname in fnames
        if os.path.splitext(fname)[1].lower() in exts
    ])[:n]

    if not all_files:
        raise FileNotFoundError(f'No images found in {folder}')

    values = []
    for fpath in tqdm(all_files, desc=os.path.basename(folder)):
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.resize(img, (model_size, model_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        values.append(normalized_entropy(gray))

    if not values:
        raise RuntimeError(f'All images failed to load in {folder}')

    return {
        'n':    len(values),
        'min':  float(np.min(values)),
        'max':  float(np.max(values)),
        'mean': float(np.mean(values)),
        'std':  float(np.std(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',     required=True, help='Image folder (searched recursively)')
    parser.add_argument('--n',          type=int, default=100, help='Max images to use')
    parser.add_argument('--model_size', type=int, default=256)
    args = parser.parse_args()

    # ── solid-color reference ────────────────────────────────────────────────
    solid = np.full((args.model_size, args.model_size), 128, dtype=np.uint8)
    solid_entropy = normalized_entropy(solid)
    print(f'\nsolid gray (ref) : entropy = {solid_entropy:.4f}')

    # ── folder stats ─────────────────────────────────────────────────────────
    s = folder_stats(args.folder, args.n, args.model_size)
    print(f'\nFolder : {args.folder}')
    print(f'  N    : {s["n"]}')
    print(f'  min  : {s["min"]:.4f}')
    print(f'  max  : {s["max"]:.4f}')
    print(f'  mean : {s["mean"]:.4f}')
    print(f'  std  : {s["std"]:.4f}')
    print()


if __name__ == '__main__':
    main()
