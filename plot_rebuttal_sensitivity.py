# R3 hyperparameter sensitivity results.
# Values are from the evaluated LocMark ablation runs.
hnm_x = ["5%", "10%", "20%"]
hnm_psnr = [32.13, 32.80, 33.25]
hnm_sp = [0.95, 0.95, 0.96]
hnm_fr = [0.93, 0.92, 0.92]

noise_x = ["[0,0.10]", "[0,0.25]", "[0,0.50]"]
noise_psnr = [32.79, 32.80, 32.77]
noise_sp = [0.95, 0.95, 0.95]
noise_fr = [0.92, 0.92, 0.92]


def plot_with_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    def plot_sensitivity(ax, x, psnr, sp_iou, fr_iou, xlabel, is_first=False, is_last=False):
        ax_psnr = ax
        ln1 = ax_psnr.plot(x, psnr, color="#d62728", marker="o", linestyle=":", linewidth=2, label="PSNR")
        if is_first:
            ax_psnr.set_ylabel("PSNR", color="#d62728", fontsize=24, fontweight="bold")
        ax_psnr.set_ylim(32.0, 33.5)
        ax_psnr.tick_params(axis="both", labelsize=18)

        ax_iou = ax.twinx()
        ln2 = ax_iou.plot(x, sp_iou, color="#1f77b4", marker="s", linewidth=2, label="IoU (SP)")
        ln3 = ax_iou.plot(x, fr_iou, color="#2ca02c", marker="^", linewidth=2, label="IoU (FR)")
        ax_iou.set_ylim(0.85, 1.0)
        ax_iou.tick_params(axis="y", labelsize=18)
        if is_last:
            ax_iou.set_ylabel("IoU Score", fontsize=24, fontweight="bold")

        ax.set_xlabel(xlabel, fontsize=24)
        ax.grid(True, axis="y", alpha=0.3)

        ax_psnr.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_iou.yaxis.set_major_locator(MaxNLocator(nbins=5))
        return ln1, ln2, ln3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lns1 = plot_sensitivity(axes[0], hnm_x, hnm_psnr, hnm_sp, hnm_fr, r"HNM ratio $k$", is_first=True)
    plot_sensitivity(axes[1], noise_x, noise_psnr, noise_sp, noise_fr, r"Noise range $\sigma$", is_last=True)

    handles = lns1[0] + lns1[1] + lns1[2]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), fontsize=22)

    plt.tight_layout()
    plt.savefig("fig_rebuttal_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("fig_rebuttal_sensitivity.png", dpi=300, bbox_inches="tight")


def plot_with_svg():
    width, height = 1200, 500
    margin = 70
    plot_w, plot_h = 430, 300
    gap = 110
    top = 55
    left1 = margin
    left2 = margin + plot_w + gap
    psnr_min, psnr_max = 32.0, 33.5
    iou_min, iou_max = 0.85, 1.0

    def sx(left, idx, n):
        return left + idx * (plot_w / (n - 1))

    def sy(value, lo, hi):
        return top + plot_h - ((value - lo) / (hi - lo)) * plot_h

    def polyline(left, values, lo, hi):
        return " ".join(f"{sx(left, i, len(values)):.1f},{sy(v, lo, hi):.1f}" for i, v in enumerate(values))

    def marker(kind, x, y, color):
        if kind == "circle":
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
        if kind == "square":
            return f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" fill="{color}"/>'
        return f'<polygon points="{x:.1f},{y-6:.1f} {x-6:.1f},{y+5:.1f} {x+6:.1f},{y+5:.1f}" fill="{color}"/>'

    def panel(left, xlabels, psnr, sp, fr, xlabel, y_left=False, y_right=False):
        items = []
        items.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#222" stroke-width="1"/>')
        for t in [0.85, 0.90, 0.95, 1.00]:
            y = sy(t, iou_min, iou_max)
            items.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
            if y_right:
                items.append(f'<text x="{left+plot_w+10}" y="{y+6:.1f}" font-size="18">{t:.2f}</text>')
        for t in [32.0, 32.5, 33.0, 33.5]:
            y = sy(t, psnr_min, psnr_max)
            if y_left:
                items.append(f'<text x="{left-48}" y="{y+6:.1f}" font-size="18" fill="#d62728">{t:.1f}</text>')

        items.append(f'<polyline points="{polyline(left, psnr, psnr_min, psnr_max)}" fill="none" stroke="#d62728" stroke-width="3" stroke-dasharray="4 6"/>')
        items.append(f'<polyline points="{polyline(left, sp, iou_min, iou_max)}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
        items.append(f'<polyline points="{polyline(left, fr, iou_min, iou_max)}" fill="none" stroke="#2ca02c" stroke-width="3"/>')

        for i, v in enumerate(psnr):
            items.append(marker("circle", sx(left, i, len(psnr)), sy(v, psnr_min, psnr_max), "#d62728"))
        for i, v in enumerate(sp):
            items.append(marker("square", sx(left, i, len(sp)), sy(v, iou_min, iou_max), "#1f77b4"))
        for i, v in enumerate(fr):
            items.append(marker("triangle", sx(left, i, len(fr)), sy(v, iou_min, iou_max), "#2ca02c"))

        for i, label in enumerate(xlabels):
            items.append(f'<text x="{sx(left, i, len(xlabels)):.1f}" y="{top+plot_h+35}" text-anchor="middle" font-size="20">{label}</text>')
        items.append(f'<text x="{left+plot_w/2}" y="{top+plot_h+78}" text-anchor="middle" font-size="26">{xlabel}</text>')
        return "\n".join(items)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        panel(left1, hnm_x, hnm_psnr, hnm_sp, hnm_fr, "HNM ratio k", y_left=True),
        panel(left2, noise_x, noise_psnr, noise_sp, noise_fr, "Noise range sigma", y_right=True),
        '<text x="30" y="220" transform="rotate(-90 30 220)" font-size="26" fill="#d62728" font-weight="bold">PSNR</text>',
        '<text x="1160" y="220" transform="rotate(90 1160 220)" font-size="26" font-weight="bold">IoU Score</text>',
        '<line x1="340" y1="455" x2="390" y2="455" stroke="#d62728" stroke-width="3" stroke-dasharray="4 6"/><circle cx="365" cy="455" r="5" fill="#d62728"/><text x="400" y="462" font-size="22">PSNR</text>',
        '<line x1="520" y1="455" x2="570" y2="455" stroke="#1f77b4" stroke-width="3"/><rect x="540" y="450" width="10" height="10" fill="#1f77b4"/><text x="580" y="462" font-size="22">IoU (SP)</text>',
        '<line x1="720" y1="455" x2="770" y2="455" stroke="#2ca02c" stroke-width="3"/><polygon points="745,449 739,460 751,460" fill="#2ca02c"/><text x="780" y="462" font-size="22">IoU (FR)</text>',
        "</svg>",
    ]
    with open("fig_rebuttal_sensitivity.svg", "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


if __name__ == "__main__":
    try:
        plot_with_matplotlib()
        print("Saved fig_rebuttal_sensitivity.pdf and fig_rebuttal_sensitivity.png")
    except ModuleNotFoundError:
        plot_with_svg()
        print("matplotlib not found; saved fig_rebuttal_sensitivity.svg")
