# R3 hyperparameter sensitivity results.
# Values are from the evaluated LocMark ablation runs.
hnm_x = ["5%", "10%", "15%"]
hnm_psnr = [32.13, 32.80, 33.08]
hnm_sp = [0.95, 0.95, 0.95]
hnm_fr = [0.93, 0.92, 0.92]

noise_x = ["[0,0.10]", "[0,0.25]", "[0,0.50]"]
noise_psnr = [32.79, 32.80, 32.77]
noise_sp = [0.95, 0.95, 0.95]
noise_fr = [0.92, 0.92, 0.92]


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


def plot_with_matplotlib():

    def plot_sensitivity(ax, x, psnr, sp_iou, fr_iou, xlabel, is_first=False, is_last=False):
        ax_psnr = ax
        ln1 = ax_psnr.plot(x, psnr, color="#d62728", marker="o", linestyle=":",
                           linewidth=3, markersize=11, label="PSNR")
        if is_first:
            ax_psnr.set_ylabel("PSNR", color="#d62728", fontsize=30, fontweight="bold")
        ax_psnr.set_ylim(32.0, 33.5)
        ax_psnr.tick_params(axis="both", labelsize=24)
        for label in ax_psnr.get_xticklabels() + ax_psnr.get_yticklabels():
            label.set_fontweight("bold")

        ax_iou = ax.twinx()
        ln2 = ax_iou.plot(x, sp_iou, color="#1f77b4", marker="s",
                          linewidth=3, markersize=11, label="IoU (SP)")
        ln3 = ax_iou.plot(x, fr_iou, color="#2ca02c", marker="^",
                          linewidth=3, markersize=12, label="IoU (FR)")
        ax_iou.set_ylim(0.85, 1.0)
        ax_iou.tick_params(axis="y", labelsize=24)
        for label in ax_iou.get_yticklabels():
            label.set_fontweight("bold")
        if is_last:
            ax_iou.set_ylabel("IoU Score", fontsize=30, fontweight="bold")

        ax.set_xlabel(xlabel, fontsize=30, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        ax_psnr.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_iou.yaxis.set_major_locator(MaxNLocator(nbins=5))
        return ln1, ln2, ln3

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    lns = plot_sensitivity(axes[0], hnm_x, hnm_psnr, hnm_sp, hnm_fr, "HNM\nratio $k$", is_first=True)
    plot_sensitivity(axes[1], noise_x, noise_psnr, noise_sp, noise_fr, "Noise\nrange $\\sigma$", is_last=True)

    handles = lns[0] + lns[1] + lns[2]
    labels = [h.get_label() for h in handles]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        prop={"size": 26, "weight": "bold"},
        frameon=True,
        columnspacing=1.0,
        handletextpad=0.5,
        markerscale=1.35,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.86))
    plt.savefig("fig_rebuttal_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("fig_rebuttal_sensitivity.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    plot_with_matplotlib()
    print("Saved fig_rebuttal_sensitivity.pdf and fig_rebuttal_sensitivity.png")
