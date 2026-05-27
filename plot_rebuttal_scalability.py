import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, LinearLocator


# Scalability comparison from 100-image quantitative tables and
# 500-image/transferability average tables.
COLORS = ['#ff7f0e', '#1f77b4']
METHODS = ["APT", "APT*"]
DATA_100 = {
    "PSNR": [32.80, 32.80],
    "FR-AUC": [0.97, 0.97], #
    "FR-IoU": [0.90, 0.92], #
    "SP-AUC": [0.99, 0.99],
    "SP-IoU": [0.94, 0.95],
}
DATA_500 = {
    "PSNR": [33.44, 33.44],
    "FR-AUC": [0.97, 0.98], #````
    "FR-IoU": [0.89, 0.91], #
    "SP-AUC": [0.99, 0.99],
    "SP-IoU": [0.96, 0.95],
}


def add_labels(ax, bars, fmt):
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            fmt.format(width),
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=18,
            fontweight="bold",
        )


def main():
    metrics = ["PSNR", "FR-AUC", "FR-IoU"]
    metric_labels = ["PSNR", "FR-\nAUC", "FR-\nIoU"]
    row_gap = 0.66
    y = [idx * row_gap for idx in range(len(metrics))]
    height = 0.18

    series = [
        ("APT* (100)", 1, DATA_100, COLORS[0]),
        ("APT* (500)", 1, DATA_500, COLORS[1]),
    ]
    offsets = [-0.55 * height, 0.55 * height]

    fig, ax_score = plt.subplots(figsize=(6, 5))
    ax_psnr = ax_score.twiny()

    ax_psnr.set_zorder(2)
    ax_score.set_zorder(1)
    ax_psnr.patch.set_alpha(0)

    legend_handles = []
    for (label, method_idx, data, color), offset in zip(series, offsets):
        psnr_pos = y[0] + offset
        psnr_bar = ax_psnr.barh(
            psnr_pos,
            data["PSNR"][method_idx],
            height,
            label=label,
            color=color,
            alpha=0.80,
        )
        add_labels(ax_psnr, psnr_bar, "{:.2f}")
        legend_handles.append(psnr_bar[0])

        metric_positions = [pos + offset for pos in y[1:]]
        metric_values = [data[metric][method_idx] for metric in metrics[1:]]
        score_bars = ax_score.barh(
            metric_positions,
            metric_values,
            height,
            color=color,
            alpha=0.80,
        )
        add_labels(ax_score, score_bars, "{:.2f}")

    ax_psnr.set_xlabel("PSNR", fontsize=24, fontweight="bold")
    ax_psnr.set_xlim(30.0, 35.2)
    ax_psnr.xaxis.set_major_locator(LinearLocator(4))
    ax_psnr.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_psnr.tick_params(axis="x", labelsize=18)

    ax_score.set_xlabel("AUC / IoU", fontsize=24, fontweight="bold")
    ax_score.set_xlim(0.80, 1.02)
    ax_score.xaxis.set_major_locator(FixedLocator([0.80, 0.85, 0.90, 0.95, 1.00]))
    ax_score.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_score.tick_params(axis="x", labelsize=18)

    ax_score.set_ylim(y[-1] + 0.34, -0.75)
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(metric_labels, fontsize=22, fontweight="bold")
    ax_score.tick_params(axis="y", length=0, pad=8)
    ax_score.grid(True, axis="x", alpha=0.30)

    ax_psnr.legend(
        legend_handles,
        [s[0] for s in series],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        prop={"size": 17, "weight": "bold"},
        frameon=True,
        columnspacing=0.8,
        handletextpad=0.45,
        handlelength=1.0,
    )

    plt.tight_layout()
    plt.savefig("fig_rebuttal_scalability.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("fig_rebuttal_scalability.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
