import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def draw_pfn_block_graph_correct(save_path="pfn_block_graph_correct.jpg"):
    """
    PFN Block Graph (Correct Semantics)
    - Dense block-to-block connections
    """

    layers = [
        {"name": "Input", "blocks": 1},
        {"name": "Conv1", "blocks": 4},
        {"name": "Conv2", "blocks": 6},
        {"name": "Conv3", "blocks": 8},
        {"name": "FC",    "blocks": 1},
    ]

    x_step = 4.8
    block_w = 1.4
    block_h = 0.9
    gap = 0.25

    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    ax.set_facecolor("white")

    block_center = {}

    # =====================
    # 1. Draw partitions & blocks
    # =====================
    for i, layer in enumerate(layers):
        x = i * x_step
        n = layer["blocks"]
        total_h = n * block_h + (n - 1) * gap
        y0 = -total_h / 2

        # Partition container
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.3, y0 - 0.4),
                block_w + 0.6,
                total_h + 0.8,
                boxstyle="round,pad=0.02",
                linewidth=1.2,
                edgecolor="#1A237E",
                facecolor="none",
                linestyle="--",
                zorder=1
            )
        )

        ax.text(
            x + block_w / 2,
            y0 + total_h + 0.6,
            f"{layer['name']} (Partitioned)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

        for b in range(n):
            y = y0 + b * (block_h + gap)
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    block_w,
                    block_h,
                    boxstyle="round,pad=0.02",
                    linewidth=1.0,
                    edgecolor="#283593",
                    facecolor="#E3F2FD",
                    zorder=3
                )
            )

            ax.text(
                x + block_w / 2,
                y + block_h / 2,
                f"B{b}",
                ha="center",
                va="center",
                fontsize=9
            )

            block_center[(i, b)] = (x + block_w, y + block_h / 2)

    # =====================
    # 2. Dense block graph edges
    # =====================
    for i in range(len(layers) - 1):
        for b1 in range(layers[i]["blocks"]):
            for b2 in range(layers[i + 1]["blocks"]):
                p1 = block_center[(i, b1)]
                p2 = block_center[(i + 1, b2)]

                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    color="#333333",
                    lw=0.8,
                    alpha=0.25,
                    zorder=2
                )

    # =====================
    # 4. Layout
    # =====================
    ax.set_xlim(-1.2, (len(layers) - 1) * x_step + 2.2)
    ax.set_ylim(-5.0, 5.0)
    ax.axis("off")

    plt.title(
        "PFN Block Graph",
        fontsize=15,
        fontweight="bold",
        pad=14
    )

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    print(f"[PFN] Correct block-graph saved to {save_path}")


if __name__ == "__main__":
    draw_pfn_block_graph_correct()
