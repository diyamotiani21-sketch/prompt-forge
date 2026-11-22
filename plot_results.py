

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Load benchmark data
s = pd.read_csv("results/summary.csv")

# Global styling
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

background = "#0B1020"
card_color = "#151F35"
text_color = "#F1F5F9"
grid_color = "#34415F"

colors = ["#8B5CF6", "#6366F1", "#06B6D4", "#A78BFA", "#38BDF8"]


def style_axis(ax, title, ylabel):
    ax.set_facecolor(card_color)
    ax.set_title(title, fontsize=21, fontweight="bold",
                 color=text_color, pad=22)
    ax.set_xlabel("Task Type", fontsize=16, color=text_color, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=16, color=text_color, labelpad=12)

    ax.tick_params(axis="both", colors=text_color, labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.25, color=grid_color)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        title="Technique",
        facecolor=card_color,
        edgecolor=grid_color,
        labelcolor=text_color,
        fontsize=12
    )
    legend.get_title().set_color(text_color)


# GRAPH 1: Technique / Quality comparison
p = s.pivot(index="task", columns="technique", values="score")

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(background)

p.plot(
    kind="bar",
    ax=ax,
    color=colors[:len(p.columns)],
    width=0.75
)

style_axis(ax, "Technique Performance Comparison", "Quality Score")

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=12,
                 color=text_color, padding=4)

ax.margins(y=0.15)
plt.tight_layout()
plt.savefig("results/score.png", dpi=300,
            facecolor=background, bbox_inches="tight")
plt.close(fig)


# GRAPH 2: Token cost comparison
p = s.pivot(index="task", columns="technique", values="cost_tokens")

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(background)

p.plot(
    kind="bar",
    ax=ax,
    color=colors[:len(p.columns)],
    width=0.75
)

style_axis(ax, "Token Cost Comparison", "Average Tokens per Call")

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", fontsize=12,
                 color=text_color, padding=4)

ax.margins(y=0.15)
plt.tight_layout()
plt.savefig("results/cost_tokens.png", dpi=300,
            facecolor=background, bbox_inches="tight")
plt.close(fig)


# GRAPH 3: Consistency comparison
p = s.pivot(index="task", columns="technique", values="consistency")

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(background)

p.plot(
    kind="bar",
    ax=ax,
    color=colors[:len(p.columns)],
    width=0.75
)

style_axis(ax, "Consistency Comparison", "Consistency Score")

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=12,
                 color=text_color, padding=4)

ax.margins(y=0.15)
plt.tight_layout()
plt.savefig("results/consistency.png", dpi=300,
            facecolor=background, bbox_inches="tight")
plt.close(fig)

print("All three upgraded PromptForge graphs generated successfully!")
