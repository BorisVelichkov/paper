import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Settings
# =========================

OUTPUT_DIR = "../../results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GRID = "#dddddd"

DPI = 300
FIGSIZE = (12, 5.5)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
})

# =========================
# Data
# =========================

groups = [
    "All Features",
    "Top 1 & Top 2 Features",
    "Top 1 Features"
]

data = {
    "Naive Bayes Stem 1": [0.430, 0.996, 0.993],
    "Naive Bayes Stem 2": [0.411, 0.975, 0.942],
    "Naive Bayes Stem 3": [0.444, 0.955, 0.968],

    "SVM Stem 1": [0.560, 0.925, 0.915],
    "SVM Stem 2": [0.560, 0.935, 0.853],
    "SVM Stem 3": [0.560, 0.914, 0.911],
}

# =========================
# Colors
# =========================

colors = {
    "Naive Bayes Stem 1": BLUE,
    "Naive Bayes Stem 2": "#4f93c6",
    "Naive Bayes Stem 3": "#8bb9dd",

    "SVM Stem 1": ORANGE,
    "SVM Stem 2": "#ff9d3f",
    "SVM Stem 3": "#ffbe73",
}

# =========================
# Bar chart
# =========================

x = np.arange(len(groups))

bar_width = 0.12

fig, ax = plt.subplots(figsize=FIGSIZE)

offsets = np.array([
    -2.5,
    -1.5,
    -0.5,
     0.5,
     1.5,
     2.5
]) * bar_width

for i, (label, values) in enumerate(data.items()):

    ax.bar(
        x + offsets[i],
        values,
        width=bar_width,
        label=label,
        color=colors[label],
        edgecolor="black",
        linewidth=0.6
    )

# =========================
# Layout
# =========================

# ax.set_title(
#     "Средна точност при многократно изпълнявана кръстосана проверка"
# )

# small padding so the x-axis label doesn't sit on the tick labels
ax.set_xlabel("Набор от данни", labelpad=15)
ax.set_ylabel("Точност")

ax.set_xticks(x)
ax.set_xticklabels(groups)

ax.set_ylim(0, 1.08)

ax.grid(
    axis="y",
    color=GRID,
    linewidth=0.8,
    alpha=0.3
)

ax.set_axisbelow(True)

# Reorder legend handles/labels to display row-wise: NB (row 1), SVM (row 2)
# matplotlib fills columns, so we interleave: [NB1, SVM1, NB2, SVM2, NB3, SVM3]
handles, labels = ax.get_legend_handles_labels()
# Reorder indices: [0, 3, 1, 4, 2, 5] → interleave NB and SVM for row-wise display
reordered_indices = [0, 3, 1, 4, 2, 5]
reordered_handles = [handles[i] for i in reordered_indices]
reordered_labels = [labels[i] for i in reordered_indices]

# Legend - 2 rows: Naive Bayes (row 1), SVM (row 2), 3 columns each
ax.legend(
    handles=reordered_handles,
    labels=reordered_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.225),
    ncol=3,
    frameon=False
)

plt.tight_layout()

# =========================
# Save
# =========================

basename = "nb_svm_unstructured_accuracy_10x10cv"

plt.savefig(
    os.path.join(OUTPUT_DIR, f"{basename}.png"),
    dpi=DPI,
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(OUTPUT_DIR, f"{basename}.pdf"),
    dpi=DPI,
    bbox_inches="tight"
)

plt.close()

print("Figure saved successfully.")