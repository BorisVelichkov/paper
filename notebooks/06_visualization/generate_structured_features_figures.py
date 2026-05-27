import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Settings
# =========================

DATASET_FILE = "../../data/raw/structured/features_dataset.csv"
OUTPUT_DIR = "../../results/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GRID = "#dddddd"

FIGSIZE_1 = (16, 12)
FIGSIZE_2 = (16, 6)
DPI = 300
SAVE_PDF = True
SHOW_SUPTITLES = False

BASIC_FEATURES_TITLE = "Разпределение на основните характеристики в структурирания набор от данни"
DERIVED_FEATURES_TITLE = "Разпределение на характеристиките, извлечени от интервютата"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.0,
})

# =========================
# Labels and Ordering
# =========================

ORDERS = {
    "Sport": ["Box", "MMA", "Tennis"],
    "Sex": ["Male", "Female"],
    "Final Result": ["YES", "NO"],
    "Prev. Match": ["W", "L", "N"],
    "Health": ["H", "S", "A", "I"],
    "Psychics": ["1", "2", "3", "4", "5"],
    "Confidence": ["1", "2", "3", "4", "5"],
}

LABELS = {
    "Box": "Бокс",
    "MMA": "MMA",
    "Tennis": "Тенис",

    "Male": "Мъж",
    "Female": "Жена",

    "YES": "Победа",
    "NO": "Загуба",

    "W": "Победа",
    "L": "Загуба",
    "N": "Няма",

    "H": "Здрав",
    "S": "Болен",
    "A": "След\nконтузия",
    "I": "Контузен",

    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
}

# =========================
# Helper Functions
# =========================

def ordered_counts(df, column):
    """Frequencies with zero values for missing categories."""

    if column not in df.columns:
        warnings.warn(f"Липсва колона: {column}")
        return None

    s = df[column].dropna().astype(str)

    if column in ORDERS:
        order = [str(x) for x in ORDERS[column]]
        counts = s.value_counts().reindex(order, fill_value=0)
    else:
        counts = s.value_counts().sort_index()

    counts.index = [
        LABELS.get(str(x), str(x))
        for x in counts.index
    ]

    return counts


def add_bar_values(ax, counts):
    """Adds values above the bars."""

    ymax = max(counts.values) if len(counts) else 0

    for patch in ax.patches:

        value = int(patch.get_height())

        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + max(ymax * 0.03, 0.15),
            str(value),
            ha="center",
            va="bottom",
            fontsize=16
        )


def plot_bar(ax, df, column, title, color):
    """Bar chart for one feature."""

    counts = ordered_counts(df, column)

    if counts is None:
        ax.set_title(title)

        ax.text(
            0.5,
            0.5,
            "Липсва колона",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        ax.axis("off")
        return

    ax.bar(
        counts.index,
        counts.values,
        color=color,
        edgecolor="black",
        linewidth=0.7
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Брой")

    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
        alpha=0.3
    )

    ax.set_axisbelow(True)

    ax.tick_params(axis="x", rotation=0)

    ymax = max(counts.values) if len(counts) else 1

    ax.set_ylim(0, ymax * 1.18 + 0.5)

    add_bar_values(ax, counts)


def save_figure(fig, basename, dpi=DPI, save_pdf=False):
    """Save the figure as PNG and optionally as PDF."""

    png_path = os.path.join(OUTPUT_DIR, f"{basename}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    if save_pdf:
        pdf_path = os.path.join(OUTPUT_DIR, f"{basename}.pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", format="pdf")

    return png_path

# =========================
# Loading the Data
# =========================

df = pd.read_csv(DATASET_FILE)

# =========================
# 1. Basic Features
# =========================

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_1)

plot_bar(axes[0, 0], df, "Sport", "Спорт", BLUE)

plot_bar(axes[0, 1], df, "Sex", "Пол", ORANGE)

plot_bar(
    axes[1, 0],
    df,
    "Final Result",
    "Краен резултат",
    BLUE
)

plot_bar(
    axes[1, 1],
    df,
    "Prev. Match",
    "Предишен мач",
    ORANGE
)

if SHOW_SUPTITLES:
    fig.suptitle(
        BASIC_FEATURES_TITLE,
        fontsize=24,
        y=1.05
    )

plt.tight_layout()

save_figure(
    fig,
    "basic_features_distribution",
    save_pdf=SAVE_PDF
)

plt.close()

# =========================
# 2. Features from Interviews
# =========================

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_2)

plot_bar(
    axes[0],
    df,
    "Health",
    "Здравословно състояние",
    BLUE
)

plot_bar(
    axes[1],
    df,
    "Psychics",
    "Психическо състояние",
    ORANGE
)

plot_bar(
    axes[2],
    df,
    "Confidence",
    "Увереност",
    BLUE
)

if SHOW_SUPTITLES:
    fig.suptitle(
        DERIVED_FEATURES_TITLE,
        fontsize=24,
        y=1.05
    )

plt.tight_layout()

save_figure(
    fig,
    "derived_features_distribution",
    save_pdf=SAVE_PDF
)

plt.close()

print(f"Figures saved in: {OUTPUT_DIR}")