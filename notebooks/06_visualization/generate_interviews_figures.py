import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk

# =========================
# Settings
# =========================

DATASET_FILE = "../../data/processed/interviews.csv"
FIGURES_DIR = "../../results/figures"
STATISTICS_DIR = "../../results/statistics"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(STATISTICS_DIR, exist_ok=True)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GRID = "#dddddd"

FIGSIZE = (16, 8)
DPI = 300
SAVE_PDF = True
SHOW_PLOT_TITLES = False
LINEWIDTH = 3.0
MARKERSIZE = 7

TOKENS_AND_SYMBOLS_TITLE = "Брой токени и символи по интервю"
TOKENS_DISTRIBUTION_TITLE = "Разпределение на интервютата по брой токени"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.0,
    "lines.linewidth": LINEWIDTH,
})

# =========================
# Tokenization
# =========================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


def tokenize(text):
    return nltk.word_tokenize(str(text))


def save_figure(fig, basename, dpi=DPI, save_pdf=False):
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    if save_pdf:
        pdf_path = os.path.join(FIGURES_DIR, f"{basename}.pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", format="pdf")

    return png_path


# =========================
# Loading the Data
# =========================

df = pd.read_csv(DATASET_FILE)

possible_text_cols = [
    "text", "Text",
    "interview", "Interview",
    "transcript", "Transcript",
    "content", "Content"
]

text_col = next((c for c in possible_text_cols if c in df.columns), None)

if text_col is None:
    raise ValueError("No text column was found.")

# =========================
# Calculating Tokens and Characters
# =========================

df["tokens"] = df[text_col].fillna("").apply(lambda x: len(tokenize(x)))
df["symbols"] = df[text_col].fillna("").astype(str).str.len()

# =========================
# Statistics
# =========================

stats = {
    "count": int(len(df)),
    "tokens": {
        "mean": round(float(df["tokens"].mean()), 2),
        "median": round(float(df["tokens"].median()), 2),
        "min": int(df["tokens"].min()),
        "max": int(df["tokens"].max())
    },
    "symbols": {
        "mean": round(float(df["symbols"].mean()), 2),
        "median": round(float(df["symbols"].median()), 2),
        "min": int(df["symbols"].min()),
        "max": int(df["symbols"].max())
    }
}

print(json.dumps(stats, ensure_ascii=False, indent=2))

stats_path = os.path.join(STATISTICS_DIR, "interview_length_stats.json")

with open(stats_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# =========================
# 1. Line Plot
# =========================

x = np.arange(len(df))
tick_step = 2
x_ticks = x[::tick_step]

plt.figure(figsize=FIGSIZE)
ax = plt.gca()

ax.plot(
    x,
    df["tokens"],
    color=BLUE,
    marker="o",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="Токени"
)

ax.plot(
    x,
    df["symbols"],
    color=ORANGE,
    marker="s",
    markersize=MARKERSIZE,
    linewidth=LINEWIDTH,
    label="Символи"
)

if SHOW_PLOT_TITLES:
    ax.set_title(TOKENS_AND_SYMBOLS_TITLE)

ax.set_xlabel("Интервю")
ax.set_ylabel("Брой")

max_y = max(df["tokens"].max(), df["symbols"].max())
ax.set_yticks(np.arange(0, max_y + 500, 500))

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks)

ax.margins(x=0.03)
ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.3)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=False)

plt.tight_layout()
save_figure(
    plt.gcf(),
    "tokens_and_symbols",
    save_pdf=SAVE_PDF
)
plt.close()

# =========================
# 2. Token Histogram
# =========================

max_tokens = int(df["tokens"].max())
bins = np.arange(0, max_tokens + 100, 100)

plt.figure(figsize=FIGSIZE)
ax = plt.gca()

ax.hist(
    df["tokens"],
    bins=bins,
    color=BLUE,
    edgecolor="black",
    linewidth=0.8
)

if SHOW_PLOT_TITLES:
    ax.set_title(TOKENS_DISTRIBUTION_TITLE)

ax.set_xlabel("Брой токени")
ax.set_ylabel("Брой интервюта")

ax.set_xticks(bins)
ax.tick_params(axis="x", rotation=45)

ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
save_figure(
    plt.gcf(),
    "tokens_histogram",
    save_pdf=SAVE_PDF
)
plt.close()

print(f"The figures were saved in: {FIGURES_DIR}")
print(f"The statistics were saved in: {STATISTICS_DIR}")