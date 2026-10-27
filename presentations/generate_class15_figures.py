"""Generate data-driven figures used in the Class 15 encoding practice slides."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks" / "Class15_Titanic_Imputed.csv"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#12355B"
GOLD = "#C98A2E"
GREEN = "#2F6B4F"
GRAY = "#4B5563"
LIGHT = "#EEF3F8"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelcolor": BLUE,
    "text.color": BLUE,
    "axes.edgecolor": "#CBD5E1",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

raw = pd.read_csv(DATA)
data = raw.copy()

# A count-and-rate view of category coverage and the univariate diagnostic across all rows.
summary = data.groupby("embarked")["survived"].agg(["count", "mean"]).reindex(["C", "Q", "S"])
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"wspace": 0.48})
ports = ["C", "Q", "S"]
colors = [BLUE, GOLD, GREEN]
y = np.arange(len(ports))
axes[0].barh(y, summary["count"], color=colors, height=0.58)
axes[0].set_yticks(y, [f"{p}  ({int(summary.loc[p, 'count'])})" for p in ports])
axes[0].invert_yaxis()
axes[0].set_xlim(0, 700)
axes[0].set_xlabel("Passengers in dataset")
axes[0].set_title("Category coverage", loc="left")
axes[0].grid(axis="x", color="#E5E7EB", linewidth=0.8)
axes[0].set_axisbelow(True)
for i, p in enumerate(ports):
    axes[0].text(summary.loc[p, "count"] + 8, i, f"{int(summary.loc[p, 'count'])}", va="center", color=GRAY, fontsize=9)

rates = summary["mean"] * 100
axes[1].barh(y, rates, color=colors, height=0.58)
axes[1].set_yticks(y, ports)
axes[1].invert_yaxis()
axes[1].set_xlim(0, 72)
axes[1].set_xlabel("Survived (%)")
axes[1].set_title("Target association", loc="left")
axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)
axes[1].set_axisbelow(True)
for i, p in enumerate(ports):
    axes[1].text(rates.loc[p] + 1.4, i, f"{rates.loc[p]:.1f}%", va="center", color=GRAY, fontsize=9)
fig.text(0.01, 0.01, "All 891 passengers · descriptive rates are not model-performance evidence", color=GRAY, fontsize=8)
fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.20)
fig.savefig(OUT / "class15_embarked_diagnostics.pdf", bbox_inches="tight")
plt.close(fig)

# The same profile PCA used in the activity notebook, with original feature vectors.
data["log_fare"] = np.log1p(data["fare"])
data["family_size"] = data["sibsp"] + data["parch"] + 1
data["is_female"] = (data["sex"] == "female").astype(int)
profile = data.groupby("embarked").agg(
    mean_age=("age", "mean"),
    mean_log_fare=("log_fare", "mean"),
    mean_pclass=("pclass", "mean"),
    mean_family_size=("family_size", "mean"),
    female_share=("is_female", "mean"),
).reindex(["C", "Q", "S"])
features = profile.columns.tolist()
scaled = StandardScaler().fit_transform(profile[features])
pca = PCA(n_components=2)
scores = pca.fit_transform(scaled)
vectors = np.corrcoef(scaled.T, scores.T)[:len(features), len(features):]

fig, ax = plt.subplots(figsize=(7.0, 5.2))
ax.scatter(scores[:, 0], scores[:, 1], s=110, color=BLUE, zorder=4)
for port, (x, yv) in zip(profile.index, scores):
    ax.annotate(port, (x, yv), xytext=(8, 6), textcoords="offset points", fontsize=11, weight="bold", color=BLUE)
span = np.abs(scores).max()
scale = 0.75 * span / np.abs(vectors).max()
labels = {
    "mean_age": "Mean age",
    "mean_log_fare": "Mean log fare",
    "mean_pclass": "Mean class",
    "mean_family_size": "Family size",
    "female_share": "Female share",
}
label_offsets = {
    "mean_age": (-12, 14),
    "mean_log_fare": (-12, -8),
    "mean_pclass": (10, 0),
    "mean_family_size": (-22, -17),
    "female_share": (8, 8),
}
for feature, (vx, vy) in zip(features, vectors):
    x, yv = vx * scale, vy * scale
    ax.annotate("", xy=(x, yv), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 1.6}, zorder=2)
    dx, dy = label_offsets[feature]
    ax.annotate(labels[feature], xy=(x, yv), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, color=BLUE, ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top")
endpoints = vectors * scale
x_limits = [min(scores[:, 0].min(), endpoints[:, 0].min()) - 0.35,
            max(scores[:, 0].max(), endpoints[:, 0].max()) + 0.35]
y_limits = [min(scores[:, 1].min(), endpoints[:, 1].min()) - 0.35,
            max(scores[:, 1].max(), endpoints[:, 1].max()) + 0.35]
ax.set_xlim(x_limits)
ax.set_ylim(y_limits)
ax.axhline(0, color="#D8DEE8", lw=0.8, zorder=0)
ax.axvline(0, color="#D8DEE8", lw=0.8, zorder=0)
ax.set_xlabel(f"PC1 scores ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 scores ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Passenger profiles by embarkation port", loc="left", color=BLUE)
ax.text(0.01, 0.01, "Arrows show feature–PC correlations; their lengths are rescaled for display.",
        transform=ax.transAxes, color=GRAY, fontsize=8, va="bottom")
ax.grid(color="#EEF1F5", lw=0.6)
fig.tight_layout()
fig.savefig(OUT / "class15_port_profile_pca.pdf", bbox_inches="tight")
plt.close(fig)


# Pairwise association heatmaps: Spearman rho for numeric fields, chi-square p-values for categorical fields.
numeric_assoc = data[["age", "fare", "sibsp", "parch", "survived"]]
spearman_matrix = numeric_assoc.corr(method="spearman")

categorical_assoc = ["sex", "pclass", "embarked", "survived"]
pvalue_matrix = np.full((len(categorical_assoc), len(categorical_assoc)), np.nan)
pvalue_labels = np.full((len(categorical_assoc), len(categorical_assoc)), "", dtype=object)
for i, left in enumerate(categorical_assoc):
    for j in range(i + 1, len(categorical_assoc)):
        right = categorical_assoc[j]
        table = pd.crosstab(data[left], data[right])
        _, p_value, _, _ = chi2_contingency(table, correction=False)
        pvalue_matrix[i, j] = pvalue_matrix[j, i] = p_value
        label = "p<.001" if p_value < 0.001 else f"p={p_value:.2f}"
        pvalue_labels[i, j] = pvalue_labels[j, i] = label

# Color represents evidence against independence (-log10 p), not effect size.
evidence = -np.log10(np.maximum(pvalue_matrix, 1e-300))
evidence = np.clip(evidence, 0, 10)
mask = np.triu(np.ones_like(evidence, dtype=bool))
fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.42})
sns.heatmap(
    spearman_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.6, linecolor="white", cbar_kws={"label": "Spearman rho"}, ax=axes[0],
)
axes[0].set_title("Numeric + target: Spearman rho", loc="left", color=BLUE, weight="bold")
axes[0].tick_params(axis="x", rotation=35, labelsize=8)
axes[0].tick_params(axis="y", rotation=0, labelsize=8)
sns.heatmap(
    evidence, mask=mask, annot=pvalue_labels, fmt="", cmap="YlOrBr", vmin=0, vmax=10,
    square=True, linewidths=0.6, linecolor="white", cbar_kws={"label": "-log10(p), clipped at 10"}, ax=axes[1],
)
axes[1].set_xticklabels(categorical_assoc, rotation=35, ha="right", fontsize=8)
axes[1].set_yticklabels(categorical_assoc, rotation=0, fontsize=8)
axes[1].set_title("Categorical pairs: chi-square p", loc="left", color=BLUE, weight="bold")
fig.text(0.01, 0.01, "Different statistics and color scales: Spearman shows signed association; chi-square color shows evidence, not effect size.", color=GRAY, fontsize=8)
fig.subplots_adjust(left=0.05, right=0.98, top=0.87, bottom=0.19)
fig.savefig(OUT / "class15_association_heatmaps.pdf", bbox_inches="tight")
plt.close(fig)
