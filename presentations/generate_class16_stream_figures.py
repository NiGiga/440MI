"""Generate the stream-feature figures used in the Class 16 slides (one pasteurization batch)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pasteurization"))
from synth_sensors import simulate_batch  # noqa: E402

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(exist_ok=True)

BLUE = "#12355B"
GOLD = "#C98A2E"
GREEN = "#2F6B4F"
GRAY = "#4B5563"
LIGHT = "#EEF3F8"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelcolor": BLUE,
    "text.color": BLUE,
    "axes.edgecolor": "#CBD5E1",
    "xtick.color": GRAY,
    "ytick.color": GRAY,
})

df, _ = simulate_batch(1)
t = df["timestamp"] - df["timestamp"].iloc[0]
runs = (df["state"] != df["state"].shift()).cumsum()
df["time_in_phase"] = df.groupby(runs).cumcount()
spans = [(t[g.index[0]], t[g.index[-1]], g["state"].iloc[0]) for _, g in df.groupby(runs)]


def shade(ax, labels=False):
    for i, (a, b, name) in enumerate(spans):
        if i % 2 == 0:
            ax.axvspan(a, b, color=LIGHT, zorder=0)
        if labels:
            ax.text((a + b) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=GRAY)


# Rolling mean with several window sizes (trailing, causal).
fig, ax = plt.subplots(figsize=(8, 3.6))
shade(ax, labels=True)
ax.plot(t, df["T"], color="#9CA3AF", linewidth=1.0, label="raw $T_t$")
for w, color in [(5, GOLD), (15, GREEN), (45, BLUE)]:
    ax.plot(t, df["T"].rolling(w, min_periods=1).mean(), color=color, linewidth=1.8,
            label=f"rolling mean, $w={w}$")
ax.set_xlabel("time $t$ [s]")
ax.set_ylabel("temperature [°C]")
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.8))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "class16_rolling_mean_windows.pdf")
plt.close(fig)

# Rate of change.
fig, ax = plt.subplots(figsize=(8, 3.0))
shade(ax, labels=True)
ax.plot(t, df["T"].diff().fillna(0.0), color=GOLD, linewidth=1.0)
ax.axhline(0, color="#CBD5E1", linewidth=0.8)
ax.set_ylabel(r"$\dot{T}_t$ [°C/s]")
ax.set_xlabel("time $t$ [s]")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "class16_rate_of_change.pdf")
plt.close(fig)

# Time in phase.
fig, ax = plt.subplots(figsize=(8, 3.0))
shade(ax, labels=True)
ax.plot(t, df["time_in_phase"], color=BLUE, linewidth=1.5)
ax.set_ylabel("time in phase [s]")
ax.set_xlabel("time $t$ [s]")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "class16_time_in_phase.pdf")
plt.close(fig)
