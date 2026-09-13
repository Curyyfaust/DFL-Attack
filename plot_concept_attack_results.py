from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_DIR = Path("concept_attack_figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

results = pd.read_csv("concept_attack_results.csv")
ca = results[results.Attack.isin(["Net-CA", "Grid-CA", "Ramp-CA"])].copy()

fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.40))
x = np.arange(len(ca))
width = 0.24
colors = ["#2878B5", "#F28522", "#7A5195"]
for j, (column, label, color) in enumerate(zip(
    ["dNet_std", "dGrid_std", "dRamp_std"],
    ["Net load", "Network", "Ramping"], colors,
)):
    axes[0].bar(x + (j - 1) * width, ca[column], width=width,
                label=label, color=color)
axes[0].axhline(0, color=".35", lw=.7)
axes[0].set_xticks(x, ["Net", "Grid", "Ramp"])
axes[0].set_ylabel("Standardized change")
axes[0].set_title("Concept selectivity")

cost_rows = results[results.Attack.isin(
    ["FGSM", "PGD", "CB-SPGA", "Net-CA", "Grid-CA", "Ramp-CA"]
)]
bar_colors = ["#9E9E9E", "#777777", "#C82423",
              "#2878B5", "#F28522", "#7A5195"]
yp = np.arange(len(cost_rows))
axes[1].barh(yp, cost_rows["dCost_%"], color=bar_colors, height=.68)
axes[1].axvline(0, color=".35", lw=.7)
axes[1].set_yticks(yp, cost_rows.Attack)
axes[1].invert_yaxis()
axes[1].set_xlabel("Cost change (%)")
axes[1].set_title("Operational consequence")

for ax in axes:
    ax.grid(axis="y" if ax is axes[0] else "x", color=".90", lw=.5)
    ax.set_axisbelow(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .01),
           ncol=3, frameon=False, handlelength=1.2,
           columnspacing=.7, handletextpad=.35)
fig.subplots_adjust(left=.15, right=.99, top=.87, bottom=.30, wspace=.62)
fig.savefig(FIG_DIR / "concept_attack_selectivity.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "concept_attack_selectivity.png", dpi=600,
            bbox_inches="tight")
plt.close(fig)


# Compact selectivity/dominance figure for the three concept attacks.
changes = ca[["dNet_std", "dGrid_std", "dRamp_std"]].to_numpy()
target_idx = np.arange(3)
target = np.abs(changes[target_idx, target_idx])
off = np.abs(changes.copy())
off[target_idx, target_idx] = np.nan
dominance = target / (np.nanmax(off, axis=1) + 1e-12)

fig, ax = plt.subplots(figsize=(3.5, 1.9))
ax.bar(ca.Attack, dominance, color=colors, width=.62)
ax.axhline(1, color=".35", ls="--", lw=.8)
ax.set_ylabel("Target/off-target dominance")
ax.set_title("Target concepts dominate individual off-target changes")
ax.grid(axis="y", color=".90", lw=.5)
ax.set_axisbelow(True)
fig.subplots_adjust(left=.16, right=.99, top=.84, bottom=.24)
fig.savefig(FIG_DIR / "concept_attack_dominance.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "concept_attack_dominance.png", dpi=600,
            bbox_inches="tight")
plt.close(fig)
