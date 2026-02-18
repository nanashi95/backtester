"""Generate exposure heatmap from backtest output."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Load data ─────────────────────────────────────────────────────────────────
trades = pd.read_csv("output/trades.csv", parse_dates=["entry_date", "exit_date"])
equity = pd.read_csv("output/equity_curve.csv", parse_dates=["date"], index_col="date")

buckets = ["fx_yen", "equity", "metals", "energy", "softs"]
all_dates = equity.index

# ── Build daily R exposure per bucket ─────────────────────────────────────────
exposure = pd.DataFrame(0.0, index=all_dates, columns=buckets)

for _, t in trades.iterrows():
    if pd.isna(t["entry_date"]) or pd.isna(t["exit_date"]):
        continue
    mask = (all_dates >= t["entry_date"]) & (all_dates < t["exit_date"])
    exposure.loc[mask, t["bucket"]] += t.get("r_multiple", 1.0) / t.get("r_multiple", 1.0)  # 1R per trade

# Add total
exposure["total"] = exposure[buckets].sum(axis=1)

# ── Resample to weekly for cleaner heatmap ────────────────────────────────────
weekly = exposure.resample("W").mean()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(7, 1, figsize=(18, 20), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1, 1]})
fig.suptitle("Exposure Heatmap — Daily R Open by Bucket", fontsize=16, fontweight="bold")

# 1. Stacked area: total exposure by bucket
ax = axes[0]
ax.stackplot(exposure.index,
             exposure["fx_yen"], exposure["equity"],
             exposure["metals"], exposure["energy"], exposure["softs"],
             labels=["FX Yen", "Equity", "Metals", "Energy", "Softs"],
             colors=["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"],
             alpha=0.8)
ax.axhline(6, color="red", linestyle="--", linewidth=1, label="6R Cap")
ax.set_ylabel("R Exposure")
ax.set_ylim(0, 7)
ax.legend(loc="upper left", ncol=6, fontsize=9)
ax.set_title("Stacked R Exposure Over Time")

# 2-6. Individual bucket heatmaps as filled line plots
for i, (bucket, color, cap) in enumerate([
    ("fx_yen", "#4e79a7", 1),
    ("equity", "#f28e2b", 2),
    ("metals", "#59a14f", 2),
    ("energy", "#e15759", 2),
    ("softs",  "#b07aa1", 2),
]):
    ax = axes[i + 1]
    ax.fill_between(exposure.index, exposure[bucket], alpha=0.6, color=color)
    ax.plot(exposure.index, exposure[bucket], color=color, linewidth=0.5)
    ax.axhline(cap, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("R Open")
    ax.set_ylim(0, max(cap + 1, exposure[bucket].max() + 0.5))
    ax.set_title(f"{bucket.replace('_', ' ').title()} (cap: {cap}R)", fontsize=10)

# 7. Total exposure
ax = axes[6]
ax.fill_between(exposure.index, exposure["total"], alpha=0.5, color="#e15759")
ax.plot(exposure.index, exposure["total"], color="#e15759", linewidth=0.5)
ax.axhline(6, color="red", linestyle="--", linewidth=0.8)
ax.set_ylabel("R Open")
ax.set_ylim(0, 7)
ax.set_title("Total Portfolio Exposure", fontsize=10)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("output/exposure_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: output/exposure_heatmap.png")

# ── Also generate a proper month×year heatmap grid ────────────────────────────
fig2, ax2 = plt.subplots(figsize=(16, 5))

monthly_r = exposure["total"].resample("ME").mean()
years = sorted(monthly_r.index.year.unique())
months = range(1, 13)

grid = np.full((len(years), 12), np.nan)
for idx, year in enumerate(years):
    for m in months:
        vals = monthly_r[(monthly_r.index.year == year) & (monthly_r.index.month == m)]
        if not vals.empty:
            grid[idx, m - 1] = vals.iloc[0]

im = ax2.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=6)
ax2.set_yticks(range(len(years)))
ax2.set_yticklabels(years)
ax2.set_xticks(range(12))
ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax2.set_title("Monthly Avg R Exposure — Heatmap Grid", fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax2, label="Avg R Open", shrink=0.8)

# Annotate cells
for i in range(len(years)):
    for j in range(12):
        val = grid[i, j]
        if not np.isnan(val):
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=7, color="black" if val < 3 else "white")

plt.tight_layout()
plt.savefig("output/exposure_heatmap_grid.png", dpi=150, bbox_inches="tight")
print("Saved: output/exposure_heatmap_grid.png")
