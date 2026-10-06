import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.dpi": 300})

RES = "/home/repo/results"
OUTDIR = "/home/latex_src"

with open(f"{RES}/experiment_results.json") as f:
    results = json.load(f)
with open(f"{RES}/delta_micro_sweep_merge.json") as f:
    sweep = json.load(f)
traces = np.load(f"{RES}/traces.npz")

methods = ["leach", "prediction", "teen", "apteen", "compressed_sensing", "proposed"]
labels = ["LEACH", "Prediction", "TEEN", "APTEEN", "Compressed\nSensing", "Ours"]
ms = results["table1_multiseed_radio_10seeds"]

# ---- comparison_chart.png: (a) data reduction (b) RMSE, using 10-seed means ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
reductions = [ms[m]["reduction_mean"] for m in methods]
errs = [ms[m]["reduction_std"] for m in methods]
colors = ["#888888"] * (len(methods) - 1) + ["#1f77b4"]
axes[0].bar(labels, reductions, yerr=errs, capsize=4, color=colors)
axes[0].set_ylabel("Data reduction vs. naive (%)")
axes[0].set_title("(a) Data reduction (mean ± std, 10 seeds)")
axes[0].set_ylim(0, 100)
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(reductions):
    axes[0].text(i, v + 2, f"{v:.1f}", ha="center", fontsize=9)

rmses = [ms[m]["rmse_mean"] for m in methods]
rmse_errs = [ms[m]["rmse_std"] for m in methods]
axes[1].bar(labels, rmses, yerr=rmse_errs, capsize=4, color=colors)
axes[1].set_ylabel("RMSE (reconstruction error)")
axes[1].set_title("(b) Reconstruction RMSE (mean ± std, 10 seeds)")
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(rmses):
    axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/comparison_chart.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- temporal_performance.png: 3-panel dynamics ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(traces["proposed_simple_mse"], color="#1f77b4")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Reconstruction MSE")
axes[0].set_title("(a) Reconstruction error over time")

axes[1].plot(traces["proposed_simple_nclusters"], color="#1f77b4")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Number of active micro-clusters")
axes[1].set_title("(b) Cluster count stabilization")

t = np.arange(len(traces["proposed_radio_battery"]))
axes[2].plot(t, traces["naive_radio_battery"], label="Naive", color="#d62728")
axes[2].plot(t, traces["leach_radio_battery"], label="LEACH", color="#ff7f0e")
axes[2].plot(t, traces["prediction_radio_battery"], label="Prediction", color="#2ca02c")
axes[2].plot(t, traces["proposed_radio_battery"], label="Ours", color="#1f77b4", linewidth=2)
axes[2].set_xlabel("Time step")
axes[2].set_ylabel("Average battery (%)")
axes[2].set_title("(c) Battery evolution (radio energy model)")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/temporal_performance.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- tradeoff_curve.png ----
dm_vals = sorted(float(k) for k in sweep["sweep"].keys())
reduction_vals = [sweep["sweep"][str(dm) if str(dm) in sweep["sweep"] else dm]["reduction_mean"] for dm in dm_vals]
# json keys are strings already since json.dump used float keys -> strings
sweep_sweep = {float(k): v for k, v in sweep["sweep"].items()}
dm_vals = sorted(sweep_sweep.keys())
reduction_vals = [sweep_sweep[dm]["reduction_mean"] for dm in dm_vals]
rmse_vals = [sweep_sweep[dm]["rmse_mean"] for dm in dm_vals]

fig, ax1 = plt.subplots(figsize=(7.5, 5))
color1 = "#1f77b4"
ax1.set_xlabel(r"Micro-cluster tolerance $\Delta_{micro}$")
ax1.set_ylabel("Data reduction (%)", color=color1)
ax1.plot(dm_vals, reduction_vals, "o-", color=color1, label="Data reduction (ours)")
ax1.axhline(sweep["teen_reduction_mean"], color=color1, linestyle="--", alpha=0.5, label="TEEN/APTEEN reduction")
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
color2 = "#d62728"
ax2.set_ylabel("RMSE", color=color2)
ax2.plot(dm_vals, rmse_vals, "s-", color=color2, label="RMSE (ours)")
ax2.axhline(sweep["teen_rmse_mean"], color=color2, linestyle="--", alpha=0.5, label="TEEN/APTEEN RMSE")
ax2.tick_params(axis="y", labelcolor=color2)

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)
plt.title("Reduction-accuracy trade-off vs. $\\Delta_{micro}$ (merge enabled)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/tradeoff_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("Paper figures written to", OUTDIR)
