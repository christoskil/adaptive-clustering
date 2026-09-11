import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.dpi": 300})

RES = "/home/christoskil/Downloads/adaptive-clustering/repo/results"
FIG = "/home/christoskil/Downloads/adaptive-clustering/repo/figures"

with open(f"{RES}/experiment_results.json") as f:
    results = json.load(f)
traces = np.load(f"{RES}/traces.npz")

# ---- Figure 5 (re-rendered, high-res): data reduction & RMSE, both energy models ----
methods = ["naive", "leach", "prediction", "teen", "apteen", "compressed_sensing", "proposed"]
labels = ["Naive", "LEACH", "Prediction", "TEEN", "APTEEN", "Compressed\nSensing", "Ours"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, model_key, title in zip(
        axes,
        ["table1_simple_model_100n_200t", "table1_radio_model_100n_200t"],
        ["Fixed-cost energy model", "Distance-dependent radio model"]):
    reductions = [results[model_key][m]["data_reduction_pct"] or 0 for m in methods]
    colors = ["#888888"] * (len(methods) - 1) + ["#1f77b4"]
    ax.bar(labels, reductions, color=colors)
    ax.set_ylabel("Data reduction vs. naive (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(reductions):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_data_reduction_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, model_key, title in zip(
        axes,
        ["table1_simple_model_100n_200t", "table1_radio_model_100n_200t"],
        ["Fixed-cost energy model", "Distance-dependent radio model"]):
    rmses = [results[model_key][m]["rmse"] for m in methods]
    colors = ["#888888"] * (len(methods) - 1) + ["#1f77b4"]
    ax.bar(labels, rmses, color=colors)
    ax.set_ylabel("RMSE (reconstruction error)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(rmses):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG}/fig5b_rmse_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- Figure 6 (re-rendered, high-res): temporal dynamics under the radio model ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

axes[0].plot(traces["proposed_simple_mse"], label="Fixed-cost model", color="#1f77b4")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Reconstruction MSE")
axes[0].set_title("(a) Reconstruction error over time")
axes[0].legend()

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
plt.savefig(f"{FIG}/fig6_temporal_dynamics.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- NEW figure (responds to Reviewer 1's short-simulation comment):
#      long-run node-depletion / survival comparison, radio energy model ----
fig, ax = plt.subplots(figsize=(8, 5))
t_long = np.arange(len(traces["proposed_long_radio_battery"]))
ax.plot(t_long, traces["naive_long_radio_battery"], label="Naive", color="#d62728")
ax.plot(t_long, traces["leach_long_radio_battery"], label="LEACH", color="#ff7f0e")
ax.plot(t_long, traces["proposed_long_radio_battery"], label="Ours", color="#1f77b4", linewidth=2)

naive_dep = results["long_run_radio_model_100n_2000t"]["naive"]["depletion_step"]
leach_dep = results["long_run_radio_model_100n_2000t"]["leach"]["depletion_step"]
if naive_dep:
    ax.axvline(naive_dep, color="#d62728", linestyle="--", alpha=0.6)
    ax.text(naive_dep, 5, f"first node\ndepleted (t={naive_dep})", color="#d62728", fontsize=8, ha="left")
if leach_dep:
    ax.axvline(leach_dep, color="#ff7f0e", linestyle="--", alpha=0.6)
    ax.text(leach_dep, 15, f"first node\ndepleted (t={leach_dep})", color="#ff7f0e", fontsize=8, ha="left")

ax.set_xlabel("Time step (2000-step extended horizon)")
ax.set_ylabel("Average battery (%)")
ax.set_title("Long-run battery evolution under the distance-dependent radio model\n"
             "(100 nodes; first-node-depletion markers show true network lifetime)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig7_long_run_depletion.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figures written to", FIG)
