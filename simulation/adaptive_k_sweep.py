import json
import numpy as np
from network import RadioEnergyModel, generate_rgg_topology
from network_v2 import generate_environmental_data_v2
from proposed_method import run_proposed_method
from baselines import run_naive, run_leach, run_teen

N_SEEDS = 5
N_NODES = 100
N_STEPS = 200
K_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0]

cached_data = {}
naive_tx = {}
leach_red, teen_red, teen_rmse = [], [], []
for seed in range(N_SEEDS):
    s = seed * 17 + 3
    positions, side = generate_rgg_topology(N_NODES, seed=s)
    data, _ = generate_environmental_data_v2(N_NODES, N_STEPS, positions, seed=s)
    cached_data[seed] = (positions, data)
    model = RadioEnergyModel()
    naive = run_naive(data, positions, model, "radio", N_STEPS)
    naive_tx[seed] = naive["total_tx"]
    leach = run_leach(data, positions, model, "radio", n_steps=N_STEPS, seed=s + 1)
    teen = run_teen(data, positions, model, "radio", n_steps=N_STEPS, seed=s + 2)
    leach_red.append((1 - leach["total_tx"] / naive["total_tx"]) * 100)
    teen_red.append((1 - teen["total_tx"] / naive["total_tx"]) * 100)
    teen_rmse.append(teen["rmse"])

print(f"Reference: LEACH reduction={np.mean(leach_red):.2f}%, TEEN reduction={np.mean(teen_red):.2f}%, TEEN RMSE={np.mean(teen_rmse):.4f}")

sweep = {}
for k in K_VALUES:
    reductions, rmses, batteries, dm_means = [], [], [], []
    for seed in range(N_SEEDS):
        positions, data = cached_data[seed]
        model = RadioEnergyModel()
        r = run_proposed_method(data, positions, model, "radio", n_steps=N_STEPS, adaptive_k=k)
        reductions.append((1 - r["total_tx"] / naive_tx[seed]) * 100)
        rmses.append(r["rmse"])
        batteries.append(r["final_avg_battery_pct"])
        dm_means.append(np.mean(r["delta_micro_trace"]))
    sweep[k] = {
        "reduction_mean": round(np.mean(reductions), 3),
        "reduction_std": round(np.std(reductions), 3),
        "rmse_mean": round(np.mean(rmses), 4),
        "rmse_std": round(np.std(rmses), 4),
        "battery_mean": round(np.mean(batteries), 3),
        "avg_delta_micro": round(np.mean(dm_means), 3),
    }
    print(f"k={k}: reduction={sweep[k]['reduction_mean']:.2f}±{sweep[k]['reduction_std']:.2f}%  "
          f"rmse={sweep[k]['rmse_mean']:.4f}  battery={sweep[k]['battery_mean']:.2f}%  "
          f"(avg delta_micro={sweep[k]['avg_delta_micro']:.3f})")

with open("/home/claude/repo/results/adaptive_k_sweep.json", "w") as f:
    json.dump({"leach_reduction_mean": float(np.mean(leach_red)),
               "teen_reduction_mean": float(np.mean(teen_red)),
               "teen_rmse_mean": float(np.mean(teen_rmse)),
               "sweep": sweep}, f, indent=2)
