import json
import numpy as np
from network import RadioEnergyModel, generate_rgg_topology
from network_v2 import generate_environmental_data_v2
from proposed_method import run_proposed_method
from baselines import run_naive, run_leach, run_teen

N_SEEDS = 5
N_NODES = 100
N_STEPS = 200
DELTA_MICROS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]

# baseline reference (delta-independent), computed once per seed
baseline_stats = {"leach": [], "teen": [], "naive_tx": []}
cached_data = {}
for seed in range(N_SEEDS):
    s = seed * 17 + 3
    positions, side = generate_rgg_topology(N_NODES, seed=s)
    data, _ = generate_environmental_data_v2(N_NODES, N_STEPS, positions, seed=s)
    cached_data[seed] = (positions, data)
    model = RadioEnergyModel()
    naive = run_naive(data, positions, model, "radio", N_STEPS)
    leach = run_leach(data, positions, model, "radio", n_steps=N_STEPS, seed=s + 1)
    teen = run_teen(data, positions, model, "radio", n_steps=N_STEPS, seed=s + 2)
    base_tx = naive["total_tx"]
    baseline_stats["leach"].append((1 - leach["total_tx"] / base_tx) * 100)
    baseline_stats["teen"].append((1 - teen["total_tx"] / base_tx) * 100)
    baseline_stats["naive_tx"].append(base_tx)

leach_red_mean = np.mean(baseline_stats["leach"])
teen_red_mean = np.mean(baseline_stats["teen"])
print(f"Reference (delta-independent): LEACH reduction={leach_red_mean:.2f}%, TEEN reduction={teen_red_mean:.2f}%")

sweep_results = {}
for dm in DELTA_MICROS:
    reductions, rmses, batteries = [], [], []
    for seed in range(N_SEEDS):
        positions, data = cached_data[seed]
        model = RadioEnergyModel()
        base_tx = baseline_stats["naive_tx"][seed]
        r = run_proposed_method(data, positions, model, "radio", n_steps=N_STEPS,
                                 delta_micro=dm, delta_meso=dm * 3.125, delta_macro=dm * 6.25)
        reductions.append((1 - r["total_tx"] / base_tx) * 100)
        rmses.append(r["rmse"])
        batteries.append(r["final_avg_battery_pct"])
    sweep_results[dm] = {
        "reduction_mean": round(np.mean(reductions), 3),
        "reduction_std": round(np.std(reductions), 3),
        "rmse_mean": round(np.mean(rmses), 4),
        "rmse_std": round(np.std(rmses), 4),
        "battery_mean": round(np.mean(batteries), 3),
    }
    print(f"delta_micro={dm}: reduction={sweep_results[dm]['reduction_mean']:.2f}±{sweep_results[dm]['reduction_std']:.2f}%  "
          f"rmse={sweep_results[dm]['rmse_mean']:.4f}  battery={sweep_results[dm]['battery_mean']:.2f}%")

with open("/home/repo/results/delta_micro_sweep.json", "w") as f:
    json.dump({"leach_reduction_mean": leach_red_mean, "teen_reduction_mean": teen_red_mean,
               "sweep": sweep_results}, f, indent=2)
