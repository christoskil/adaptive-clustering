import json
import numpy as np
from network import RadioEnergyModel, SimpleEnergyModel, generate_rgg_topology, generate_environmental_data
from proposed_method import run_proposed_method
from baselines import run_naive, run_leach, run_prediction, run_teen, run_apteen, run_compressed_sensing

N_SEEDS = 10
N_NODES = 100
N_STEPS = 200

def run_one_seed(seed, model_type, energy_model):
    positions, side = generate_rgg_topology(N_NODES, seed=seed)
    data, _ = generate_environmental_data(N_NODES, N_STEPS, seed=seed)

    naive = run_naive(data, positions, energy_model, model_type, N_STEPS)
    leach = run_leach(data, positions, energy_model, model_type, n_steps=N_STEPS, seed=seed + 1)
    pred = run_prediction(data, positions, energy_model, model_type, n_steps=N_STEPS)
    teen = run_teen(data, positions, energy_model, model_type, n_steps=N_STEPS, seed=seed + 2)
    apteen = run_apteen(data, positions, energy_model, model_type, n_steps=N_STEPS, seed=seed + 4)
    cs = run_compressed_sensing(data, positions, energy_model, model_type, n_steps=N_STEPS, seed=seed + 3)
    proposed = run_proposed_method(data, positions, energy_model, model_type, n_steps=N_STEPS)

    base_tx = naive["total_tx"]
    def red(r):
        return (1 - r["total_tx"] / base_tx) * 100

    return {
        "leach": (red(leach), leach["rmse"], leach["final_avg_battery_pct"]),
        "prediction": (red(pred), pred["rmse"], pred["final_avg_battery_pct"]),
        "teen": (red(teen), teen["rmse"], teen["final_avg_battery_pct"]),
        "apteen": (red(apteen), apteen["rmse"], apteen["final_avg_battery_pct"]),
        "compressed_sensing": (red(cs), cs["rmse"], cs["final_avg_battery_pct"]),
        "proposed": (red(proposed), proposed["rmse"], proposed["final_avg_battery_pct"]),
        "naive_battery": naive["final_avg_battery_pct"],
    }

results = {"simple": [], "radio": []}
for model_type, model_factory in [("simple", lambda: SimpleEnergyModel(pct_per_tx=0.05)),
                                    ("radio", lambda: RadioEnergyModel())]:
    for seed in range(N_SEEDS):
        r = run_one_seed(seed * 17 + 3, model_type, model_factory())
        results[model_type].append(r)
        print(model_type, seed, {k: (round(v[0],2) if isinstance(v, tuple) else v) for k,v in r.items() if k != "naive_battery"})

# Aggregate
summary = {}
for model_type in ["simple", "radio"]:
    methods = ["leach", "prediction", "teen", "apteen", "compressed_sensing", "proposed"]
    summary[model_type] = {}
    for m in methods:
        reductions = np.array([r[m][0] for r in results[model_type]])
        rmses = np.array([r[m][1] for r in results[model_type]])
        batteries = np.array([r[m][2] for r in results[model_type]])
        summary[model_type][m] = {
            "reduction_mean": round(reductions.mean(), 3),
            "reduction_std": round(reductions.std(), 3),
            "rmse_mean": round(rmses.mean(), 4),
            "rmse_std": round(rmses.std(), 4),
            "battery_mean": round(batteries.mean(), 3),
            "battery_std": round(batteries.std(), 3),
        }
    naive_batt = np.array([r["naive_battery"] for r in results[model_type]])
    summary[model_type]["naive"] = {"battery_mean": round(naive_batt.mean(),3), "battery_std": round(naive_batt.std(),3)}

with open("/home/claude/repo/results/multiseed_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
