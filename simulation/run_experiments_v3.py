import json
import numpy as np
from network import RadioEnergyModel, SimpleEnergyModel, generate_rgg_topology
from network_v2 import generate_environmental_data_v2, normalize_data
from proposed_method_v3 import run_proposed_method
from baselines import run_naive, run_leach, run_prediction, run_teen, run_apteen, run_compressed_sensing

N_SEEDS = 10
N_NODES = 100
N_STEPS = 200
DELTA_MICRO, DELTA_MESO, DELTA_MACRO = 0.8, 2.5, 5.0


def run_one_seed(seed, model_type, energy_model):
    positions, side = generate_rgg_topology(N_NODES, seed=seed)
    raw_data, _ = generate_environmental_data_v2(N_NODES, N_STEPS, positions, seed=seed)
    norm_data, dm, dme, dma, mean, std = normalize_data(raw_data, DELTA_MICRO, DELTA_MESO, DELTA_MACRO)


    naive = run_naive(raw_data, positions, energy_model, model_type, N_STEPS)
    leach = run_leach(raw_data, positions, energy_model, model_type, n_steps=N_STEPS, seed=seed + 1)
    pred = run_prediction(raw_data, positions, energy_model, model_type, n_steps=N_STEPS)
    teen = run_teen(raw_data, positions, energy_model, model_type, n_steps=N_STEPS, seed=seed + 2)
    apteen = run_apteen(raw_data, positions, energy_model, model_type, n_steps=N_STEPS, seed=seed + 4)
    cs = run_compressed_sensing(raw_data, positions, energy_model, model_type, n_steps=N_STEPS, seed=seed + 3)
    proposed = run_proposed_method(norm_data, positions, energy_model, model_type, n_steps=N_STEPS,
                                    delta_micro=dm, delta_meso=dme, delta_macro=dma, rng_seed=seed,
                                    raw_data=raw_data, denorm_mean=mean, denorm_std=std)

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


if __name__ == "__main__":
    results = {"simple": [], "radio": []}
    for model_type, model_factory in [("simple", lambda: SimpleEnergyModel(pct_per_tx=0.05)),
                                        ("radio", lambda: RadioEnergyModel())]:
        for seed in range(N_SEEDS):
            r = run_one_seed(seed * 17 + 3, model_type, model_factory())
            results[model_type].append(r)
            print(model_type, seed,
                  {k: (round(v[0], 2), round(v[1], 3), round(v[2], 2)) if isinstance(v, tuple) else round(v, 2)
                   for k, v in r.items()})

    summary = {}
    for model_type in ["simple", "radio"]:
        methods = ["leach", "prediction", "teen", "apteen", "compressed_sensing", "proposed"]
        summary[model_type] = {}
        for m in methods:
            reductions = np.array([r[m][0] for r in results[model_type]])
            rmses = np.array([r[m][1] for r in results[model_type]])
            batteries = np.array([r[m][2] for r in results[model_type]])
            summary[model_type][m] = {
                "reduction_mean": round(reductions.mean(), 3), "reduction_std": round(reductions.std(), 3),
                "rmse_mean": round(rmses.mean(), 4), "rmse_std": round(rmses.std(), 4),
                "battery_mean": round(batteries.mean(), 3), "battery_std": round(batteries.std(), 3),
            }
        naive_batt = np.array([r["naive_battery"] for r in results[model_type]])
        summary[model_type]["naive"] = {"battery_mean": round(naive_batt.mean(), 3), "battery_std": round(naive_batt.std(), 3)}

    with open("/home/repo/results/table1_v3_final.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
