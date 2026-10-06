import json
import numpy as np
from network import RadioEnergyModel, SimpleEnergyModel, generate_rgg_topology
from network_v2 import generate_environmental_data_v2
import proposed_method_v3 as pm3
from baselines import run_naive, run_leach, run_prediction, run_teen, run_apteen, run_compressed_sensing

STD = 3.239
RAW_DM, RAW_DME, RAW_DMA = 2.0, 6.25, 12.5
N_SEEDS = 10
N_NODES = 100
N_STEPS = 200

def normalize(raw_data):
    return (raw_data - raw_data.mean()) / STD, raw_data.mean(), STD

def run_one_seed(seed, model_type, energy_model_factory):
    positions, side = generate_rgg_topology(N_NODES, seed=seed)
    raw_data, _ = generate_environmental_data_v2(N_NODES, N_STEPS, positions, seed=seed)
    norm_data, mean, std = normalize(raw_data)
    dm, dme, dma = RAW_DM/std, RAW_DME/std, RAW_DMA/std

    em = energy_model_factory()
    naive = run_naive(raw_data, positions, em, model_type, N_STEPS)
    em = energy_model_factory()
    leach = run_leach(raw_data, positions, em, model_type, n_steps=N_STEPS, seed=seed+1)
    em = energy_model_factory()
    pred = run_prediction(raw_data, positions, em, model_type, n_steps=N_STEPS)
    em = energy_model_factory()
    teen = run_teen(raw_data, positions, em, model_type, n_steps=N_STEPS, seed=seed+2)
    em = energy_model_factory()
    apteen = run_apteen(raw_data, positions, em, model_type, n_steps=N_STEPS, seed=seed+4)
    em = energy_model_factory()
    cs = run_compressed_sensing(raw_data, positions, em, model_type, n_steps=N_STEPS, seed=seed+3)
    em = energy_model_factory()
    proposed = pm3.run_proposed_method(norm_data, positions, em, model_type, n_steps=N_STEPS,
                                        delta_micro=dm, delta_meso=dme, delta_macro=dma, rng_seed=seed,
                                        raw_data=raw_data, denorm_mean=mean, denorm_std=std)

    base = naive["total_tx"]
    def red(r): return (1 - r["total_tx"]/base) * 100
    return {
        "naive": {"battery": naive["final_avg_battery_pct"]},
        "leach": {"reduction": red(leach), "rmse": leach["rmse"], "battery": leach["final_avg_battery_pct"]},
        "prediction": {"reduction": red(pred), "rmse": pred["rmse"], "battery": pred["final_avg_battery_pct"]},
        "teen": {"reduction": red(teen), "rmse": teen["rmse"], "battery": teen["final_avg_battery_pct"]},
        "apteen": {"reduction": red(apteen), "rmse": apteen["rmse"], "battery": apteen["final_avg_battery_pct"]},
        "compressed_sensing": {"reduction": red(cs), "rmse": cs["rmse"], "battery": cs["final_avg_battery_pct"]},
        "proposed": {"reduction": red(proposed), "rmse": proposed["rmse"], "battery": proposed["final_avg_battery_pct"],
                     "avg_clusters": proposed["avg_clusters"]},
    }

def summarize(results):
    methods = ["leach", "prediction", "teen", "apteen", "compressed_sensing", "proposed"]
    out = {}
    for m in methods:
        red = np.array([r[m]["reduction"] for r in results])
        rmse = np.array([r[m]["rmse"] for r in results])
        batt = np.array([r[m]["battery"] for r in results])
        out[m] = {"reduction_mean": round(red.mean(),3), "reduction_std": round(red.std(),3),
                   "rmse_mean": round(rmse.mean(),4), "rmse_std": round(rmse.std(),4),
                   "battery_mean": round(batt.mean(),3), "battery_std": round(batt.std(),3)}
    naive_b = np.array([r["naive"]["battery"] for r in results])
    out["naive"] = {"battery_mean": round(naive_b.mean(),3), "battery_std": round(naive_b.std(),3)}
    if "avg_clusters" in results[0]["proposed"]:
        ac = np.array([r["proposed"]["avg_clusters"] for r in results])
        out["proposed"]["avg_clusters_mean"] = round(ac.mean(), 2)
    return out

if __name__ == "__main__":
    all_out = {}
    for model_type, factory in [("simple", lambda: SimpleEnergyModel(pct_per_tx=0.05)), ("radio", lambda: RadioEnergyModel())]:
        results = [run_one_seed(i*17+3, model_type, factory) for i in range(N_SEEDS)]
        all_out[f"table1_{model_type}"] = summarize(results)
        print(model_type, "done")

    with open("/home/repo/results/table1_v3_FINAL.json", "w") as f:
        json.dump(all_out, f, indent=2)
    print(json.dumps(all_out, indent=2))
