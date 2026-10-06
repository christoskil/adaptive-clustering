
import json
import numpy as np
from network import RadioEnergyModel, generate_rgg_topology
from network_v2 import generate_environmental_data_v2
import proposed_method_v3 as pm3
from baselines import run_naive, run_leach, run_teen

STD = 3.239
RAW_DM, RAW_DME, RAW_DMA = 2.0, 6.25, 12.5
N_SEEDS = 5

results = {}
for amp in [0.0, 1.0, 1.5, 3.0, 4.5, 6.0]:
    reds, rmses = [], []
    leach_reds = []
    for seed in [1, 2, 3, 4, 5][:N_SEEDS]:
        positions, side = generate_rgg_topology(100, seed=seed)
        raw_data, _ = generate_environmental_data_v2(100, 200, positions, seed=seed,
                                                        microclimate_amplitude=amp)
        norm_data = (raw_data - raw_data.mean()) / STD
        dm, dme, dma = RAW_DM/STD, RAW_DME/STD, RAW_DMA/STD
        naive = run_naive(raw_data, positions, RadioEnergyModel(), "radio", 200)
        leach = run_leach(raw_data, positions, RadioEnergyModel(), "radio", n_steps=200, seed=seed+1)
        r = pm3.run_proposed_method(norm_data, positions, RadioEnergyModel(), "radio", n_steps=200,
                                      delta_micro=dm, delta_meso=dme, delta_macro=dma, rng_seed=seed,
                                      raw_data=raw_data, denorm_mean=raw_data.mean(), denorm_std=STD)
        base = naive["total_tx"]
        reds.append((1-r["total_tx"]/base)*100); rmses.append(r["rmse"])
        leach_reds.append((1-leach["total_tx"]/base)*100)
    results[amp] = {"reduction_mean": float(np.mean(reds)), "reduction_std": float(np.std(reds)),
                     "rmse_mean": float(np.mean(rmses)), "leach_reduction_mean": float(np.mean(leach_reds))}
    print(f"microclimate_amplitude={amp}: ours reduction={np.mean(reds):.2f}+-{np.std(reds):.2f}%  "
          f"rmse={np.mean(rmses):.3f}  LEACH reduction={np.mean(leach_reds):.2f}%")

with open("/home/repo/results/sensitivity_microclimate.json", "w") as f:
    json.dump(results, f, indent=2)
