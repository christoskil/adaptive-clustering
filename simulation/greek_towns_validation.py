
import json
import numpy as np
import pandas as pd
import glob
from network import RadioEnergyModel
import proposed_method_v3 as pm3
from baselines import run_naive, run_leach, run_prediction, run_teen, run_apteen, run_compressed_sensing

GREEK_TOWNS_CSV_GLOB = "/home/claude/greekdata/*.csv"  # update to your local path
RAW_DM, RAW_DME, RAW_DMA = 2.0, 6.25, 12.5
N_STEPS = 200
WINDOWS = ["2023-11-01", "2024-02-01", "2024-07-01", "2024-10-15", "2025-01-15", "2025-05-01"]
SEEDS_PER_WINDOW = [0, 1, 2]

# --- load and prepare real data/positions once ---
files = sorted(glob.glob(GREEK_TOWNS_CSV_GLOB))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df['time'] = pd.to_datetime(df['time'])
towns = df[['city', 'latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
n_towns = len(towns)

lat0, lon0 = towns['latitude'].mean(), towns['longitude'].mean()
x_km = (towns['longitude'] - lon0) * 111.32 * np.cos(np.radians(lat0))
y_km = (towns['latitude'] - lat0) * 110.57
positions_km = np.column_stack([x_km, y_km])
span = positions_km.max(axis=0) - positions_km.min(axis=0)
positions = (positions_km - positions_km.min(axis=0)) * (200.0 / span.max())

all_results = {m: {"reduction": [], "rmse": [], "battery": []}
               for m in ["leach", "prediction", "teen", "apteen", "compressed_sensing", "proposed"]}
naive_batteries = []

for w in WINDOWS:
    start = pd.Timestamp(w)
    window = df[(df['time'] >= start) & (df['time'] < start + pd.Timedelta(hours=N_STEPS))]
    pivot = window.pivot_table(index='city', columns='time', values='temperature_2m').loc[towns['city']]
    raw_data = pivot.values
    mean, std = float(raw_data.mean()), float(raw_data.std())
    dm, dme, dma = RAW_DM / std, RAW_DME / std, RAW_DMA / std
    norm_data = (raw_data - mean) / std

    naive = run_naive(raw_data, positions, RadioEnergyModel(), "radio", N_STEPS)
    base = naive["total_tx"]
    naive_batteries.append(naive["final_avg_battery_pct"])

    for seed in SEEDS_PER_WINDOW:
        leach = run_leach(raw_data, positions, RadioEnergyModel(), "radio", n_steps=N_STEPS, seed=seed + 1)
        pred = run_prediction(raw_data, positions, RadioEnergyModel(), "radio", n_steps=N_STEPS)
        teen = run_teen(raw_data, positions, RadioEnergyModel(), "radio", n_steps=N_STEPS, seed=seed + 2)
        apteen = run_apteen(raw_data, positions, RadioEnergyModel(), "radio", n_steps=N_STEPS, seed=seed + 4)
        cs = run_compressed_sensing(raw_data, positions, RadioEnergyModel(), "radio", n_steps=N_STEPS, seed=seed + 3)
        ours = pm3.run_proposed_method(norm_data, positions, RadioEnergyModel(), "radio", n_steps=N_STEPS,
                                        delta_micro=dm, delta_meso=dme, delta_macro=dma, rng_seed=seed,
                                        raw_data=raw_data, denorm_mean=mean, denorm_std=std)

        def red(r): return (1 - r["total_tx"] / base) * 100
        for name, r in [("leach", leach), ("prediction", pred), ("teen", teen),
                        ("apteen", apteen), ("compressed_sensing", cs), ("proposed", ours)]:
            all_results[name]["reduction"].append(red(r))
            all_results[name]["rmse"].append(r["rmse"])
            all_results[name]["battery"].append(r["final_avg_battery_pct"])

summary = {"n_towns": n_towns, "n_windows": len(WINDOWS), "n_seeds_per_window": len(SEEDS_PER_WINDOW),
           "n_trials": len(WINDOWS) * len(SEEDS_PER_WINDOW),
           "naive": {"battery_mean": round(float(np.mean(naive_batteries)), 3)}}
for m, vals in all_results.items():
    summary[m] = {
        "reduction_mean": round(float(np.mean(vals["reduction"])), 3),
        "reduction_std": round(float(np.std(vals["reduction"])), 3),
        "rmse_mean": round(float(np.mean(vals["rmse"])), 4),
        "rmse_std": round(float(np.std(vals["rmse"])), 4),
        "battery_mean": round(float(np.mean(vals["battery"])), 3),
        "battery_std": round(float(np.std(vals["battery"])), 3),
    }

print(json.dumps(summary, indent=2))
with open("/home/claude/repo/results/greek_towns_validation_FINAL.json", "w") as f:
    json.dump(summary, f, indent=2)
np.savez("/home/claude/repo/results/greek_towns_positions.npz", positions=positions, town_names=towns['city'].values,
         positions_km=positions_km)
