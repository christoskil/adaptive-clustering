import json
import numpy as np
from network import RadioEnergyModel, SimpleEnergyModel, generate_rgg_topology
from network_v2 import generate_environmental_data_v2
from proposed_method import run_proposed_method
from baselines import run_naive, run_leach, run_prediction, run_teen, run_apteen, run_compressed_sensing

OUT = "/home/claude/repo/results"


def run_all(n_nodes, n_steps, model_type, energy_model, seed=0, include_extra=True):
    positions, side = generate_rgg_topology(n_nodes, seed=seed)
    data, _ = generate_environmental_data_v2(n_nodes, n_steps, positions, seed=seed)

    results = {}
    results["naive"] = run_naive(data, positions, energy_model, model_type, n_steps)
    results["leach"] = run_leach(data, positions, energy_model, model_type, n_steps=n_steps, seed=seed + 1)
    results["prediction"] = run_prediction(data, positions, energy_model, model_type, n_steps=n_steps)
    # Official configuration: delta_micro = 0.8 (unchanged from the original
    # submission) WITH cluster merging enabled (Sect. 4.1 cluster-lifecycle
    # behaviour, previously described but not implemented in Algorithm 1).
    results["proposed"] = run_proposed_method(data, positions, energy_model, model_type, n_steps=n_steps,
                                               enable_merge=True)
    if include_extra:
        results["teen"] = run_teen(data, positions, energy_model, model_type, n_steps=n_steps, seed=seed + 2)
        results["apteen"] = run_apteen(data, positions, energy_model, model_type, n_steps=n_steps, seed=seed + 4)
        results["compressed_sensing"] = run_compressed_sensing(data, positions, energy_model, model_type,
                                                                 n_steps=n_steps, seed=seed + 3)
    return results


def summarize(results, baseline_key="naive"):
    base_tx = results[baseline_key]["total_tx"]
    summary = {}
    for k, r in results.items():
        reduction = None if k == baseline_key else (1 - r["total_tx"] / base_tx) * 100
        summary[k] = {
            "total_tx": r["total_tx"],
            "data_reduction_pct": reduction,
            "rmse": round(r["rmse"], 4),
            "final_avg_battery_pct": round(r["final_avg_battery_pct"], 3),
            "battery_variance": round(r["battery_variance"], 5),
            "depletion_step": r["depletion_step"],
        }
    return summary


def run_multiseed_summary(n_nodes, n_steps, model_type, model_factory, n_seeds=10, base_seed=3, seed_stride=17):
    """Mean +/- std over n_seeds independent topologies/datasets -- the
    headline numbers used in the revised manuscript, since a single seed
    was found not to be statistically robust (see response to reviewers)."""
    per_method = {m: {"reduction": [], "rmse": [], "battery": []}
                  for m in ["leach", "prediction", "teen", "apteen", "compressed_sensing", "proposed"]}
    for i in range(n_seeds):
        seed = base_seed + i * seed_stride
        res = run_all(n_nodes, n_steps, model_type, model_factory(), seed=seed, include_extra=True)
        base_tx = res["naive"]["total_tx"]
        for m in per_method:
            per_method[m]["reduction"].append((1 - res[m]["total_tx"] / base_tx) * 100)
            per_method[m]["rmse"].append(res[m]["rmse"])
            per_method[m]["battery"].append(res[m]["final_avg_battery_pct"])
    summary = {}
    for m, vals in per_method.items():
        summary[m] = {
            "reduction_mean": round(float(np.mean(vals["reduction"])), 3),
            "reduction_std": round(float(np.std(vals["reduction"])), 3),
            "rmse_mean": round(float(np.mean(vals["rmse"])), 4),
            "rmse_std": round(float(np.std(vals["rmse"])), 4),
            "battery_mean": round(float(np.mean(vals["battery"])), 3),
            "battery_std": round(float(np.std(vals["battery"])), 3),
        }
    return summary


if __name__ == "__main__":
    all_out = {}

    # --- Experiment A: replicate Table 1 setup with the ORIGINAL simple
    #     energy model (100 nodes, 200 steps) ---
    simple_model = SimpleEnergyModel(pct_per_tx=0.05)
    res_simple = run_all(100, 200, "simple", simple_model, seed=0)
    all_out["table1_simple_model_100n_200t"] = summarize(res_simple)

    # --- Experiment B: same setup with the DISTANCE-DEPENDENT first-order
    #     radio model (responds to Reviewer 1's Eq. 6 comment) ---
    radio_model = RadioEnergyModel()
    res_radio = run_all(100, 200, "radio", radio_model, seed=0)
    all_out["table1_radio_model_100n_200t"] = summarize(res_radio)

    # --- Experiment C: long-run simulation to examine node depletion
    #     patterns beyond the original 200-step window (responds to
    #     Reviewer 1's comment on simulation length / >99% battery) ---
    res_long_simple = run_all(100, 2000, "simple", SimpleEnergyModel(pct_per_tx=0.05), seed=0, include_extra=False)
    all_out["long_run_simple_model_100n_2000t"] = summarize(res_long_simple)

    res_long_radio = run_all(100, 2000, "radio", RadioEnergyModel(), seed=0, include_extra=False)
    all_out["long_run_radio_model_100n_2000t"] = summarize(res_long_radio)

    # --- Experiment D: the DEFINITIVE headline numbers -- mean +/- std over
    #     10 independent seeds, both energy models. A single seed (A/B
    #     above) was found not to be statistically robust; this is what the
    #     revised manuscript's Table 1 should report. ---
    all_out["table1_multiseed_simple_10seeds"] = run_multiseed_summary(
        100, 200, "simple", lambda: SimpleEnergyModel(pct_per_tx=0.05))
    all_out["table1_multiseed_radio_10seeds"] = run_multiseed_summary(
        100, 200, "radio", lambda: RadioEnergyModel())

    with open(f"{OUT}/experiment_results.json", "w") as f:
        json.dump(all_out, f, indent=2)

    # Save raw traces needed for figures
    np.savez(f"{OUT}/traces.npz",
             proposed_simple_battery=res_simple["proposed"]["battery_trace"],
             naive_simple_battery=res_simple["naive"]["battery_trace"],
             leach_simple_battery=res_simple["leach"]["battery_trace"],
             prediction_simple_battery=res_simple["prediction"]["battery_trace"],
             proposed_simple_mse=res_simple["proposed"]["mse_trace"],
             proposed_simple_nclusters=res_simple["proposed"]["n_clusters_trace"],
             proposed_radio_battery=res_radio["proposed"]["battery_trace"],
             naive_radio_battery=res_radio["naive"]["battery_trace"],
             leach_radio_battery=res_radio["leach"]["battery_trace"],
             prediction_radio_battery=res_radio["prediction"]["battery_trace"],
             proposed_long_simple_battery=res_long_simple["proposed"]["battery_trace"],
             naive_long_simple_battery=res_long_simple["naive"]["battery_trace"],
             leach_long_simple_battery=res_long_simple["leach"]["battery_trace"],
             proposed_long_radio_battery=res_long_radio["proposed"]["battery_trace"],
             naive_long_radio_battery=res_long_radio["naive"]["battery_trace"],
             leach_long_radio_battery=res_long_radio["leach"]["battery_trace"],
             )

    print(json.dumps(all_out, indent=2))
