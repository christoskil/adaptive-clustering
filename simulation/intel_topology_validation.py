"""
Partial validation on a real-world sensor deployment topology.

The full Intel Berkeley Research Lab readings file (data.txt.gz, ~150MB
uncompressed) could not be retrieved in this environment (see response to
Reviewer 1). As an honest partial step, we import the REAL 54-node mote
layout (mote_locs.txt, from https://db.csail.mit.edu/labdata/labdata.html)
and run our synthetic-but-realistic environmental generator over that real,
irregular office topology instead of our own uniformly-random RGG topology.
This tests whether the method's behaviour (cluster count, data reduction,
reconstruction accuracy) is sensitive to a genuinely non-uniform, clustered
real deployment geometry, which is the topological property the reviewer's
concern is really about; it does NOT substitute for validating against real
temperature/humidity values, which remains an open item (flagged explicitly
in the response letter).
"""
import json
import numpy as np
from network import RadioEnergyModel, SimpleEnergyModel
from network_v2 import generate_environmental_data_v2
from proposed_method import run_proposed_method
from baselines import run_naive, run_leach, run_prediction, run_teen, run_compressed_sensing

positions = []
with open("/home/claude/repo/results/intel_lab_mote_locs.txt") as f:
    for line in f:
        parts = line.split()
        positions.append([float(parts[1]), float(parts[2])])
positions = np.array(positions)  # 54 nodes, real office coordinates in meters

n_nodes = positions.shape[0]
n_steps = 200
data, _ = generate_environmental_data_v2(n_nodes, n_steps, positions, seed=42)

simple_model = SimpleEnergyModel(pct_per_tx=0.05)
radio_model = RadioEnergyModel()

out = {}
for model_type, model in [("simple", simple_model), ("radio", radio_model)]:
    r = {}
    r["naive"] = run_naive(data, positions, model, model_type, n_steps)
    r["leach"] = run_leach(data, positions, model, model_type, n_steps=n_steps, seed=1)
    r["prediction"] = run_prediction(data, positions, model, model_type, n_steps=n_steps)
    r["teen"] = run_teen(data, positions, model, model_type, n_steps=n_steps, seed=2)
    r["proposed"] = run_proposed_method(data, positions, model, model_type, n_steps=n_steps, enable_merge=True)

    base_tx = r["naive"]["total_tx"]
    out[model_type] = {
        k: {
            "total_tx": v["total_tx"],
            "data_reduction_pct": None if k == "naive" else round((1 - v["total_tx"] / base_tx) * 100, 2),
            "rmse": round(v["rmse"], 4),
            "final_avg_battery_pct": round(v["final_avg_battery_pct"], 3),
        } for k, v in r.items()
    }

with open("/home/claude/repo/results/intel_topology_validation.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
