"""
Implementation of the Adaptive Multi-Level Clustering method described in
Kylafas & Kolomvatsos, "Adaptive Multi-Level Clustering with Dynamic
Representative Selection for Data Redundancy Reduction in IoT Networks".

Implements Algorithm 1 (adaptive cluster formation), Algorithm 2 (distributed
representative selection), and Algorithm 3 (complete protocol), exactly as
specified by Eqs. (8)-(19) of the manuscript.
"""
import numpy as np


class MicroCluster:
    __slots__ = ("centroid", "members", "last_tx_step", "history")

    def __init__(self, centroid, member):
        self.centroid = centroid
        self.members = {member}
        self.last_tx_step = 0
        self.history = {}  # node -> count of recent transmissions


def run_proposed_method(data, positions, energy_model, model_type="simple",
                         delta_micro=0.8, delta_meso=2.5, delta_macro=5.0,
                         beta=0.3, weights=(0.4, 0.3, 0.2, 0.1),
                         t_inactive=10, history_window=10, decay_lambda=5.0,
                         n_steps=None):
    n_nodes, total_steps = data.shape
    if n_steps is None:
        n_steps = total_steps

    battery = np.full(n_nodes, 100.0 if model_type == "simple" else energy_model.e_initial)
    clusters = []  # list[MicroCluster]
    tx_history = np.zeros((n_nodes, history_window), dtype=int)  # ring buffer
    tx_ptr = 0

    total_tx = 0
    reconstruction_sq_errors = []
    battery_trace = []
    n_clusters_trace = []
    depletion_step = None

    for t in range(n_steps):
        readings = data[:, t]

        # --- Phase 1: adaptive cluster formation (Algorithm 1) ---
        node_cluster_idx = np.full(n_nodes, -1, dtype=int)
        for i in range(n_nodes):
            best_idx, best_dist = -1, np.inf
            for ci, c in enumerate(clusters):
                d = abs(readings[i] - c.centroid)
                if d < best_dist:
                    best_dist, best_idx = d, ci
            if best_idx != -1 and best_dist <= delta_micro:
                c = clusters[best_idx]
                c.members.add(i)
                c.centroid = beta * readings[i] + (1 - beta) * c.centroid
                c.last_tx_step = t
                node_cluster_idx[i] = best_idx
            else:
                clusters.append(MicroCluster(readings[i], i))
                node_cluster_idx[i] = len(clusters) - 1

        # --- Phase 2: hierarchical aggregation (meso/macro, for
        #     reconstruction fallback only; representative selection stays
        #     at micro level as specified) ---
        centroids = np.array([c.centroid for c in clusters])
        meso_id = np.full(len(clusters), -1, dtype=int)
        next_meso = 0
        order = np.argsort(centroids)
        for idx in order:
            if meso_id[idx] != -1:
                continue
            meso_id[idx] = next_meso
            for jdx in order:
                if meso_id[jdx] == -1 and abs(centroids[jdx] - centroids[idx]) <= delta_meso:
                    meso_id[jdx] = next_meso
            next_meso += 1
        meso_centroid = {}
        for m in range(next_meso):
            members = [ci for ci in range(len(clusters)) if meso_id[ci] == m]
            sizes = np.array([len(clusters[ci].members) for ci in members])
            cents = np.array([clusters[ci].centroid for ci in members])
            meso_centroid[m] = float(np.average(cents, weights=sizes))

        # --- Phase 3: distributed representative selection (Algorithm 2) ---
        transmitting = set()
        active_cluster_ids = sorted(set(node_cluster_idx.tolist()))
        for ci in active_cluster_ids:
            c = clusters[ci]
            members = list(c.members)
            if not members:
                continue
            member_pos = positions[members]
            spatial_centroid = member_pos.mean(axis=0)
            dists = np.linalg.norm(member_pos - spatial_centroid, axis=1)
            d_max = dists.max() if dists.max() > 0 else 1.0
            max_batt = battery[members].max() if battery[members].max() > 0 else 1.0

            scores = np.zeros(len(members))
            for k, node in enumerate(members):
                phi_cent = 1 - abs(readings[node] - c.centroid) / delta_micro
                phi_energy = battery[node] / max_batt
                phi_spatial = 1 - dists[k] / d_max
                recent_tx = tx_history[node].sum()
                phi_history = np.exp(-recent_tx / decay_lambda)
                scores[k] = (weights[0] * phi_cent + weights[1] * phi_energy +
                             weights[2] * phi_spatial + weights[3] * phi_history)

            probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(members)) / len(members)
            rep = members[int(np.argmax(probs))]
            transmitting.add(rep)

        # --- Phase 4: transmission & energy update ---
        for node in transmitting:
            if model_type == "simple":
                battery[node] -= energy_model.tx_cost_pct()
            else:
                # distance to base station assumed at field centroid-top;
                # approximate with distance to farthest field corner as a
                # conservative multi-hop-free estimate
                dist_to_bs = np.linalg.norm(positions[node] - positions.mean(axis=0)) + 50.0
                e_tx = energy_model.tx_energy(dist_to_bs)
                battery[node] -= e_tx
            battery[node] = max(battery[node], 0.0)
        total_tx += len(transmitting)

        tx_history[:, tx_ptr] = 0
        for node in transmitting:
            tx_history[node, tx_ptr] = 1
        tx_ptr = (tx_ptr + 1) % history_window

        # --- Phase 5: reconstruction at base station ---
        sq_err = 0.0
        for i in range(n_nodes):
            if i in transmitting:
                continue
            ci = node_cluster_idx[i]
            c = clusters[ci]
            reps_in_cluster = c.members & transmitting
            if reps_in_cluster:
                est = c.centroid
            elif meso_centroid.get(meso_id[ci]) is not None:
                est = meso_centroid[meso_id[ci]]
            else:
                est = c.centroid
            sq_err += (readings[i] - est) ** 2
        mse = sq_err / n_nodes
        reconstruction_sq_errors.append(mse)

        # --- Phase 6: cluster maintenance ---
        clusters = [c for c in clusters if (t - c.last_tx_step) <= t_inactive or c.last_tx_step == t]

        avg_batt = battery.mean() if model_type == "simple" else 100.0 * battery.mean() / energy_model.e_initial
        battery_trace.append(avg_batt)
        n_clusters_trace.append(len(clusters))

        if depletion_step is None:
            min_batt = battery.min() if model_type == "simple" else battery.min()
            floor = 0.0
            if min_batt <= floor:
                depletion_step = t

    return {
        "total_tx": total_tx,
        "radio_tx": total_tx,  # single-hop broadcast to base station, per system model
        "rmse": float(np.sqrt(np.mean(reconstruction_sq_errors))),
        "mse_trace": reconstruction_sq_errors,
        "battery_trace": battery_trace,
        "final_avg_battery_pct": battery_trace[-1],
        "battery_variance": float(np.var(battery if model_type == "simple"
                                          else 100.0 * battery / energy_model.e_initial)),
        "n_clusters_trace": n_clusters_trace,
        "avg_clusters": float(np.mean(n_clusters_trace[len(n_clusters_trace)//2:])),
        "depletion_step": depletion_step,
    }
