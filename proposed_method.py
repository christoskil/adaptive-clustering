
import numpy as np


class MicroCluster:
    __slots__ = ("centroid", "members", "last_tx_step", "history")

    def __init__(self, centroid, member):
        self.centroid = centroid
        self.members = {member}
        self.last_tx_step = 0
        self.history = {}  # node -> count of recent transmissions


def _merge_micro_clusters(clusters, delta_micro):
    order = sorted(range(len(clusters)), key=lambda i: clusters[i].centroid)
    merged = []
    mapping = {}
    i = 0
    while i < len(order):
        idx = order[i]
        c = clusters[idx]
        members = set(c.members)
        weight = len(c.members)
        weighted_centroid = c.centroid * weight
        last_tx = c.last_tx_step
        group = [idx]
        j = i + 1
        while j < len(order):
            idx2 = order[j]
            c2 = clusters[idx2]
            current_centroid = weighted_centroid / weight
            if abs(c2.centroid - current_centroid) <= delta_micro:
                members |= c2.members
                w2 = len(c2.members)
                weighted_centroid += c2.centroid * w2
                weight += w2
                last_tx = max(last_tx, c2.last_tx_step)
                group.append(idx2)
                j += 1
            else:
                break
        new_idx = len(merged)
        new_c = MicroCluster(weighted_centroid / weight, next(iter(members)))
        new_c.members = members
        new_c.last_tx_step = last_tx
        merged.append(new_c)
        for g in group:
            mapping[g] = new_idx
        i = j
    return merged, mapping


def run_proposed_method(data, positions, energy_model, model_type="simple",
                         delta_micro=0.8, delta_meso=2.5, delta_macro=5.0,
                         beta=0.3, weights=(0.4, 0.3, 0.2, 0.1),
                         t_inactive=10, history_window=10, decay_lambda=5.0,
                         n_steps=None, adaptive_k=None, adaptive_meso_ratio=3.125,
                         adaptive_macro_ratio=6.25, adaptive_min_delta=0.05,
                         enable_merge=False):
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
    delta_micro_trace = []

    for t in range(n_steps):
        readings = data[:, t]

        if adaptive_k is not None:
            prev_readings = data[:, t - 1] if t > 0 else readings
            diffs = readings - prev_readings
            med = np.median(diffs)
            mad = np.median(np.abs(diffs - med))
            robust_std = 1.4826 * mad
            delta_micro = max(adaptive_k * robust_std, adaptive_min_delta)
            delta_meso = delta_micro * adaptive_meso_ratio
            delta_macro = delta_micro * adaptive_macro_ratio
        delta_micro_trace.append(delta_micro)

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

        if enable_merge and len(clusters) > 1:
            clusters, merge_map = _merge_micro_clusters(clusters, delta_micro)
            node_cluster_idx = np.array([merge_map[idx] for idx in node_cluster_idx])

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

        for node in transmitting:
            if model_type == "simple":
                battery[node] -= energy_model.tx_cost_pct()
            else:
                dist_to_bs = np.linalg.norm(positions[node] - positions.mean(axis=0)) + 50.0
                e_tx = energy_model.tx_energy(dist_to_bs)
                battery[node] -= e_tx
            battery[node] = max(battery[node], 0.0)
        total_tx += len(transmitting)

        tx_history[:, tx_ptr] = 0
        for node in transmitting:
            tx_history[node, tx_ptr] = 1
        tx_ptr = (tx_ptr + 1) % history_window

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
        "delta_micro_trace": delta_micro_trace,
    }
