
import numpy as np
from baselines import base_station_position

FULL_REPORT_BITS = 2000       # micro representative full report (value, centroid, variance, member ids)
DELTA_REPORT_BITS = 200       # micro representative delta-encoded report (E8): header + small delta value only
MESO_SUMMARY_BITS = 2000      # meso representative summary report (coarser, not richer, than a micro report)
MACRO_SUMMARY_BITS = 2000     # macro representative summary report (coarser, not richer, than a micro report)
DOWNLINK_BEACON_BITS = 200    # base-station -> all-nodes beacon carrying known cluster centroids (E12-14)

DELTA_ENCODE_THRESHOLD_RATIO = 0.5   # fraction of delta_micro; report is "delta" if value moved less than this since last report
SPLIT_STD_RATIO = 2.0                 # a cluster splits if its current-member std exceeds SPLIT_STD_RATIO * delta_micro
STABILITY_WINDOW = 5                  # consecutive stable rounds before a meso/macro cluster's reporting interval grows
MAX_REPORT_INTERVAL = 8               # cap on how infrequently a stable meso/macro cluster reports


class ClusterIdentity:
    __slots__ = ("centroid", "last_rep_tx_step", "last_reported_value")

    def __init__(self, centroid, step):
        self.centroid = centroid
        self.last_rep_tx_step = step        # last step a REPRESENTATIVE of this cluster transmitted (E23)
        self.last_reported_value = centroid  # for delta encoding (E8)


class MesoMacroTracker:

    __slots__ = ("last_centroid", "stable_rounds", "report_interval", "last_report_step")

    def __init__(self, centroid, step):
        self.last_centroid = centroid
        self.stable_rounds = 0
        self.report_interval = 1
        self.last_report_step = step - 1  # force a report the first time it is seen

    def update_and_should_report(self, new_centroid, step, delta_scale):
        moved = abs(new_centroid - self.last_centroid)
        if moved < 0.1 * delta_scale:
            self.stable_rounds += 1
        else:
            self.stable_rounds = 0
            self.report_interval = 1
        if self.stable_rounds >= STABILITY_WINDOW:
            self.report_interval = min(MAX_REPORT_INTERVAL, 1 + self.stable_rounds // STABILITY_WINDOW)
        self.last_centroid = new_centroid
        due = (step - self.last_report_step) >= self.report_interval
        if due:
            self.last_report_step = step
        return due


def _build_adjacency(positions, r_c):

    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    return d <= r_c


def _split_oversized_clusters(clusters, current_members_map, readings, delta_micro, step):

    next_id = (max(clusters.keys()) + 1) if clusters else 0
    node_cluster_idx = {}
    for cid, members in list(current_members_map.items()):
        if len(members) < 4:
            for m in members:
                node_cluster_idx[m] = cid
            continue
        vals = readings[members]
        std = vals.std()
        if std <= SPLIT_STD_RATIO * delta_micro:
            for m in members:
                node_cluster_idx[m] = cid
            continue
        median = np.median(vals)
        low = [m for m in members if readings[m] <= median]
        high = [m for m in members if readings[m] > median]
        if not low or not high:
            for m in members:
                node_cluster_idx[m] = cid
            continue
        clusters[cid] = ClusterIdentity(float(np.mean(readings[low])), step)
        for m in low:
            node_cluster_idx[m] = cid
        new_id = next_id
        next_id += 1
        clusters[new_id] = ClusterIdentity(float(np.mean(readings[high])), step)
        for m in high:
            node_cluster_idx[m] = new_id
    return clusters, node_cluster_idx


def run_proposed_method(data, positions, energy_model, model_type="simple",
                         delta_micro=0.8, delta_meso=2.5, delta_macro=5.0,
                         beta=0.3, weights=(0.4, 0.3, 0.2, 0.1),
                         t_inactive=10, history_window=10, decay_lambda=5.0,
                         n_steps=None, enable_merge=True, r_c=35.0,
                         rng_seed=0, raw_data=None, denorm_mean=0.0, denorm_std=1.0):

    rng = np.random.default_rng(rng_seed)
    n_nodes, total_steps = data.shape
    if n_steps is None:
        n_steps = total_steps
    if raw_data is None:
        raw_data = data

    battery = np.full(n_nodes, 100.0 if model_type == "simple" else energy_model.e_initial)
    adjacency = _build_adjacency(positions, r_c)
    base_station = base_station_position(positions)

    clusters = {}          # cluster_id -> ClusterIdentity (persistent)
    meso_trackers = {}     # meso key -> MesoMacroTracker
    macro_trackers = {}    # macro key -> MesoMacroTracker
    tx_history = np.zeros((n_nodes, history_window), dtype=int)
    tx_ptr = 0

    total_tx = 0            # messages reaching the base station (micro + meso + macro reports)
    reconstruction_sq_errors = []
    battery_trace = []
    n_clusters_trace = []
    depletion_step = None

    def charge(node, bits, distance):
        if model_type == "simple":
            battery[node] -= energy_model.tx_cost_pct()
        else:
            e = energy_model.tx_energy_bits(distance, bits)
            battery[node] -= e
        battery[node] = max(battery[node], 0.0)

    def charge_rx(node, bits):
        if model_type != "simple":
            battery[node] -= energy_model.rx_energy_bits(bits)
            battery[node] = max(battery[node], 0.0)

    for t in range(n_steps):
        readings = data[:, t]

        node_cluster_idx = {}
        cluster_ids_snapshot = list(clusters.keys())
        centroid_arr = np.array([clusters[c].centroid for c in cluster_ids_snapshot]) if cluster_ids_snapshot else np.array([])
        for i in range(n_nodes):
            best_cid = -1
            if centroid_arr.size:
                dists = np.abs(readings[i] - centroid_arr)
                j = int(np.argmin(dists))
                if dists[j] <= delta_micro:
                    best_cid = cluster_ids_snapshot[j]
            if best_cid == -1:
                new_id = (max(clusters.keys()) + 1) if clusters else 0
                clusters[new_id] = ClusterIdentity(readings[i], t)
                cluster_ids_snapshot.append(new_id)
                centroid_arr = np.append(centroid_arr, readings[i])
                best_cid = new_id
            node_cluster_idx[i] = best_cid

        current_members_map = {}
        for i, cid in node_cluster_idx.items():
            current_members_map.setdefault(cid, []).append(i)
        for cid, members in current_members_map.items():
            mean_val = float(np.mean(readings[members]))
            clusters[cid].centroid = beta * mean_val + (1 - beta) * clusters[cid].centroid


        clusters, node_cluster_idx = _split_oversized_clusters(clusters, current_members_map, readings, delta_micro, t)
        current_members_map = {}
        for i, cid in node_cluster_idx.items():
            current_members_map.setdefault(cid, []).append(i)

        # === Phase 1c: optional merge (ablation kept; default True) ===
        if enable_merge and len(current_members_map) > 1:
            ordered = sorted(current_members_map.keys(), key=lambda c: clusters[c].centroid)
            merge_target = {}
            i = 0
            while i < len(ordered):
                base_cid = ordered[i]
                acc_members = list(current_members_map[base_cid])
                j = i + 1
                while j < len(ordered) and abs(clusters[ordered[j]].centroid - clusters[base_cid].centroid) <= delta_micro:
                    acc_members += current_members_map[ordered[j]]
                    merge_target[ordered[j]] = base_cid
                    j += 1
                merge_target[base_cid] = base_cid
                clusters[base_cid].centroid = float(np.mean(readings[acc_members]))
                current_members_map[base_cid] = acc_members
                i = j
            for cid in list(current_members_map.keys()):
                if merge_target.get(cid, cid) != cid:
                    del current_members_map[cid]
            node_cluster_idx = {i: merge_target.get(cid, cid) for i, cid in node_cluster_idx.items()}

        active_cids = sorted(current_members_map.keys(), key=lambda c: clusters[c].centroid)
        micro_centroids = np.array([clusters[c].centroid for c in active_cids])
        meso_of_micro = {}
        meso_centroid = {}
        m = 0
        idx = 0
        while idx < len(active_cids):
            meso_of_micro[active_cids[idx]] = m
            group = [active_cids[idx]]
            idx2 = idx + 1
            while idx2 < len(active_cids) and abs(micro_centroids[idx2] - micro_centroids[idx]) <= delta_meso:
                meso_of_micro[active_cids[idx2]] = m
                group.append(active_cids[idx2])
                idx2 += 1
            sizes = np.array([len(current_members_map[c]) for c in group])
            cents = np.array([clusters[c].centroid for c in group])
            meso_centroid[m] = float(np.average(cents, weights=sizes))
            m += 1
            idx = idx2

        meso_ids_sorted = sorted(meso_centroid.keys(), key=lambda k: meso_centroid[k])
        macro_of_meso = {}
        macro_centroid = {}
        M = 0
        idx = 0
        while idx < len(meso_ids_sorted):
            mid = meso_ids_sorted[idx]
            macro_of_meso[mid] = M
            group = [mid]
            idx2 = idx + 1
            while idx2 < len(meso_ids_sorted) and abs(meso_centroid[meso_ids_sorted[idx2]] - meso_centroid[mid]) <= delta_macro:
                macro_of_meso[meso_ids_sorted[idx2]] = M
                group.append(meso_ids_sorted[idx2])
                idx2 += 1
            macro_centroid[M] = float(np.mean([meso_centroid[k] for k in group]))
            M += 1
            idx = idx2

        transmitting_micro = {}  # node -> cluster_id, nodes that actually transmit this step
        for cid, members in current_members_map.items():
            member_pos = positions[members]
            spatial_centroid = member_pos.mean(axis=0)
            dists = np.linalg.norm(member_pos - spatial_centroid, axis=1)
            d_max = dists.max() if dists.max() > 0 else 1.0
            max_batt = battery[members].max() if battery[members].max() > 0 else 1.0

            scores = np.zeros(len(members))
            for k, node in enumerate(members):
                phi_cent = 1 - abs(readings[node] - clusters[cid].centroid) / delta_micro
                phi_energy = battery[node] / max_batt
                phi_spatial = 1 - dists[k] / d_max
                recent_tx = tx_history[node].sum()
                phi_history = np.exp(-recent_tx / decay_lambda)
                scores[k] = (weights[0] * phi_cent + weights[1] * phi_energy +
                             weights[2] * phi_spatial + weights[3] * phi_history)
            scores = np.clip(scores, 1e-6, None)
            probs = scores / scores.sum()

            chosen = rng.choice(len(members), p=probs)
            for k, node in enumerate(members):
                if k == chosen:
                    transmitting_micro[node] = cid

        micro_tx_this_step = set()
        for node, cid in transmitting_micro.items():
            ident = clusters[cid]
            if abs(readings[node] - ident.last_reported_value) < DELTA_ENCODE_THRESHOLD_RATIO * delta_micro:
                bits = DELTA_REPORT_BITS
            else:
                bits = FULL_REPORT_BITS
            dist = np.linalg.norm(positions[node] - base_station)
            charge(node, bits, dist)
            ident.last_reported_value = readings[node]
            ident.last_rep_tx_step = t
            micro_tx_this_step.add(node)
        total_tx += len(micro_tx_this_step)

        for mid, cval in meso_centroid.items():
            if mid not in meso_trackers:
                meso_trackers[mid] = MesoMacroTracker(cval, t)
            tracker = meso_trackers[mid]
            due = tracker.update_and_should_report(cval, t, delta_meso)
            member_micro = [c for c in active_cids if meso_of_micro[c] == mid]
            reps_here = [n for n, c in transmitting_micro.items() if c in member_micro]
            if due and reps_here:
                relay = reps_here[0]
                dist = np.linalg.norm(positions[relay] - base_station)
                charge(relay, MESO_SUMMARY_BITS, dist)
                total_tx += 1

        for Mid, cval in macro_centroid.items():
            if Mid not in macro_trackers:
                macro_trackers[Mid] = MesoMacroTracker(cval, t)
            tracker = macro_trackers[Mid]
            due = tracker.update_and_should_report(cval, t, delta_macro)
            member_meso = [k for k, v in macro_of_meso.items() if v == Mid]
            member_micro = [c for c in active_cids if meso_of_micro[c] in member_meso]
            reps_here = [n for n, c in transmitting_micro.items() if c in member_micro]
            if due and reps_here:
                relay = reps_here[0]
                dist = np.linalg.norm(positions[relay] - base_station)
                charge(relay, MACRO_SUMMARY_BITS, dist)
                total_tx += 1

        for i in range(n_nodes):
            charge_rx(i, DOWNLINK_BEACON_BITS)

        tx_history[:, tx_ptr] = 0
        for node in micro_tx_this_step:
            tx_history[node, tx_ptr] = 1
        tx_ptr = (tx_ptr + 1) % history_window

        sq_err = 0.0
        raw_readings = raw_data[:, t]
        for i in range(n_nodes):
            if i in micro_tx_this_step:
                continue
            cid = node_cluster_idx[i]
            reps_in_cluster = [n for n, c in transmitting_micro.items() if c == cid]
            if reps_in_cluster:
                est = clusters[cid].centroid
            else:
                mid = meso_of_micro[cid]
                member_micro = [c for c in active_cids if meso_of_micro[c] == mid]
                meso_reps = [n for n, c in transmitting_micro.items() if c in member_micro]
                if meso_reps:
                    est = meso_centroid[mid]
                elif micro_tx_this_step:
                    tx_nodes = list(micro_tx_this_step)
                    d = np.linalg.norm(positions[tx_nodes] - positions[i], axis=1)
                    d[d == 0] = 1e-6
                    w = 1.0 / (d ** 2)
                    est = float(np.sum(w * data[tx_nodes, t]) / np.sum(w))
                else:
                    est = clusters[cid].centroid
            est_raw = est * denorm_std + denorm_mean
            sq_err += (raw_readings[i] - est_raw) ** 2
        reconstruction_sq_errors.append(sq_err / n_nodes)

        clusters = {cid: c for cid, c in clusters.items() if (t - c.last_rep_tx_step) <= t_inactive}

        avg_batt = battery.mean() if model_type == "simple" else 100.0 * battery.mean() / energy_model.e_initial
        battery_trace.append(avg_batt)
        n_clusters_trace.append(len(clusters))
        if depletion_step is None and battery.min() <= 0.0:
            depletion_step = t

    return {
        "total_tx": total_tx,
        "rmse": float(np.sqrt(np.mean(reconstruction_sq_errors))),
        "mse_trace": reconstruction_sq_errors,
        "battery_trace": battery_trace,
        "final_avg_battery_pct": battery_trace[-1],
        "battery_variance": float(np.var(battery if model_type == "simple"
                                          else 100.0 * battery / energy_model.e_initial)),
        "n_clusters_trace": n_clusters_trace,
        "avg_clusters": float(np.mean(n_clusters_trace[len(n_clusters_trace) // 2:])),
        "depletion_step": depletion_step,
    }
