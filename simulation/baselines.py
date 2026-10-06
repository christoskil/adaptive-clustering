
import numpy as np


def base_station_position(positions):

    field_min, field_max = positions.min(axis=0), positions.max(axis=0)
    field_side = max(field_max[0] - field_min[0], field_max[1] - field_min[1], 1.0)
    return np.array([(field_min[0] + field_max[0]) / 2.0, field_max[1] + field_side])


def _tx_cost(model_type, energy_model, distance_m=None):
    if model_type == "simple":
        return energy_model.tx_cost_pct()
    return energy_model.tx_energy(distance_m if distance_m is not None else 50.0)


def run_naive(data, positions, energy_model, model_type="simple", n_steps=None):
    n_nodes, total_steps = data.shape
    if n_steps is None:
        n_steps = total_steps
    battery = np.full(n_nodes, 100.0 if model_type == "simple" else energy_model.e_initial)
    total_tx = 0
    battery_trace = []
    depletion_step = None
    center = base_station_position(positions)  # fixed BS location (E20), not the field centroid

    for t in range(n_steps):
        for i in range(n_nodes):
            dist_to_bs = np.linalg.norm(positions[i] - center)
            battery[i] -= _tx_cost(model_type, energy_model, dist_to_bs)
            battery[i] = max(battery[i], 0.0)
        total_tx += n_nodes
        avg = battery.mean() if model_type == "simple" else 100.0 * battery.mean() / energy_model.e_initial
        battery_trace.append(avg)
        if depletion_step is None and battery.min() <= 0.0:
            depletion_step = t

    return {
        "total_tx": total_tx, "radio_tx": total_tx, "rmse": 0.0, "battery_trace": battery_trace,
        "final_avg_battery_pct": battery_trace[-1],
        "battery_variance": float(np.var(battery if model_type == "simple"
                                          else 100.0 * battery / energy_model.e_initial)),
        "depletion_step": depletion_step,
    }


def run_leach(data, positions, energy_model, model_type="simple",
              cluster_head_prob=0.1, n_steps=None, seed=1):
    """Classic LEACH: probabilistic cluster-head rotation, spatial clusters,
    intra-cluster aggregation, one hop CH -> base station."""
    n_nodes, total_steps = data.shape
    if n_steps is None:
        n_steps = total_steps
    rng = np.random.default_rng(seed)
    battery = np.full(n_nodes, 100.0 if model_type == "simple" else energy_model.e_initial)
    center = base_station_position(positions)  # fixed BS location (E20), not the field centroid

    round_r = 0
    ch_history = np.zeros(n_nodes, dtype=int)  # rounds since last CH
    total_tx = 0       # messages that actually reach the base station
    radio_tx = 0       # all radio transmission events (intra-cluster + CH->BS)
    sq_errors = []
    battery_trace = []
    depletion_step = None

    for t in range(n_steps):
        alive = battery > 0
        # LEACH threshold formula
        thresh = cluster_head_prob / (1 - cluster_head_prob * (round_r % int(1 / cluster_head_prob)))
        is_ch = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            if not alive[i]:
                continue
            if ch_history[i] > 0:
                continue
            if rng.random() < thresh:
                is_ch[i] = True
        if not is_ch.any():
            candidates = np.where(alive)[0]
            if len(candidates) > 0:
                is_ch[rng.choice(candidates)] = True
        ch_history[is_ch] = int(1 / cluster_head_prob)
        ch_history[ch_history > 0] -= 1

        ch_indices = np.where(is_ch)[0]
        node_to_ch = {}
        for i in range(n_nodes):
            if not alive[i] or is_ch[i]:
                continue
            d = np.linalg.norm(positions[ch_indices] - positions[i], axis=1)
            node_to_ch[i] = ch_indices[np.argmin(d)]

        readings = data[:, t]
        transmitting = set(ch_indices.tolist())
        # non-CH members transmit to their CH (intra-cluster tx, short range;
        # counted in radio_tx/energy but does NOT reach the base station)
        for i, ch in node_to_ch.items():
            d_to_ch = np.linalg.norm(positions[i] - positions[ch])
            battery[i] -= _tx_cost(model_type, energy_model, d_to_ch)
            battery[i] = max(battery[i], 0.0)
            radio_tx += 1
        for ch in ch_indices:
            dist_to_bs = np.linalg.norm(positions[ch] - center)
            battery[ch] -= _tx_cost(model_type, energy_model, dist_to_bs)
            battery[ch] = max(battery[ch], 0.0)
            radio_tx += 1
        total_tx += len(ch_indices)  # only CH->BS messages reach the base station

        # reconstruction: CH aggregate (mean of members) represents the cluster
        sq_err = 0.0
        cluster_means = {}
        for ch in ch_indices:
            members = [i for i, c in node_to_ch.items() if c == ch] + [ch]
            cluster_means[ch] = readings[members].mean()
        for i in range(n_nodes):
            est = cluster_means[node_to_ch[i]] if i in node_to_ch else readings[i]
            sq_err += (readings[i] - est) ** 2
        sq_errors.append(sq_err / n_nodes)

        avg = battery.mean() if model_type == "simple" else 100.0 * battery.mean() / energy_model.e_initial
        battery_trace.append(avg)
        if depletion_step is None and battery.min() <= 0.0:
            depletion_step = t
        round_r += 1

    return {
        "total_tx": total_tx, "radio_tx": radio_tx, "rmse": float(np.sqrt(np.mean(sq_errors))),
        "battery_trace": battery_trace, "final_avg_battery_pct": battery_trace[-1],
        "battery_variance": float(np.var(battery if model_type == "simple"
                                          else 100.0 * battery / energy_model.e_initial)),
        "depletion_step": depletion_step,
    }


def run_prediction(data, positions, energy_model, model_type="simple",
                    alpha=0.3, threshold=0.5, n_steps=None):

    n_nodes, total_steps = data.shape
    if n_steps is None:
        n_steps = total_steps
    battery = np.full(n_nodes, 100.0 if model_type == "simple" else energy_model.e_initial)
    center = base_station_position(positions)  # fixed BS location (E20), not the field centroid
    pred = data[:, 0].copy()
    total_tx = 0
    sq_errors = []
    battery_trace = []
    depletion_step = None

    for t in range(n_steps):
        readings = data[:, t]
        deviation = np.abs(readings - pred)
        transmit_mask = deviation > threshold
        for i in np.where(transmit_mask)[0]:
            dist_to_bs = np.linalg.norm(positions[i] - center)
            battery[i] -= _tx_cost(model_type, energy_model, dist_to_bs)
            battery[i] = max(battery[i], 0.0)
        total_tx += int(transmit_mask.sum())

        est = np.where(transmit_mask, readings, pred)
        sq_errors.append(np.mean((readings - est) ** 2))

        pred = np.where(transmit_mask, readings, pred)
        pred = alpha * readings + (1 - alpha) * pred

        avg = battery.mean() if model_type == "simple" else 100.0 * battery.mean() / energy_model.e_initial
        battery_trace.append(avg)
        if depletion_step is None and battery.min() <= 0.0:
            depletion_step = t

    return {
        "total_tx": total_tx, "radio_tx": total_tx, "rmse": float(np.sqrt(np.mean(sq_errors))),
        "battery_trace": battery_trace, "final_avg_battery_pct": battery_trace[-1],
        "battery_variance": float(np.var(battery if model_type == "simple"
                                          else 100.0 * battery / energy_model.e_initial)),
        "depletion_step": depletion_step,
    }


def run_teen(data, positions, energy_model, model_type="simple",
             hard_threshold=1.0, soft_threshold=0.3, cluster_head_prob=0.1,
             n_steps=None, seed=2):

    n_nodes, total_steps = data.shape
    if n_steps is None:
        n_steps = total_steps
    rng = np.random.default_rng(seed)
    battery = np.full(n_nodes, 100.0 if model_type == "simple" else energy_model.e_initial)
    center = base_station_position(positions)  # fixed BS location (E20), not the field centroid
    last_sent_value = data[:, 0].copy()

    round_r = 0
    ch_history = np.zeros(n_nodes, dtype=int)
    total_tx = 0      # messages reaching the base station (CH reports only)
    radio_tx = 0      # all radio transmission events (member->CH + CH->BS)
    sq_errors = []
    battery_trace = []
    depletion_step = None

    for t in range(n_steps):
        alive = battery > 0
        thresh = cluster_head_prob / (1 - cluster_head_prob * (round_r % int(1 / cluster_head_prob)))
        is_ch = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            if not alive[i] or ch_history[i] > 0:
                continue
            if rng.random() < thresh:
                is_ch[i] = True
        if not is_ch.any():
            candidates = np.where(alive)[0]
            if len(candidates) > 0:
                is_ch[rng.choice(candidates)] = True
        ch_history[is_ch] = int(1 / cluster_head_prob)
        ch_history[ch_history > 0] -= 1

        ch_indices = np.where(is_ch)[0]
        node_to_ch = {}
        for i in range(n_nodes):
            if not alive[i] or is_ch[i]:
                continue
            d = np.linalg.norm(positions[ch_indices] - positions[i], axis=1)
            node_to_ch[i] = ch_indices[np.argmin(d)]

        readings = data[:, t]
        transmit_mask = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            if not alive[i]:
                continue
            if is_ch[i]:
                continue  # CH transmission to BS handled separately below
            if (readings[i] > hard_threshold and
                    abs(readings[i] - last_sent_value[i]) > soft_threshold):
                transmit_mask[i] = True

        for i in np.where(transmit_mask)[0]:
            dist = np.linalg.norm(positions[i] - positions[node_to_ch[i]])
            battery[i] -= _tx_cost(model_type, energy_model, dist)
            battery[i] = max(battery[i], 0.0)
            last_sent_value[i] = readings[i]
            radio_tx += 1
        # cluster heads periodically report to the base station regardless
        # of member triggering (standard TEEN CH behaviour)
        for ch in ch_indices:
            dist_to_bs = np.linalg.norm(positions[ch] - center)
            battery[ch] -= _tx_cost(model_type, energy_model, dist_to_bs)
            battery[ch] = max(battery[ch], 0.0)
            radio_tx += 1
        transmit_mask[ch_indices] = True
        total_tx += len(ch_indices)

        sq_err = 0.0
        for i in range(n_nodes):
            est = readings[i] if transmit_mask[i] else last_sent_value[i]
            sq_err += (readings[i] - est) ** 2
        sq_errors.append(sq_err / n_nodes)

        avg = battery.mean() if model_type == "simple" else 100.0 * battery.mean() / energy_model.e_initial
        battery_trace.append(avg)
        if depletion_step is None and battery.min() <= 0.0:
            depletion_step = t
        round_r += 1

    return {
        "total_tx": total_tx, "radio_tx": radio_tx, "rmse": float(np.sqrt(np.mean(sq_errors))),
        "battery_trace": battery_trace, "final_avg_battery_pct": battery_trace[-1],
        "battery_variance": float(np.var(battery if model_type == "simple"
                                          else 100.0 * battery / energy_model.e_initial)),
        "depletion_step": depletion_step,
    }


def run_apteen(data, positions, energy_model, model_type="simple",
               hard_threshold=1.0, soft_threshold=0.3, count_time=20,
               cluster_head_prob=0.1, n_steps=None, seed=4):

    n_nodes, total_steps = data.shape
    if n_steps is None:
        n_steps = total_steps
    rng = np.random.default_rng(seed)
    battery = np.full(n_nodes, 100.0 if model_type == "simple" else energy_model.e_initial)
    center = base_station_position(positions)  # fixed BS location (E20), not the field centroid
    last_sent_value = data[:, 0].copy()
    steps_since_report = np.zeros(n_nodes, dtype=int)

    round_r = 0
    ch_history = np.zeros(n_nodes, dtype=int)
    total_tx = 0
    radio_tx = 0
    sq_errors = []
    battery_trace = []
    depletion_step = None

    for t in range(n_steps):
        alive = battery > 0
        thresh = cluster_head_prob / (1 - cluster_head_prob * (round_r % int(1 / cluster_head_prob)))
        is_ch = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            if not alive[i] or ch_history[i] > 0:
                continue
            if rng.random() < thresh:
                is_ch[i] = True
        if not is_ch.any():
            candidates = np.where(alive)[0]
            if len(candidates) > 0:
                is_ch[rng.choice(candidates)] = True
        ch_history[is_ch] = int(1 / cluster_head_prob)
        ch_history[ch_history > 0] -= 1

        ch_indices = np.where(is_ch)[0]
        node_to_ch = {}
        for i in range(n_nodes):
            if not alive[i] or is_ch[i]:
                continue
            d = np.linalg.norm(positions[ch_indices] - positions[i], axis=1)
            node_to_ch[i] = ch_indices[np.argmin(d)]

        readings = data[:, t]
        transmit_mask = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            if not alive[i] or is_ch[i]:
                continue
            hard_ok = readings[i] > hard_threshold
            soft_ok = abs(readings[i] - last_sent_value[i]) > soft_threshold
            forced = steps_since_report[i] >= count_time
            if (hard_ok and soft_ok) or (hard_ok and forced):
                transmit_mask[i] = True

        for i in np.where(transmit_mask)[0]:
            dist = np.linalg.norm(positions[i] - positions[node_to_ch[i]])
            battery[i] -= _tx_cost(model_type, energy_model, dist)
            battery[i] = max(battery[i], 0.0)
            last_sent_value[i] = readings[i]
            steps_since_report[i] = 0
            radio_tx += 1
        steps_since_report += 1
        steps_since_report[transmit_mask] = 0

        for ch in ch_indices:
            dist_to_bs = np.linalg.norm(positions[ch] - center)
            battery[ch] -= _tx_cost(model_type, energy_model, dist_to_bs)
            battery[ch] = max(battery[ch], 0.0)
            radio_tx += 1
        transmit_mask[ch_indices] = True
        total_tx += len(ch_indices)

        sq_err = 0.0
        for i in range(n_nodes):
            est = readings[i] if transmit_mask[i] else last_sent_value[i]
            sq_err += (readings[i] - est) ** 2
        sq_errors.append(sq_err / n_nodes)

        avg = battery.mean() if model_type == "simple" else 100.0 * battery.mean() / energy_model.e_initial
        battery_trace.append(avg)
        if depletion_step is None and battery.min() <= 0.0:
            depletion_step = t
        round_r += 1

    return {
        "total_tx": total_tx, "radio_tx": radio_tx, "rmse": float(np.sqrt(np.mean(sq_errors))),
        "battery_trace": battery_trace, "final_avg_battery_pct": battery_trace[-1],
        "battery_variance": float(np.var(battery if model_type == "simple"
                                          else 100.0 * battery / energy_model.e_initial)),
        "depletion_step": depletion_step,
    }


def run_compressed_sensing(data, positions, energy_model, model_type="simple",
                            sample_ratio=0.15, n_steps=None, seed=3):

    n_nodes, total_steps = data.shape
    if n_steps is None:
        n_steps = total_steps
    rng = np.random.default_rng(seed)
    battery = np.full(n_nodes, 100.0 if model_type == "simple" else energy_model.e_initial)
    center = base_station_position(positions)  # fixed BS location (E20), not the field centroid
    n_sample = max(1, int(sample_ratio * n_nodes))
    total_tx = 0
    sq_errors = []
    battery_trace = []
    depletion_step = None

    for t in range(n_steps):
        readings = data[:, t]
        alive_idx = np.where(battery > 0)[0]
        if len(alive_idx) == 0:
            sq_errors.append(np.mean(readings ** 2))
            battery_trace.append(0.0)
            continue
        sampled = rng.choice(alive_idx, size=min(n_sample, len(alive_idx)), replace=False)
        for i in sampled:
            dist_to_bs = np.linalg.norm(positions[i] - center)
            battery[i] -= _tx_cost(model_type, energy_model, dist_to_bs)
            battery[i] = max(battery[i], 0.0)
        total_tx += len(sampled)

        # inverse-distance-weighted reconstruction from sampled nodes
        est = np.zeros(n_nodes)
        for i in range(n_nodes):
            if i in sampled:
                est[i] = readings[i]
                continue
            d = np.linalg.norm(positions[sampled] - positions[i], axis=1)
            d[d == 0] = 1e-6
            w = 1.0 / (d ** 2)
            est[i] = np.sum(w * readings[sampled]) / np.sum(w)
        sq_errors.append(np.mean((readings - est) ** 2))

        avg = battery.mean() if model_type == "simple" else 100.0 * battery.mean() / energy_model.e_initial
        battery_trace.append(avg)
        if depletion_step is None and battery.min() <= 0.0:
            depletion_step = t

    return {
        "total_tx": total_tx, "radio_tx": total_tx, "rmse": float(np.sqrt(np.mean(sq_errors))),
        "battery_trace": battery_trace, "final_avg_battery_pct": battery_trace[-1],
        "battery_variance": float(np.var(battery if model_type == "simple"
                                          else 100.0 * battery / energy_model.e_initial)),
        "depletion_step": depletion_step,
    }
