# Algorithm/Equation-to-Code Mapping

Requested by Editor comment #24. All line references are to
`simulation/proposed_method_v3.py` unless otherwise noted.

| Manuscript element | What it specifies | Code location |
|---|---|---|
| Algorithm 1 (Adaptive Cluster Formation) | Per-node nearest-centroid join/create | `run_proposed_method`, "Phase 1" block |
| Eq. 9 ($\delta_k$, join condition) | Distance to each cluster centroid | Phase 1: `dists = np.abs(readings[i] - centroid_arr)` |
| Eq. 10 (centroid EMA update) | $\mu_{k+1} = \beta d_i(t) + (1-\beta)\mu_k$ | Immediately after Phase 1: `clusters[cid].centroid = beta * mean_val + (1 - beta) * clusters[cid].centroid` |
| Sect. 4.1 cluster merging | Merge clusters within $\Delta_{micro}$ | "Phase 1c" block (`enable_merge`) |
| Sect. 4.1 cluster splitting | Split a cluster whose variance is too high | `_split_oversized_clusters()`, called as "Phase 1b" |
| Sect. 4.1 cluster lifecycle (inactivity) | Remove clusters with no recent representative transmission | "Phase 6": `clusters = {cid: c for cid, c ... if (t - c.last_rep_tx_step) <= t_inactive}` |
| Eq. 11 (meso centroid, weighted avg) | Membership-weighted average of micro centroids | "Phase 2", `meso_centroid[m] = float(np.average(cents, weights=sizes))` |
| Macro-cluster aggregation (Sect. 3.3/4.3, Algorithm 3 Phase 2) | Aggregate meso-clusters into macro-clusters | "Phase 2", `macro_of_meso` / `macro_centroid` block |
| Sect. 4.3 adaptive reporting frequency | Stable clusters report less often | `MesoMacroTracker.update_and_should_report()` |
| Sect. 4.3 delta encoding | Small packet if value close to last report | Phase 4: `DELTA_ENCODE_THRESHOLD_RATIO` check, `DELTA_REPORT_BITS` vs `FULL_REPORT_BITS` |
| Algorithm 2 (Distributed Representative Selection) | Per-cluster score + selection | "Phase 3" block |
| Eq. 12 (combined score $\sigma_i$) | Weighted sum of four components | Phase 3: `scores[k] = weights[0]*phi_cent + ...` |
| Eq. 13 ($\phi_{cent}$) | Value centrality | Phase 3: `phi_cent = 1 - abs(readings[node] - clusters[cid].centroid) / delta_micro` |
| Eq. 14 ($\phi_{energy}$) | Battery ratio | Phase 3: `phi_energy = battery[node] / max_batt` |
| Eq. 15 ($\phi_{spatial}$) | Spatial centrality | Phase 3: `phi_spatial = 1 - dists[k] / d_max` |
| Eq. 16 ($\phi_{history}$) | Recent-transmission decay | Phase 3: `phi_history = np.exp(-recent_tx / decay_lambda)` |
| Eq. 17 ($p_i(t)$, transmission probability) | Probabilistic selection | Phase 3: `probs = scores/scores.sum(); chosen = rng.choice(len(members), p=probs)` -- see README.md "Eq. 17: two readings" for the categorical-vs-Bernoulli discussion (Editor comment #11) |
| Sect. 3.1 communication range $r_c$ / $\mathcal{N}_i^{space}$ | Local-only direct radio reachability | `_build_adjacency()`; base-station downlink beacon carries remote cluster centroids so distant value-similar nodes coordinate without violating $r_c$ (Editor comments #12-14) |
| Sect. 3.4 / Eq. 6-7 (energy model) | Fixed-cost and radio models | `network.py`: `SimpleEnergyModel`, `RadioEnergyModel` (`tx_energy_bits`, `rx_energy_bits`) |
| Eq. 7 $E_{rx}(k)$ | Reception energy | Phase 4: `charge_rx(i, DOWNLINK_BEACON_BITS)` for every node every step (Editor comment #21) |
| Base station location (Fig. 1) | Fixed point outside the field | `baselines.base_station_position()`, shared by every method (Editor comment #20) |
| Eq. 18 (micro reconstruction) | $\hat{d}_i = \mu_k + \epsilon_i$ | Phase 5: `est = clusters[cid].centroid` |
| Eq. 19 (meso reconstruction) | $\hat{d}_i = \mu_m^{meso}$ | Phase 5: `est = meso_centroid[mid]` |
| Eq. 20 (spatial interpolation fallback) | Inverse-distance weighting from transmitting neighbours | Phase 5: `w = 1.0 / (d ** 2); est = float(np.sum(w * data[tx_nodes, t]) / np.sum(w))` -- now a genuine, reachable code path (Editor comments #9/10) |
| Algorithm 3 (complete protocol) | All phases per time step | The full `for t in range(n_steps):` loop, Phases 1-6 |
| Sect. 5.1 normalization | Zero-mean, unit-variance preprocessing | `network_v2.normalize_data()`; baselines run on raw data with their own published thresholds, only this method's tolerances are normalized (Editor comments #15/16) |
