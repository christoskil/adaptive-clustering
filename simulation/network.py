
import numpy as np


class RadioEnergyModel:

    def __init__(self, e_elec=50e-9, e_fs=10e-12, e_mp=0.0013e-12,
                 packet_bits=2000, e_initial=2.0):
        self.e_elec = e_elec
        self.e_fs = e_fs
        self.e_mp = e_mp
        self.packet_bits = packet_bits
        self.d0 = np.sqrt(e_fs / e_mp)
        self.e_initial = e_initial  # Joules per node at full battery

    def tx_energy(self, distance_m):
        k = self.packet_bits
        if distance_m < self.d0:
            return self.e_elec * k + self.e_fs * k * (distance_m ** 2)
        return self.e_elec * k + self.e_mp * k * (distance_m ** 4)

    def tx_energy_bits(self, distance_m, bits):

        k = bits
        if distance_m < self.d0:
            return self.e_elec * k + self.e_fs * k * (distance_m ** 2)
        return self.e_elec * k + self.e_mp * k * (distance_m ** 4)

    def rx_energy(self):
        return self.e_elec * self.packet_bits

    def rx_energy_bits(self, bits):
        return self.e_elec * bits

    def to_battery_pct(self, energy_j):
        return 100.0 * energy_j / self.e_initial


class SimpleEnergyModel:

    def __init__(self, pct_per_tx=0.05):
        self.pct_per_tx = pct_per_tx

    def tx_cost_pct(self, distance_m=None):
        return self.pct_per_tx


def generate_rgg_topology(n_nodes, rc=35.0, seed=0):
    rng = np.random.default_rng(seed)
    side = 200.0 * np.sqrt(n_nodes / 100.0)
    positions = rng.uniform(0, side, size=(n_nodes, 2))
    return positions, side


def generate_environmental_data(n_nodes, n_steps, seed=0, n_microclimates=None):

    rng = np.random.default_rng(seed)
    if n_microclimates is None:
        n_microclimates = max(3, n_nodes // 15)

    microclimate_of_node = rng.integers(0, n_microclimates, size=n_nodes)
    microclimate_offset = rng.normal(0, 2.5, size=n_microclimates)

    t = np.arange(n_steps)
    # 3-minute sampling interval -> ~480 steps/day
    steps_per_day = 480
    diurnal = 6.0 * np.sin(2 * np.pi * (t / steps_per_day) - np.pi / 2)
    base_temp = 18.0

    slow_drift = np.cumsum(rng.normal(0, 0.01, size=n_steps))

    data = np.zeros((n_nodes, n_steps))
    for i in range(n_nodes):
        node_offset = microclimate_offset[microclimate_of_node[i]]
        node_noise = rng.normal(0, 0.3, size=n_steps)
        ar_noise = np.zeros(n_steps)
        for tstep in range(1, n_steps):
            ar_noise[tstep] = 0.85 * ar_noise[tstep - 1] + node_noise[tstep]
        data[i] = base_temp + diurnal + slow_drift + node_offset + ar_noise

    return data, microclimate_of_node
