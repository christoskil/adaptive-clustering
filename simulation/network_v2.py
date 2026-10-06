"""
V2 environmental data generator.

The original generator (network.generate_environmental_data) had every node
share the same diurnal cycle AND the same slow drift, with only a modest
per-microclimate offset layered on top. That makes almost all node pairs
correlated most of the time regardless of position, which does not fairly
separate "spatial redundancy" (what LEACH-style clustering already
exploits) from "value redundancy between distant nodes" (the specific
phenomenon this paper's method targets). This generator fixes that by
building two genuinely distinct structures:

  1. A smooth SPATIAL field with a fixed correlation length (~25m, close to
     the paper's communication range) - nearby nodes are correlated,
     distant nodes are not, purely because of position.
  2. A position-INDEPENDENT microclimate class - nodes are assigned to one
     of a few classes uniformly at random regardless of location, each
     class with a distinct, persistent offset. Two nodes in the same class
     can be far apart yet read almost identical values; two spatially
     adjacent nodes can be in different classes and read quite different
     values.

A method that only exploits spatial proximity (LEACH) can capture (1) but
not (2). A method that clusters on value similarity regardless of position
(the paper's method) should capture both. This is the fair test of the
paper's central hypothesis.
"""
import numpy as np


def generate_environmental_data_v2(n_nodes, n_steps, positions, seed=0,
                                    n_microclimate_classes=4,
                                    spatial_corr_length=25.0,
                                    spatial_amplitude=2.5,
                                    microclimate_amplitude=3.0,
                                    n_spatial_sources=6,
                                    base_temp=27.77, diurnal_amplitude=4.5,
                                    ar_coefficient=0.99, ar_noise_std=0.36):
    """
    base_temp, diurnal_amplitude, ar_coefficient and ar_noise_std default
    to values calibrated against the real iPRISM DHT11 greenhouse dataset
    (Bodik... no -- Matsoukas, BSc thesis, iPRISM repository): mean air
    temperature 27.77 degC, half-amplitude diurnal swing ~4-5 degC, lag-1
    autocorrelation 0.994, and AR(1) innovation std 0.36, computed on a
    1105-sample (~2.3-day) excerpt of the real deployment retrieved from
    http://www.iprism.eu/assets/greenhouse_dataset_sept_2020.csv (see
    results/real_dht11/ and Editor comment #17). The real dataset is a
    single greenhouse sensor, not a multi-node spatial deployment, so this
    calibration covers only the TEMPORAL statistics (mean, diurnal swing,
    autocorrelation); the spatial-field and microclimate-class structure
    below has no real-world analogue to calibrate against and remains a
    documented modelling choice, not a fitted one.
    """
    rng = np.random.default_rng(seed)
    side = positions[:, 0].max() - positions[:, 0].min() + 1.0

    steps_per_day = 480
    t = np.arange(n_steps)
    diurnal = diurnal_amplitude * np.sin(2 * np.pi * (t / steps_per_day) - np.pi / 2)

    # --- (1) spatially-correlated field: a handful of slowly-drifting
    #     Gaussian "sources" whose influence decays with distance ---
    source_centers = rng.uniform(positions.min(), positions.max(), size=(n_spatial_sources, 2))
    source_phase = rng.uniform(0, 2 * np.pi, size=n_spatial_sources)
    source_freq = rng.uniform(0.5, 1.5, size=n_spatial_sources) / steps_per_day

    dist_to_sources = np.linalg.norm(
        positions[:, None, :] - source_centers[None, :, :], axis=2)  # (n_nodes, n_sources)
    kernel = np.exp(-(dist_to_sources ** 2) / (2 * spatial_corr_length ** 2))  # (n_nodes, n_sources)

    source_amplitude_t = np.sin(2 * np.pi * source_freq[None, :] * t[:, None] + source_phase[None, :])  # (n_steps, n_sources)
    spatial_field = spatial_amplitude * (kernel @ source_amplitude_t.T)  # (n_nodes, n_steps)

    # --- (2) position-independent microclimate class ---
    microclimate_class = rng.integers(0, n_microclimate_classes, size=n_nodes)
    class_offsets = np.linspace(-microclimate_amplitude, microclimate_amplitude, n_microclimate_classes)
    rng.shuffle(class_offsets)
    node_microclimate_offset = class_offsets[microclimate_class]  # (n_nodes,)

    # --- (3) sensor noise, AR(1) with real-data-calibrated persistence
    #     and innovation scale ---
    data = np.zeros((n_nodes, n_steps))
    for i in range(n_nodes):
        node_noise = rng.normal(0, ar_noise_std, size=n_steps)
        ar_noise = np.zeros(n_steps)
        for tt in range(1, n_steps):
            ar_noise[tt] = ar_coefficient * ar_noise[tt - 1] + node_noise[tt]
        data[i] = (base_temp + diurnal + spatial_field[i] +
                   node_microclimate_offset[i] + ar_noise)

    return data, microclimate_class


def normalize_data(data, delta_micro, delta_meso, delta_macro, calibrated_std=3.239):
    """Zero-mean, unit-variance normalization (Sect. 5.1's stated
    preprocessing step, Editor comments #15/16).

    The standard deviation used for normalization is the CALIBRATED REAL
    sensor value (3.239 degC, from the actual DHT11 deployment excerpt,
    see results/real_dht11/calibration_stats.json and Editor comment
    #17) rather than this particular synthetic dataset instance's own
    empirical std. This matters: our synthetic generator's total spread
    also includes the deliberately-engineered, position-independent
    microclimate-class separation (Sect. 5.1's fairness fix -- see
    network_v2.py's module docstring), which has no real-sensor
    analogue and would otherwise inflate the normalization denominator
    well past what a real deployment's natural variability looks like.
    Normalizing against the real sensor's own noise/variability scale is
    the more faithful reading of "preprocess by normalizing" for a
    tolerance parameter (Delta_micro=0.8) that was originally specified
    in real degree-like units.

    Delta_micro/meso/macro are rescaled by this same std, preserving the
    same physical cluster granularity (e.g. 0.8 degC) once clustering
    operates on normalized values. Returns (normalized_data,
    rescaled_delta_micro, rescaled_delta_meso, rescaled_delta_macro,
    mean, std)."""
    mean = float(data.mean())
    std = float(calibrated_std)
    normalized = (data - mean) / std
    return (normalized, delta_micro / std, delta_meso / std, delta_macro / std, mean, std)
