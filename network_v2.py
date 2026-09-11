
import numpy as np


def generate_environmental_data_v2(n_nodes, n_steps, positions, seed=0,
                                    n_microclimate_classes=4,
                                    spatial_corr_length=25.0,
                                    spatial_amplitude=2.5,
                                    microclimate_amplitude=3.0,
                                    n_spatial_sources=6):
    rng = np.random.default_rng(seed)
    side = positions[:, 0].max() - positions[:, 0].min() + 1.0

    steps_per_day = 480
    t = np.arange(n_steps)
    diurnal = 4.0 * np.sin(2 * np.pi * (t / steps_per_day) - np.pi / 2)
    base_temp = 18.0

    source_centers = rng.uniform(positions.min(), positions.max(), size=(n_spatial_sources, 2))
    source_phase = rng.uniform(0, 2 * np.pi, size=n_spatial_sources)
    source_freq = rng.uniform(0.5, 1.5, size=n_spatial_sources) / steps_per_day

    dist_to_sources = np.linalg.norm(
        positions[:, None, :] - source_centers[None, :, :], axis=2)  # (n_nodes, n_sources)
    kernel = np.exp(-(dist_to_sources ** 2) / (2 * spatial_corr_length ** 2))  # (n_nodes, n_sources)

    source_amplitude_t = np.sin(2 * np.pi * source_freq[None, :] * t[:, None] + source_phase[None, :])  # (n_steps, n_sources)
    spatial_field = spatial_amplitude * (kernel @ source_amplitude_t.T)  # (n_nodes, n_steps)

    microclimate_class = rng.integers(0, n_microclimate_classes, size=n_nodes)
    class_offsets = np.linspace(-microclimate_amplitude, microclimate_amplitude, n_microclimate_classes)
    rng.shuffle(class_offsets)
    node_microclimate_offset = class_offsets[microclimate_class]  # (n_nodes,)

    data = np.zeros((n_nodes, n_steps))
    for i in range(n_nodes):
        node_noise = rng.normal(0, 0.25, size=n_steps)
        ar_noise = np.zeros(n_steps)
        for tt in range(1, n_steps):
            ar_noise[tt] = 0.8 * ar_noise[tt - 1] + node_noise[tt]
        data[i] = (base_temp + diurnal + spatial_field[i] +
                   node_microclimate_offset[i] + ar_noise)

    return data, microclimate_class
