# Adaptive Multi-Level Clustering — Reference Simulation Code

Reference implementation and evaluation code for:

> C. Kylafas, K. Kolomvatsos, "Adaptive Multi-Level Clustering with Dynamic
> Representative Selection for Data Redundancy Reduction in IoT Networks",
> submitted to *Computing* (Springer).

This repository accompanies the revised manuscript and the point-by-point
response to reviewers. It was written to let anyone independently
reproduce every number reported in Table 1 and Figures 5–7, and to make
the reconstruction/energy-accounting logic fully auditable (this was the
Editor's explicit request during review).

## What's here

```
simulation/
  network.py                    Topology generation, both energy models,
                                 synthetic environmental data generator
  proposed_method.py            The paper's method (Algorithms 1-3)
  baselines.py                  Naive, LEACH, Prediction, TEEN, APTEEN,
                                 and a Compressed-Sensing-style baseline
  run_experiments.py            Reproduces Table 1 (both energy models)
                                 and the extended 2000-step long-run
  intel_topology_validation.py  Partial validation on the REAL 54-node
                                 Intel Berkeley Research Lab topology
  make_figures.py                Regenerates all figures at 300 dpi
results/
  experiment_results.json       Every metric behind Table 1 / the response
  intel_topology_validation.json
  traces.npz                    Raw time series backing Figures 6-7
figures/
  fig5_data_reduction_comparison.png
  fig5b_rmse_comparison.png
  fig6_temporal_dynamics.png
  fig7_long_run_depletion.png   New: long-run node-depletion comparison
```

## Reproducing the results

```bash
pip install -r requirements.txt
cd simulation
python run_experiments.py               # writes results/experiment_results.json
python intel_topology_validation.py      # writes results/intel_topology_validation.json
python make_figures.py                   # writes figures/*.png at 300 dpi
```

Everything is seeded (`numpy.random.default_rng`), so results are
deterministic and exactly reproducible.

## Two energy models

The original submission used a fixed percentage of battery per
transmission (`SimpleEnergyModel`), which does not depend on distance.
In response to a reviewer comment, `network.py` also implements the
standard first-order radio model (Heinzelman et al., as used in LEACH),
with a free-space/multipath crossover distance. `run_experiments.py` runs
every method under **both** models so the two can be compared directly.

## "Total TX" convention

To remove ambiguity (again, the reviewers' and editor's core concern),
every result object reports two transmission counts:

- `total_tx` — messages that actually reach the base station (this is
  what "Data reduction %" is computed from, consistently across all
  methods).
- `radio_tx` — every individual radio transmission event, including
  intra-cluster hops (LEACH/TEEN/APTEEN member → cluster-head traffic).
  Energy consumption is always computed from `radio_tx`, never from
  `total_tx`, since intra-cluster hops cost energy even though they don't
  count as a message reaching the base station.

This distinction changes the LEACH/TEEN/APTEEN "Data reduction" figures
compared to the original submission's Table 1 (see the response letter
for a full discussion) — it does not change the qualitative conclusions
about energy balancing, but it does narrow the headline data-reduction
gap and shows TEEN/APTEEN slightly ahead on raw RMSE. Both are reported
honestly in `results/experiment_results.json`.

## Known limitation: dataset validation

The original DHT11 dataset (BSc thesis project, iPRISM repository) is not
bundled here. For the Reviewer's requested validation against an
established benchmark, `intel_topology_validation.py` imports the real
54-node spatial layout of the Intel Berkeley Research Lab deployment
(`db.csail.mit.edu/labdata`) and runs the same synthetic-but-realistic
environmental generator over that real, irregular topology. It does
**not** use the real temperature/humidity readings (the ~150 MB readings
file was not retrievable in the environment this code was prepared in) —
this remains an open item, disclosed explicitly rather than worked around.

## License

MIT — see `LICENSE`.
