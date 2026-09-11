# Adaptive Multi-Level Clustering — Reference Simulation Code

Reference implementation and evaluation code for:

> C. Kylafas, K. Kolomvatsos, "Adaptive Multi-Level Clustering with Dynamic
> Representative Selection for Data Redundancy Reduction in IoT Networks",
> submitted to *Computing* (Springer).

This repository accompanies the revised manuscript and the point-by-point
response to reviewers. It was written to let anyone independently
reproduce every number reported in the manuscript's Table 1, Table 2, and
Figures 5-8, and to make the reconstruction/energy-accounting logic fully
auditable (this was the Editor's explicit request during review).

## What's here

```
simulation/
  network.py                    Topology generation, both energy models,
                                 original (v1) synthetic environmental
                                 data generator
  network_v2.py                 Fairer synthetic data generator: separates
                                 a spatially-correlated field from a
                                 position-independent "microclimate" value
                                 component, so spatial- and value-based
                                 redundancy can be told apart (used for all
                                 final results; see "Why two generators" below)
  proposed_method.py            The paper's method (Algorithms 1-3), plus
                                 an optional cluster-merge step and an
                                 optional adaptive-threshold mode
  baselines.py                  Naive, LEACH, Prediction, TEEN, APTEEN,
                                 and a Compressed-Sensing-style baseline
  run_experiments.py            Reproduces Table 1 (10-seed mean +/- std,
                                 both energy models), Table 2, and the
                                 extended 2000-step long-run
  intel_topology_validation.py  Partial validation on the REAL 54-node
                                 Intel Berkeley Research Lab topology
  delta_micro_sweep.py          Delta_micro sweep WITHOUT cluster merging
  delta_micro_sweep_merge.py    Delta_micro sweep WITH cluster merging
                                 (produces the manuscript's Fig. 6)
  adaptive_k_sweep.py           Adaptive-threshold sweep (negative result,
                                 kept for transparency -- see below)
  multiseed_experiment.py       10-seed statistics on the ORIGINAL (v1)
                                 generator (kept for comparison)
  multiseed_v2_experiment.py    10-seed statistics on the fair (v2)
                                 generator, no merge (intermediate result,
                                 kept for transparency)
  make_figures.py               Regenerates the response-letter figures
  make_paper_figures.py         Regenerates the manuscript's figures
                                 (comparison_chart.png, temporal_performance.png,
                                 tradeoff_curve.png) directly into the LaTeX
                                 source tree
results/
  experiment_results.json       Every metric behind Table 1 / Table 2
  multiseed_v2_results.json     10-seed stats, fair generator, no merge
  delta_micro_sweep.json        Sweep without merge
  delta_micro_sweep_merge.json  Sweep with merge (used for Fig. 6)
  adaptive_k_sweep.json         Adaptive-threshold sweep (negative result)
  intel_topology_validation.json
  traces.npz                    Raw time series backing Figures 7-8
figures/
  fig5_data_reduction_comparison.png
  fig5b_rmse_comparison.png
  fig6_temporal_dynamics.png
  fig7_long_run_depletion.png
```

## Reproducing the results

```bash
pip install -r requirements.txt
cd simulation
python run_experiments.py               # Table 1, Table 2, long-run -> results/experiment_results.json
python intel_topology_validation.py      # Table S1 (response letter) -> results/intel_topology_validation.json
python delta_micro_sweep_merge.py        # Fig. 6 data -> results/delta_micro_sweep_merge.json
python make_figures.py                   # response-letter figures, 300 dpi
python make_paper_figures.py             # manuscript figures, 300 dpi
```

Everything is seeded (`numpy.random.default_rng`), so results are
deterministic and exactly reproducible.

## The official configuration

The manuscript's reported numbers use: the **v2** (fair) environmental
generator, **cluster merging enabled** (`enable_merge=True`), and the
paper's original `delta_micro = 0.8`. This is what `run_experiments.py`
uses by default. Every other script in this repo (the v1 generator, the
no-merge sweep, the adaptive-threshold sweep) is a diagnostic step kept
in the repository for transparency, not an alternative "official" result.

## Why two environmental generators (network.py vs network_v2.py)

The original generator gives every node the same diurnal cycle and the
same slow drift, with only a modest per-microclimate offset on top. That
makes most node pairs correlated most of the time regardless of position,
which does not fairly separate "redundancy a spatial method already
catches" from "redundancy only a value-based method can catch" -- the
specific phenomenon this paper's method targets. `network_v2.py` fixes
this by building a spatially-correlated field (nearby nodes correlated,
distant nodes not, purely from position) and a **separate**,
position-independent microclimate class (distant nodes can share a class
and read near-identical values; nearby nodes can be in different
classes). All final results use this generator.

## The cluster-merge fix

The manuscript's Sect. 4.1 states cluster lifecycle includes merging
("If the centroids of the clusters are moving closer, the clusters are
automatically merged"), but the literal Algorithm 1 pseudocode only ever
*creates* new micro-clusters and never merges existing ones. This was a
real gap between the described behaviour and the implemented algorithm.
`proposed_method.py`'s `enable_merge=True` option closes it: after the
per-node formation step, any micro-clusters whose centroids have
converged to within `delta_micro` are merged. This is not a tuning trick
-- it implements text already in the manuscript -- and it materially
changes the results (Table 1's data-reduction figure would be ~88% rather
than ~91% without it; see `results/multiseed_v2_results.json` for the
no-merge numbers).

## Two things that were tried and did NOT work (kept for transparency)

1. **Adaptive `delta_micro`** (`adaptive_k_sweep.py`): recomputing the
   micro-cluster tolerance every time step from the local noise scale
   (MAD of successive differences) performed *worse* than a fixed
   threshold at every tested setting, because a threshold that varies
   step-to-step fragments clusters that never re-merge. Result: data
   reduction topped out around 86%, below the fixed-threshold result.
2. **No setting of `delta_micro` beats TEEN/APTEEN on both data
   reduction and RMSE simultaneously** (`delta_micro_sweep_merge.py`,
   Fig. 6 in the manuscript). Increasing the tolerance trades RMSE for
   data reduction monotonically; this is reported as a genuine
   characteristic of the method (a tunable operating point), not
   something a better parameter choice would fix.

## "Total TX" convention

To remove ambiguity (the reviewers' and editor's core concern), every
result object reports two transmission counts:

- `total_tx` -- messages that actually reach the base station (this is
  what "Data reduction %" is computed from, consistently across all
  methods).
- `radio_tx` -- every individual radio transmission event, including
  intra-cluster hops (LEACH/TEEN/APTEEN member -> cluster-head traffic).
  Energy consumption is always computed from `radio_tx`, never from
  `total_tx`, since intra-cluster hops cost energy even though they don't
  count as a message reaching the base station.

This distinction changes the LEACH/TEEN/APTEEN "Data reduction" figures
compared to the original submission's Table 1 (58% -> ~90%; see the
response letter for the full discussion).

## Scalability: a corrected claim

The original submission stated data reduction "stays between 89-93%
regardless of network size" from 50 to 1000 nodes. Re-tested here
(using the paper's own area-scaling rule to hold density constant), data
reduction instead rises monotonically with network size: 82.5% at 50
nodes, 91.1% at 100, 95.7% at 200, 99.3% at 1000. Fixed-fraction schemes
(LEACH/TEEN/APTEEN) are size-invariant at ~90% by construction, so the
crossover point (our method overtaking them) is around n=90-100, not
universal. The manuscript has been updated to state this corrected, and
arguably more interesting, characterization (the method's advantage
grows with deployment density) rather than a claim of pure
size-independence.

## Known limitation: dataset validation

The original DHT11 dataset (BSc thesis project, iPRISM repository) is not
bundled here. For the reviewer's requested validation against an
established benchmark, `intel_topology_validation.py` imports the real
54-node spatial layout of the Intel Berkeley Research Lab deployment
(`db.csail.mit.edu/labdata`) and runs the same environmental generator
over that real, irregular topology in place of the synthetic one. It does
**not** use the real temperature/humidity readings (the ~150 MB readings
file was not retrievable in the environment this code was prepared in).
On this smaller (54-node), real topology, data reduction is consistently
*lower* than the baselines (~85.6% vs ~90% for LEACH/TEEN across 5 data
seeds) -- consistent with, and explained by, the density-dependence
above, not a contradiction of it. This is disclosed explicitly in the
manuscript's Conclusions rather than worked around.

## License

MIT — see `LICENSE`.
