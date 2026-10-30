# Causal Radar Tracklet Forecasting

Code companion to **Causal Recurrent Models for Space-Debris Radar Tracklet
Forecasting**. Three opt-in notebooks cover canonical forecasting, controlled
recurrent-memory experiments, geometric residual baselines and private export
analysis.

**Code only. No data, orbital simulator, split manifests, models, scalers,
result exports or figures are distributed. Numerical reproduction requires
authorized private inputs, original manifests and compatible archived artifacts.**
A fresh training run is not a guarantee of identical historical values.

## Run

Python 3.11 is the reference interpreter. TensorFlow 2.18.0 and Keras 3.8.0
match the audited software family; these dependencies are not a complete
historical environment lock or a cross-hardware determinism guarantee.

```bash
python -m pip install -e ".[neural,notebooks,test]"
python -m pytest -q
python tools/check_release.py
python -m jupyterlab
```

For tests without TensorFlow: install `.[test]`. Neural tests are explicitly
skipped. In Colab, install into the **active kernel** with `%pip`; all paths
must be visible inside its local container.

Start with [private input configuration](docs/DATA_INTERFACE.md).
Each notebook is disabled by default and writes only to external, unique
run directories. The same experiment can be run without Jupyter:

```bash
python -m radar_tracklets /external/private/configuration.json
```

## Protocols

| Mode | Purpose |
| --- | --- |
| `canonical` | Stateless GRU, 25 noisy inputs, one future target; GRU-CNN is optional |
| `focused` | Archived/fresh nine-feature stateful GRU with Huber delta 2 |
| `memory20` | Matched retain/reset pairs: 20 full epochs, shared initialization/scalers |
| `geometry40` | Matched angular/geometric residual pairs: 40 full epochs, shared initialization/scalers |

Checkpoint selection uses validation, never test scores. The fixed-budget
protocols do not early-stop. The geometric baseline and fallback output-fusion
rule are selected on validation. These remain exploratory follow-ups after
historical test consultation, not a newly independent confirmatory study.

## Minimal layout

- `notebooks/`: elevation, range and azimuth entry points.
- `src/radar_tracklets/`: features, models, stateful batching, training,
  external-data workflows, physical metrics and title-free plotting.
- `tests/`: arbitrary toy-array tests; no orbital simulator or benchmark data.
- `tools/check_release.py`: allowlisted files, clean notebooks and disclosure checks.
- `docs/`: [chapter mapping](docs/CHAPTER_ALIGNMENT.md),
  [provenance](docs/SOURCE_PROVENANCE.json),
  [local verification](docs/LOCAL_VALIDATION.md), [publishing](docs/PUBLISH_IT.md).

The original unfiltered benchmark and new visibility-corrected-generator
illustrations are different evidence sets. This repository never filters the
original splits to conceal negative elevation or replaces their results with
new examples. External posterior UKF comparisons are labelled as unequal
information sets; the missing filter implementation is not reconstructed.

Only azimuth errors are circular. Physical metrics pool valid target errors,
not per-trajectory RMSEs; seed summaries report mean and sample SD of seed-level
metrics. All figures are computed from external predictions, without manual
result curves, titles or captions.

Publication permission for data/simulator has not been established. No license
is assigned automatically: the author/institution should confirm code rights
and choose a license if appropriate. Public visibility alone does not grant
reuse rights.

