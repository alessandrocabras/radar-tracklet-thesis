# Authorized private inputs and execution

Keep all private configurations, data, models and outputs outside this Git tree.
Paths must be visible to the active kernel/container. No download or simulator
is provided. Only load trusted pickle/joblib artifacts.

## Original data

A private data root contains `noise/` and `clean/`. Noisy CSV basenames map to
clean basenames by replacing `_noise` with `_clean`. Required columns:

| Column | Unit |
| --- | --- |
| `Azimuth (rad)` | rad |
| `Elevation (rad)` | rad |
| `Range (m)` | m |
| `Doppler (m/s)` | radial velocity, m/s |
| `Time (s)` | recorded seconds; recommended |

All four observables must be finite. Counts must match and be >=26.
If timestamps are present, both members must provide them, with increasing
time and agreement within 1e-9 s. No nearest-neighbour pairing of noisy/clean
observations is performed.

The original JSON schema is accepted:
```json
{"split_files": {"train": ["..."], "val": ["..."], "test": ["..."]}}
```
The previous adapter's top-level `train_files`, `val_files`, `test_files`
schema is also accepted. Lists must be nonempty, unique, split-disjoint
basenames. Contradictory representations fail. No new random split is created.
The chapter's original split sizes are 7,000/1,500/1,500; the reader reports
the supplied lists, not an invented fixed split. Name disjointness does not
prove physical independence.

## Canonical evaluation

Save an external configuration such as:
```json
{
  "mode": "canonical",
  "coordinate": "elevation",
  "architecture": "GRU",
  "execute_training": false,
  "data_root": "/external/private/data",
  "manifest": "/external/private/elevation_dataset_manifest.json",
  "model": "/external/private/elevation.keras",
  "scaler_X": "/external/private/elevation_scaler_X.pkl",
  "scaler_y": "/external/private/elevation_scaler_y.pkl",
  "output_root": "/external/private/review_runs",
  "figures": true
}
```
Use the corresponding coordinate/model/scalers for range or canonical
six-feature azimuth. Frozen evaluation processes one trajectory's windows
at a time rather than materializing the whole test set as windows.

To fit a new window model, remove `model`/scaler keys, set
`execute_training: true` and provide `seed` and an explicit `training` object:
`epochs`, `batch_size`, `patience`, `lr_patience`, optionally `min_delta`.
The constructor uses Adam 0.001 and MSE on standardized targets; callbacks
monitor validation MSE on that scale. This full-array fitting path can use
substantial RAM. Supply archived settings if reproducing a specific protocol;
the example does not manufacture missing tuning logs. Choose `GRU-CNN`
explicitly for the optional hybrid. For an architecture experiment, run each
explicit architecture with the same private manifest/settings and compare
their exported scores, keeping budget differences visible.

## Azimuth fits

```json
{
  "mode": "memory20",
  "execute_training": true,
  "seeds": [42, 43, 44],
  "data_root": "/external/private/data",
  "manifest": "/external/private/azimuth_dataset_manifest.json",
  "output_root": "/external/private/review_runs",
  "figures": true
}
```

Change mode to `geometry40` for the paired geometric experiment or `focused`
for the earlier focused protocol. Fixed-budget runs always complete their
20/40 epochs unless execution is interrupted or fails; a partial run is not
labelled complete. They have no deadline controller or silent time cap.
A new unique output directory is created on every invocation; original
checkpoints are never overwritten. Best weights are saved during fixed-budget
training. Interrupted fits are not resumed automatically.

For frozen focused evaluation, set `execute_training: false`, supply the
original nine-feature `scaler_X`, residual `scaler_y`, and a `weights` object
mapping seed strings to private weight paths. For example:
`"weights": {"42": "/external/private/focused_seed42.weights.h5"}` with
`"seeds": [42]`. No scaler is refitted.

Fixed-budget variants share the training-only scalers and same initial weights
within each seed; hashes and selected epochs are exported. Controlled memory
uses partial fitting in original training-manifest order. Geometry uses a
concatenated fit in sorted training-name order, matching the respective source
workflows. Both geometric variants share the ANGULAR residual target scaler.

## Outputs and figures

Each new run writes configuration/provenance, metrics, per-target errors with
trajectory IDs and original target indices, trajectory-level metrics, extreme
errors, profiles where available, retained/omitted support and title-free PDFs.
Seed summaries are mean and sample SD of seed-level statistics, not pooled
seed errors. Training runs also save private checkpoints, scalers, histories
and validation selection information. These files are not suitable for commit.

Sample-level exports require `trajectory_id,target_index,prediction,clean`.
Coordinates are rad/rad/m. The loader recomputes errors, never trusts an
archived error column. It accepts `timestamp_s`/`time_s` and
`noise_profile`/`profile`. Functions in `analysis`:

- `compare_common_support`: intersect keys and verify references/timestamps.
- `evaluate_fallback`: separate seed-matched validation/test pairs, select
  the validation rule before evaluating its test outputs.
- `spatial_metrics`: predicted azimuth/elevation/range on shared keys;
  direction mrad and local-position m, with explicit elevation strata.
- `ukf_posterior_comparison`: one external trajectory, declared units,
  recorded-time matching, explicit warm-up and reference verification.

The three notebooks expose these guarded operations; azimuth additionally
exposes follow-up training, fallback and spatial analysis. Library use is also
supported. For `plot_errors(frame, path, trajectory_id=...)`, the selected
trajectory error and its ECDF are computed from actual samples. Target-time
noisy-clean values are plotted at the same original target index; these are
not the perturbations of the preceding input. No manual data curves are used.

UKF loader keyword example for the separately audited legacy CSV:
```python
UKF_LOADER = dict(
    filtered_column="filtered [el, deg]", measured_column="measured [el, deg]",
    real_column="real [el, deg]", filtered_unit="rad", measured_unit="rad",
    real_unit="rad", time_column="time[s]", warmup_target_index=50,
    timestamp_tolerance_s=1e-6, reference_atol=1e-8)
```
Those declared radian units apply to that audited artifact despite its labels,
not to arbitrary CSVs. Loader settings and common-support size are exported.
The UKF implementation and initial covariances remain unavailable; this code
does not invent them or assert matched information sets.

New visibility-corrected-generator cases may be supplied as external prediction
exports for illustrations. Keep their output directory/configuration distinct
from original benchmark experiments. The simulator is not part of this interface.

