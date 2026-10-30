# Chapter alignment and evidence limits

The companion follows *Causal Recurrent Models for Space-Debris Radar Tracklet
Forecasting*, including the controlled-memory and geometric-baseline follow-ups.

| Topic | Implementation | Contract |
| --- | --- | --- |
| Canonical inputs | `preprocessing`, `workflows.window_arrays` | Noisy indices 0-24 predict clean index 25 |
| Window families | `models.build_window_model` | GRU default; explicit MLP/CNN/GRU-CNN alternatives |
| Elevation | `build_elevation_features_and_target` | Eight noisy features; direct radian target |
| Range | `build_range_features_and_target` | Nine features; residual around a causal quadratic forecast |
| Canonical azimuth | `build_azimuth_windows` | Six window-relative inputs; measured first-window pivot |
| Focused azimuth | `build_final_azimuth`, `fit_final_azimuth` | GRU 256, Dense 64 ReLU, Dense 1; Huber-d2 |
| Controlled memory | `fit_fixed_budget`, mode `memory20` | Matched retain/reset; 20 full epochs |
| Geometric residual | `geometry`, mode `geometry40` | Noisy Cartesian baseline; paired 40-epoch fits |
| Fallback | `reporting.select_fallback`, `analysis.evaluate_fallback` | Validation-selected output fusion of two networks |
| Metrics and exports | `evaluation`, `reporting`, `analysis` | IDs, target indices, P99, maxima, per-trajectory scores |
| Spatial errors | `analysis.spatial_metrics` | Three predicted coordinates on shared support |
| UKF | `analysis.ukf_posterior_comparison` | External posterior, declared units and timestamp matching |

## Features and units

Input order before standardization:

- Elevation (8): E, delta E, A, delta A, range, delta range, radial velocity,
  delta radial velocity.
- Range (9): E, delta E, unwrapped A, delta A, range, delta range,
  causal local range residual, radial velocity, delta radial velocity.
- Canonical azimuth (6): unwrapped A minus first noisy window pivot,
  delta A, E, delta E, delta range, radial velocity.
- Stateful azimuth (9): sin A, cos A, wrapped first increment,
  second increment, median-seven increment, E, delta E, delta range,
  radial velocity.

Angles/increments are radians; range terms are metres; radial velocity terms
are m/s. First/second increments are per sample, not timestamp derivatives.
Initial differences are zero. The median excludes the artificial initial zero
and uses up to seven real increments. Its baseline is A[t]+median_increment[t],
with no extra delta-t factor. Protocols assume nominal near-one-second cadence;
they do not implement irregular-time forecasting. `Doppler (m/s)` is used as
radial velocity, not converted from frequency.

Clean values create supervision and references only, never feature inputs,
recurrent feedback or reconstruction baselines. There is no clean teacher
forcing. Elevation uses ordinary azimuth differences; range and azimuth unwrap
azimuth. Only azimuth evaluation errors use shortest-arc wrapping.

Range baseline: quadratic `numpy.polyfit`, at most 25 noisy samples through t,
absolute sample indices, extrapolation at t+1. Prefix degree is reduced.
The local residual uses at most 24 strictly preceding noisy samples; insufficient
history gives zero. Zero network correction recovers the polynomial forecast,
not the target-time radar measurement.

## Architecture and time support

The plain GRU returns its final state, followed by Dense 64 ReLU and a linear
scalar. The hybrid returns sequences, then three same-padded kernel-5
convolutions with LeakyReLU 0.1. Elevation/azimuth widths are 32,64,32,
global-average pooling and Dense 64. Range widths are 64,64,32,
last-time-step selection and Dense 128. MLP: flatten, 128/64 ReLU.
CNN: two 64-channel kernel-5 same-padded ReLU layers, average pooling,
Dense 64 ReLU. All window models emit one scalar. Same padding is confined
to the observed window, not the unavailable future.

Stateful batches keep eight trajectories in fixed slots and process consecutive
non-overlapping 25-input chunks. State resets at trajectory-group boundaries;
the reset control also resets before each chunk. Gradients stop between batches.
Each real observation is processed once in a sequence. Chunk length is not a
memory horizon for the retained-state network.

For 319 original records, 318 one-step pairs form 13 chunks with seven padded
positions. Training includes real warm-up pairs. Padding has zero loss weight;
metrics separately require positive weights and target index >=25. This leaves
294 evaluated targets per retained trajectory. Standard Keras weighted-loss
reduction can include padded positions in its denominator.

Sorted incomplete eight-trajectory groups are omitted during evaluation.
The original 1,500-trajectory splits thus retain 1,496 trajectories (439,824
targets); exports identify actual retained/omitted names. Variable-length groups
raise an error rather than silently truncating. New visibility-corrected
examples can instead be analysed through their external prediction exports;
they are not substituted for the original benchmark.

## Selection and training

Both input/target StandardScalers fit training only. Original persisted scalers
are required for frozen checkpoints. Window elevation/range scaling precedes
windowing; canonical azimuth uses overlapping window-relative rows.
Focused azimuth uses the training angular residual scaler, Adam 0.001, Huber
delta 2 in standardized target units, at most 60 epochs, patience 10,
LR patience 4, and physical-RMSE min_delta 0.001 mrad. Its checkpoint and
patience track improvements exceeding that min_delta.

Controlled memory: 20 complete epochs; learning rates at zero-based epochs
0/5/10/15 are 0.001/0.0005/0.00025/0.000125. Geometric follow-up: 40 complete
epochs; the same rates start at 0/10/20/30. Both use exact minimum physical
validation RMSE checkpoints, no early stopping, identical initial weights
within each seed pair, and the same epoch ordering. Seeds default to 42/43/44.
The geometric pair shares the scaler fitted to ANGULAR training residuals,
keeping the physical Huber transition identical. Outputs record its value.

The geometric grid contains degree 1/2 crossed with lookback 5/7/13/25;
angular-median and persistence references give ten validation candidates.
Noisy spherical measurements are mapped to local Cartesian coordinates,
polynomial-fitted on normalized sample indices, and extrapolated one nominal
sample. The best geometric candidate is selected by validation circular RMSE,
then candidate name; neural fitting requires >=1% relative improvement over
the angular baseline.

Fallback: 26 validation rules (always retain/reset, four elevation thresholds,
four increment thresholds, sixteen OR combinations), ranked by mean seed RMSE,
mean seed P99, then name. Gates use noisy E[k-1] and abs(wrapped A[k-1]-A[k-2]).
They select network outputs, not the reset operation of one shared network.
The controlled family is a matched comparison; mixing focused and controlled
models is descriptive fusion of different protocols.

## Limits, not inferred results

The archived material does not establish historical test-independent
architecture selection. This companion neither invents validation logs nor
chooses architectures by test scores. Canonical new-run training settings
must be supplied explicitly; they are not claimed to recover a historical
search or interrupted deadline run.

The public package contains no historical numerical result constants or manual
curves. Private checkpoint re-evaluation and new training are distinct.
Results can vary with hardware/software and optimization. Repeated historical
test consultation prevents treating these follow-ups as untouched confirmation.

The UKF CSV cannot establish filter initialization, process/measurement
covariances, update equations or smoothing implementation. Its posterior is
not a matched-information forecast. Unit labels in old CSVs can be misleading:
the caller must declare measured, filtered and real units independently.
Single-case metrics never establish dataset-wide superiority or independence
from training without the separate physical-provenance audit.

No historical hardware throughput/latency value is reproduced by these tests.
The original simulator and the later corrected generator are intentionally
not distributed. Illustrations from new visible tracks are not new benchmark
training or evidence that the original dataset was visibility-filtered.

