# Private input interface

Keep private material **outside the Git working tree**. Repository ignore rules
are defence in depth, not a permission decision or a guarantee against leaks.

The notebooks default to no private execution. With authorization, supply a
directory containing `noise/` and `clean/` and the original private JSON split
manifest. Required top-level manifest keys are `train_files`, `val_files`,
`test_files`: each is a nonempty list of noisy CSV basenames. A corresponding
clean filename replaces `_noise` with `_clean`. Missing files and overlapping
split names raise errors. No new random split is inferred. Filename separation
alone is not proof of physical independence; the original audit is separate.

Expected CSV observables:

| Column | Units / interpretation |
| --- | --- |
| `Elevation (rad)` | elevation in radians |
| `Azimuth (rad)` | azimuth in radians |
| `Range (m)` | slant range in metres |
| `Doppler (m/s)` | radial velocity, not frequency in Hz |
| `Time (s)` | optional recorded time; positive increasing cadence |

The review adapters require all four observables explicitly, even though some
original notebook functions allowed missing Doppler. Cartesian coordinates
are not used. No radar-frequency conversion is applied to radial velocity.
No coordinate transformation or orbit propagation is distributed.

The arrays are paired by original sample index, not re-aligned using noisy
values. `window_arrays` exposes `target_indices`, baseline/pivot, clean target
and trajectory identifier. They remain private when using real input files.
The metadata field `radar` in streaming is the target-time measurement for
diagnostics; it is never a predictor feature or used to reconstruct a forecast.

`prepare_windows` fits both scalers on training only. Elevation/range scaler
statistics use raw pre-window rows; pivot-azimuth statistics use the
window-relative arrays (with overlap multiplicity), matching the source.
Validation/test never refit a scaler. For final stateful evaluation supply
the persisted nine-feature input scaler, one-target residual scaler and a
compatible `stateful_dense_huber_d2` weight file. The example notebook loads
these only after explicit opt-in. Only load pickle/joblib files you trust.

For a new final-azimuth training run, use `build_azimuth_residual_series` on
each original split, fit `streaming.fit_residual_scalers` on training sequences
only, and apply `streaming.chunk_sequence` to each split with those scalers.
Pass only training and validation chunk dictionaries to
`training.fit_final_azimuth`. It selects the physical-validation-RMSE checkpoint
and returns in-memory history without writing files. The test split is evaluated
afterwards with `predict_stateful_stream` and `physical_predictions`.

No dataset or split manifest can be downloaded from this repository. No
weights are bundled. Public tests do not need them. New experiment outputs
should be stored outside this repository and labelled as new runs, not as
reproductions of historical tables.

