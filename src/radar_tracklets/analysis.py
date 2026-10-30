"""Read-only analysis of external exports; no fabricated curves or filter model."""
import numpy as np
import pandas as pd
from .experiments import external_path, write_json
from .geometry import joint_errors
from .reporting import (KEYS, prediction_frame, common_support, metrics,
                        export_predictions, plot_errors, select_fallback, apply_fallback)


def load_predictions(path, coordinate):
    source = pd.read_csv(external_path(path))
    source = source.rename(columns={'timestamp_s': 'time_s', 'noise_profile': 'profile'})
    metadata = {c: source[c].to_numpy() for c in source if c not in
                KEYS+['prediction', 'clean', 'coordinate', 'error', 'unit']}
    return prediction_frame(coordinate, source.prediction, source.clean,
                             source.trajectory_id, source.target_index, **metadata)


def compare_common_support(frames, output_directory):
    """Descriptive comparison; no attribution of gains to statefulness alone."""
    if not frames:
        raise ValueError('At least one prediction export is required')
    out = external_path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    indices = [pd.MultiIndex.from_frame(frame[KEYS]) for frame in frames.values()]
    common = indices[0]
    for index in indices[1:]:
        common = common.intersection(index)
    if not len(common):
        raise ValueError('No shared target support')
    rows, reference = [], None
    for label, frame in frames.items():
        selected = frame.set_index(KEYS).loc[common].reset_index()
        if reference is not None:
            common_support(reference, selected)
        reference = selected
        rows.append(dict(configuration=label, original_n=len(frame), **export_predictions(selected, out, label)))
        plot_errors(selected, out/f'{label}_distribution.pdf')
    table = pd.DataFrame(rows)
    table.to_csv(out/'common_support_metrics.csv', index=False)
    common.to_frame(index=False).to_csv(out/'common_support.csv', index=False)
    return table


def evaluate_fallback(validation_pairs, test_pairs, output_directory):
    """One rule across seeds; validation and test must be passed separately."""
    if len(validation_pairs) != len(test_pairs):
        raise ValueError('Validation and test require matched seed counts')
    out = external_path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    rule, candidates = select_fallback(validation_pairs)
    candidates.to_csv(out/'validation_candidates.csv', index=False)
    write_json(out/'selection.json', dict(rule=rule, selection_split='validation',
        interpretation='fusion of separately evaluated networks; not dynamic state resetting'))
    rows = []
    for i, (retain, reset) in enumerate(test_pairs):
        frame = apply_fallback(retain, reset, rule)
        rows.append(dict(pair_index=i, **export_predictions(frame, out, f'test_pair{i}')))
    pd.DataFrame(rows).to_csv(out/'test_metrics.csv', index=False)
    return rule


def spatial_metrics(azimuth, elevation, range_frame, output_directory):
    out = external_path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    frames = []
    for coordinate, frame in zip(('azimuth', 'elevation', 'range'), (azimuth, elevation, range_frame)):
        if not frame.coordinate.eq(coordinate).all() or frame.duplicated(KEYS).any():
            raise ValueError('Expected unique keys and the specified coordinate')
        frames.append(frame.set_index(KEYS))
    common = frames[0].index.intersection(frames[1].index).intersection(frames[2].index)
    if not len(common):
        raise ValueError('No shared spatial support')
    a, e, r = [f.loc[common] for f in frames]
    if all('time_s' in f for f in (a, e, r)):
        for f in (e, r):
            np.testing.assert_allclose(a.time_s, f.time_s, atol=1e-9, rtol=0)
    direction, position = joint_errors(a.prediction.to_numpy(), e.prediction.to_numpy(), r.prediction.to_numpy(),
                                       a.clean.to_numpy(), e.clean.to_numpy(), r.clean.to_numpy())
    samples = common.to_frame(index=False)
    samples['direction_error_mrad'], samples['position_error_m'] = direction, position
    samples['clean_elevation_rad'] = e.clean.to_numpy()
    samples.to_csv(out/'spatial_errors.csv', index=False)
    rows = []
    for condition, mask in (('all', np.ones(len(a), bool)), ('clean_elevation_ge_0', e.clean.to_numpy() >= 0),
                             ('clean_elevation_ge_85', e.clean.to_numpy() >= np.radians(85))):
        for name, error, unit in (('direction', direction, 'mrad'), ('position', position, 'm')):
            if not mask.any():
                rows.append(dict(condition=condition, quantity=name, unit=unit, n=0))
                continue
            score = metrics(prediction_frame('range', error[mask], np.zeros(mask.sum()),
                    samples.trajectory_id.to_numpy()[mask], samples.target_index.to_numpy()[mask]))
            score['unit'] = unit
            rows.append(dict(condition=condition, quantity=name, **score))
    table = pd.DataFrame(rows)
    table.to_csv(out/'spatial_metrics.csv', index=False)
    write_json(out/'protocol.json', dict(support='intersection of three predicted coordinates',
        interpretation='local radar-frame position, not propagated orbital-state error',
        clean_coordinates_used_as_predictors=False))
    return table


def ukf_posterior_comparison(neural, csv_path, output_directory, *, filtered_column,
                             measured_column, real_column, filtered_unit, measured_unit,
                             real_unit, time_column='time[s]', warmup_target_index=50,
                             timestamp_tolerance_s=1e-6, reference_atol=1e-8):
    """Explicit-unit posterior comparison. Never infer units from a column label."""
    if neural.trajectory_id.nunique() != 1 or 'time_s' not in neural:
        raise ValueError('One named trajectory with recorded timestamps is required')
    coordinate = neural.coordinate.iloc[0]
    ukf = pd.read_csv(external_path(csv_path))
    times = ukf[time_column].to_numpy(float)
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('UKF timestamps must be finite, unique and increasing')
    def convert(column, unit):
        expected = ('m',) if coordinate == 'range' else ('rad', 'deg')
        if unit not in expected:
            raise ValueError('Explicit physical units are required for each UKF column')
        values = ukf[column].to_numpy(float)
        return np.radians(values) if unit == 'deg' else values
    filtered, measured, real = (convert(column, unit) for column, unit in (
        (filtered_column, filtered_unit), (measured_column, measured_unit), (real_column, real_unit)))
    selected = neural[neural.target_index >= warmup_target_index].copy()
    if not len(selected):
        raise ValueError('No neural targets survive the declared warm-up')
    delta = np.abs(selected.time_s.to_numpy()[:, None]-times[None, :])
    matches = delta.argmin(axis=1)
    keep = delta[np.arange(len(selected)), matches] <= timestamp_tolerance_s
    selected, matches = selected.loc[keep].reset_index(drop=True), matches[keep]
    if not len(selected) or len(set(matches)) != len(matches):
        raise ValueError('Empty or non-unique timestamp support')
    reference_error = real[matches]-selected.clean.to_numpy()
    if coordinate == 'azimuth':
        from .preprocessing import wrap_to_pi
        reference_error = wrap_to_pi(reference_error)
    if np.max(np.abs(reference_error)) > reference_atol:
        raise ValueError('UKF and neural clean references differ; do not realign by score')
    posterior = prediction_frame(coordinate, filtered[matches], selected.clean,
                                 selected.trajectory_id, selected.target_index,
                                 time_s=selected.time_s.to_numpy(), radar=measured[matches])
    out = external_path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    results = [dict(method='neural_one_step_forecast', **export_predictions(selected, out, 'neural')),
               dict(method='ukf_posterior', **export_predictions(posterior, out, 'ukf_posterior'))]
    pd.DataFrame(results).to_csv(out/'comparison.csv', index=False)
    write_json(out/'loader.json', dict(filtered_column=filtered_column, filtered_unit=filtered_unit,
        measured_column=measured_column, measured_unit=measured_unit, real_column=real_column,
        real_unit=real_unit, time_column=time_column, warmup_target_index=warmup_target_index,
        timestamp_tolerance_s=timestamp_tolerance_s, reference_atol=reference_atol,
        matched_n=len(selected), information_sets_matched=False,
        ukf_configuration='not reconstructed: external posterior output only'))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout='constrained')
    for frame, label, color in ((selected, 'GRU forecast', 'firebrick'),
                                 (posterior, 'Posterior UKF', 'purple')):
        axes[0].plot(frame.time_s, frame.error, label=label, color=color)
        axes[1].hist(frame.error, bins=40, density=True, alpha=.45, label=label, color=color)
    axes[0].set(xlabel='Recorded time (s)', ylabel=f'Error ({selected.unit.iloc[0]})')
    axes[1].set(xlabel=f'Error ({selected.unit.iloc[0]})', ylabel='Density')
    for axis in axes:
        axis.legend(); axis.grid(alpha=.25)
    fig.savefig(out/'posterior_vs_forecast.pdf')
    plt.close(fig)
    return pd.DataFrame(results)
