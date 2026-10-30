"""Private prediction exports, common-support comparisons and title-free figures."""
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .evaluation import pooled_metrics
from .preprocessing import wrap_to_pi

KEYS = ['trajectory_id', 'target_index']


def prediction_frame(coordinate, prediction, clean, trajectory_id, target_index, **metadata):
    if coordinate not in ('elevation', 'range', 'azimuth'):
        raise ValueError(f'Unknown coordinate: {coordinate}')
    prediction, clean = np.asarray(prediction).ravel(), np.asarray(clean).ravel()
    frame = pd.DataFrame(dict(trajectory_id=np.asarray(trajectory_id).ravel(),
                              target_index=np.asarray(target_index).ravel(),
                              prediction=prediction, clean=clean))
    if not len(frame) or frame.duplicated(KEYS).any() or not np.isfinite(frame[['prediction', 'clean']]).all().all():
        raise ValueError('Prediction keys must be unique and values finite')
    frame['coordinate'] = coordinate
    error = prediction-clean
    if coordinate == 'azimuth':
        error = wrap_to_pi(error)
    frame['error'] = error*(1 if coordinate == 'range' else 1000)
    frame['unit'] = 'm' if coordinate == 'range' else 'mrad'
    for key, value in metadata.items():
        if key in frame:
            raise ValueError(f'Metadata would overwrite a prediction field: {key}')
        frame[key] = value
    return frame


def common_support(left, right):
    """Match identifiers/indices and verify the reference, never shift errors."""
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction keys')
    keys = left[KEYS].merge(right[KEYS], on=KEYS, validate='one_to_one').sort_values(KEYS)
    a = keys.merge(left, on=KEYS, validate='one_to_one')
    b = keys.merge(right, on=KEYS, validate='one_to_one')
    if not len(a) or not np.allclose(a.clean, b.clean, rtol=0, atol=1e-10):
        raise ValueError('Empty support or incompatible clean references')
    if 'time_s' in a and 'time_s' in b and not np.allclose(a.time_s, b.time_s, rtol=0, atol=1e-9):
        raise ValueError('Incompatible timestamps on common support')
    return a.reset_index(drop=True), b.reset_index(drop=True)


def metrics(frame):
    coordinates = frame.coordinate.unique()
    if len(coordinates) != 1:
        raise ValueError('Metrics require a single coordinate')
    return pooled_metrics(frame.prediction, frame.clean, coordinates[0])


def export_predictions(frame, directory, prefix):
    """Directory must be external; entry points enforce this before calling."""
    directory = Path(directory)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', prefix):
        raise ValueError('Output prefix must be a simple alphanumeric label')
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob(f'{prefix}_*')):
        raise FileExistsError(f'Output prefix already exists: {prefix}')
    frame.to_csv(directory/f'{prefix}_errors.csv', index=False)
    rows = [dict(trajectory_id=name, **metrics(group)) for name, group in frame.groupby('trajectory_id')]
    per_trajectory = pd.DataFrame(rows).sort_values('rmse', ascending=False)
    per_trajectory.to_csv(directory/f'{prefix}_per_trajectory.csv', index=False)
    frame.loc[frame.error.abs().nlargest(min(100, len(frame))).index].to_csv(
        directory/f'{prefix}_extremes.csv', index=False)
    summary = metrics(frame)
    (directory/f'{prefix}_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if 'profile' in frame:
        pd.DataFrame([dict(profile=name, **metrics(group)) for name, group in frame.groupby('profile')]).to_csv(
            directory/f'{prefix}_profiles.csv', index=False)
    return summary


def seed_summary(rows):
    """Mean and sample SD of seed-level metrics, not pooled errors across seeds."""
    table = pd.DataFrame(rows)
    result = []
    for key in ('rmse', 'mae', 'bias', 'p95', 'p99', 'maximum'):
        values = table[key].to_numpy(float)
        result.append(dict(metric=key, seeds=len(values), mean=values.mean(),
                           sample_sd=np.std(values, ddof=1) if len(values) > 1 else np.nan))
    return pd.DataFrame(result)


def fallback_rules():
    rules = ['always_retain', 'always_reset']
    rules += [f'elevation_{e}' for e in (60, 75, 80, 85)]
    rules += [f'increment_{d}' for d in (25, 50, 100, 200)]
    rules += [f'elevation_{e}_or_increment_{d}' for e in (60, 75, 80, 85) for d in (25, 50, 100, 200)]
    return rules


def fallback_mask(frame, rule):
    """True selects the independently evaluated reset-model output."""
    if rule == 'always_retain':
        return np.zeros(len(frame), bool)
    if rule == 'always_reset':
        return np.ones(len(frame), bool)
    if rule not in fallback_rules():
        raise ValueError(f'Unknown fallback rule: {rule}')
    mask = np.zeros(len(frame), bool)
    for component in rule.split('_or_'):
        kind, threshold = component.rsplit('_', 1)
        if kind == 'elevation':
            mask |= frame.input_elevation_deg.to_numpy() >= float(threshold)
        else:
            mask |= frame.input_increment_abs_mrad.to_numpy() >= float(threshold)
    return mask


def apply_fallback(retain, reset, rule):
    a, b = common_support(retain, reset)
    if not (a.coordinate.eq('azimuth').all() and b.coordinate.eq('azimuth').all()):
        raise ValueError('Fallback requires azimuth predictions')
    prediction = np.where(fallback_mask(a, rule), b.prediction, a.prediction)
    a['prediction'] = prediction
    a['error'] = 1000*wrap_to_pi(prediction-a.clean.to_numpy())
    a['reset_selected'] = fallback_mask(a, rule)
    return a


def select_fallback(validation_pairs):
    """Accept validation predictions only; rank seed means by RMSE, P99, name."""
    if not validation_pairs:
        raise ValueError('At least one validation seed pair is required')
    rows = []
    for rule in fallback_rules():
        scores = [metrics(apply_fallback(a, b, rule)) for a, b in validation_pairs]
        rows.append(dict(rule=rule, seeds=len(scores),
                         mean_rmse=np.mean([s['rmse'] for s in scores]),
                         mean_p99=np.mean([s['p99'] for s in scores])))
    table = pd.DataFrame(rows).sort_values(['mean_rmse', 'mean_p99', 'rule'])
    return table.iloc[0].rule, table


def plot_errors(frame, path, trajectory_id=None):
    """Plot all errors and ECDF, or one named trajectory; no titles/captions."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout='constrained')
    unit = frame.unit.iloc[0]
    if trajectory_id is None:
        axes[0].hist(frame.error, bins=100, density=True, color='firebrick', alpha=.7)
        axes[0].set(xlabel=f'Error ({unit})', ylabel='Density')
    else:
        part = frame[frame.trajectory_id.eq(trajectory_id)].sort_values('target_index')
        if not len(part):
            plt.close(fig)
            raise ValueError('Requested trajectory is absent')
        x = part.time_s if 'time_s' in part else part.target_index
        axes[0].plot(x, part.error, color='firebrick', label='Forecast error')
        if 'radar' in part:
            disturbance = part.radar.to_numpy()-part.clean.to_numpy()
            if part.coordinate.iloc[0] == 'azimuth':
                disturbance = wrap_to_pi(disturbance)
            disturbance *= 1 if unit == 'm' else 1000
            axes[0].plot(x, disturbance, color='.65', alpha=.6, label='Target-time noisy-clean')
        axes[0].set(xlabel='Recorded time (s)' if 'time_s' in part else 'Target index',
                    ylabel=f'Error ({unit})')
        axes[0].legend()
    absolute = np.sort(np.abs((frame if trajectory_id is None else part).error.to_numpy()))
    axes[1].plot(absolute, np.arange(1, len(absolute)+1)/len(absolute), color='firebrick')
    axes[1].set(xlabel=f'Absolute error ({unit})', ylabel='Empirical cumulative probability')
    for axis in axes:
        axis.grid(alpha=.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)
