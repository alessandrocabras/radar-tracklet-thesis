"""Measurement-domain geometry; no orbital propagation or data generation."""
import numpy as np
from .preprocessing import build_azimuth_residual_series, wrap_to_pi


def unit_vector(azimuth, elevation):
    a, e = np.broadcast_arrays(azimuth, elevation)
    return np.stack((np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)), axis=-1)


def cartesian_forecast(noisy, degree=1, lookback=25):
    """One-step polynomial extrapolation of noisy local Cartesian measurements.

    Time is a normalized sample index, not an irregular-timestamp fit.
    Prefixes use reduced degree; one sample gives persistence.
    """
    if degree not in (1, 2) or lookback not in (5, 7, 13, 25):
        raise ValueError("Expected degree 1/2 and lookback 5/7/13/25")
    xyz = unit_vector(noisy['Azimuth (rad)'].to_numpy(float),
                      noisy['Elevation (rad)'].to_numpy(float))
    xyz *= noisy['Range (m)'].to_numpy(float)[:, None]
    output = np.empty_like(xyz)
    for t in range(min(lookback-1, len(xyz))):
        count = t+1
        if count == 1:
            output[t] = xyz[t]
            continue
        x = np.arange(-count+1, 1, dtype=float)/(count-1)
        p = min(degree, count-1)
        weights = (1/(count-1))**np.arange(p+1) @ np.linalg.pinv(
            np.vander(x, p+1, increasing=True))
        output[t] = weights @ xyz[:count]
    if len(xyz) >= lookback:
        x = np.arange(-lookback+1, 1, dtype=float)/(lookback-1)
        weights = (1/(lookback-1))**np.arange(degree+1) @ np.linalg.pinv(
            np.vander(x, degree+1, increasing=True))
        windows = np.lib.stride_tricks.sliding_window_view(xyz, lookback, axis=0)
        output[lookback-1:] = np.einsum('l,tcl->tc', weights, windows)
    return output


def baseline_names():
    return ['angular_median7', 'persistence'] + [
        f'cartesian_degree{degree}_L{length}' for degree in (1, 2) for length in (5, 7, 13, 25)]


def residual_series(noisy, clean, baseline='angular_median7'):
    """Keep the same nine inputs; replace only the residual reference."""
    series = build_azimuth_residual_series(noisy, clean)
    if baseline == 'angular_median7':
        return series
    if baseline not in baseline_names():
        raise ValueError(f'Unknown baseline: {baseline}')
    if baseline == 'persistence':
        values = noisy['Azimuth (rad)'].to_numpy(float)[:-1]
    else:
        kind, length = baseline.split('_L')
        xyz = cartesian_forecast(noisy, int(kind[-1]), int(length))
        values = np.arctan2(xyz[:-1, 1], xyz[:-1, 0])
    series['baseline'] = values
    series['target_residual'] = wrap_to_pi(series['clean']-values).astype(np.float32)
    return series


def joint_errors(azimuth_prediction, elevation_prediction, range_prediction,
                 azimuth_clean, elevation_clean, range_clean):
    """All three predicted coordinates are required; no clean substitution."""
    up = unit_vector(azimuth_prediction, elevation_prediction)
    uc = unit_vector(azimuth_clean, elevation_clean)
    direction = 1000*np.arctan2(np.linalg.norm(np.cross(up, uc), axis=-1),
                              np.sum(up*uc, axis=-1))
    position = np.linalg.norm(np.asarray(range_prediction)[..., None]*up -
                              np.asarray(range_clean)[..., None]*uc, axis=-1)
    return direction, position
