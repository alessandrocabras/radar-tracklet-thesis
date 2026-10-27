"""Coordinate-specific preprocessing extracted from the thesis working notebooks.

See docs/SOURCE_PROVENANCE.json for exact source locations and hashes.
No trajectory generation or dataset is included.
"""
import numpy as np
import pandas as pd

WINDOW_LENGTH = 25
PREDICTION_HORIZON = 1
KINEMATIC_LOOKBACK = 7

def wrap_to_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_to_2pi(angle):
    return np.mod(np.asarray(angle), 2.0 * np.pi)


def build_elevation_features_and_target(df_noise, df_clean, horizon=1, include_cartesian_features=False):
    """
    Costruisce feature e target per elevation forecasting.

    Default: usa solo misure radar rumorose e loro delta. I campi rx/ry/rz sono esclusi
    per evitare leakage se derivano dalla traiettoria clean/simulata.
    """
    elevation = df_noise['Elevation (rad)'].to_numpy()
    azimuth = df_noise['Azimuth (rad)'].to_numpy()
    r_range = df_noise['Range (m)'].to_numpy()

    feature_columns = [
        elevation,
        np.diff(elevation, prepend=elevation[0]),
        azimuth,
        np.diff(azimuth, prepend=azimuth[0]),
        r_range,
        np.diff(r_range, prepend=r_range[0]),
    ]
    feature_names = [
        'elevation', 'd_elevation', 'azimuth', 'd_azimuth', 'range', 'd_range'
    ]

    if 'Doppler (m/s)' in df_noise.columns:
        doppler = df_noise['Doppler (m/s)'].to_numpy()
        feature_columns.extend([doppler, np.diff(doppler, prepend=doppler[0])])
        feature_names.extend(['doppler', 'd_doppler'])

    if include_cartesian_features:
        for col, name in [('rx (m)', 'rx'), ('ry (m)', 'ry'), ('rz (m)', 'rz')]:
            values = df_noise[col].to_numpy()
            feature_columns.extend([values, np.diff(values, prepend=values[0])])
            feature_names.extend([name, f'd_{name}'])

    clean_elevation = df_clean['Elevation (rad)'].to_numpy()
    max_len = min(*(len(col) for col in feature_columns), len(clean_elevation) - horizon)

    features_raw = np.stack([col[:max_len] for col in feature_columns], axis=-1)
    target_raw = clean_elevation[horizon:horizon + max_len]
    return features_raw, target_raw, feature_names


def causal_poly_residual(values, lookback=24, degree=2):
    # Causal local residual: deviation of the current radar range from a trend
    # estimated using only previous samples in the same tracklet.
    values = np.asarray(values, dtype=np.float64)
    residual = np.zeros_like(values, dtype=np.float64)
    for i in range(len(values)):
        start = max(0, i - lookback)
        hist = values[start:i]
        if len(hist) <= degree:
            residual[i] = 0.0
            continue
        x_hist = np.arange(start, i, dtype=np.float64)
        deg = min(degree, len(hist) - 1)
        coeff = np.polyfit(x_hist, hist, deg=deg)
        trend_i = np.polyval(coeff, float(i))
        residual[i] = values[i] - trend_i
    return residual


def causal_poly_forecast(values, horizon=1, lookback=25, degree=2):
    """Forecast from samples available up to and including index i; no future leakage."""
    if horizon < 1:
        raise ValueError('Forecast horizon must be >= 1.')
    values = np.asarray(values, dtype=np.float64)
    forecast = np.empty_like(values, dtype=np.float64)
    for i in range(len(values)):
        start = max(0, i - lookback + 1)
        history = values[start:i + 1]
        if len(history) == 1:
            forecast[i] = history[-1]
            continue
        x_history = np.arange(start, i + 1, dtype=np.float64)
        local_degree = min(degree, len(history) - 1)
        coefficients = np.polyfit(x_history, history, deg=local_degree)
        forecast[i] = np.polyval(coefficients, float(i + horizon))
    return forecast


def build_range_features_and_target(df_noise, df_clean, horizon=PREDICTION_HORIZON):
    """Strict t+1 target: clean future range minus a causal forecast available at t."""
    if horizon < 1:
        raise ValueError('Canonical thesis protocol requires horizon >= 1.')
    r_range = df_noise['Range (m)'].to_numpy(dtype=np.float64)
    elevation = df_noise['Elevation (rad)'].to_numpy(dtype=np.float64)
    azimuth = np.unwrap(df_noise['Azimuth (rad)'].to_numpy(dtype=np.float64))
    range_clean = df_clean['Range (m)'].to_numpy(dtype=np.float64)
    range_local_residual = causal_poly_residual(r_range, lookback=24, degree=2)
    feature_columns = [
        elevation, np.diff(elevation, prepend=elevation[0]),
        azimuth, np.diff(azimuth, prepend=azimuth[0]),
        r_range, np.diff(r_range, prepend=r_range[0]), range_local_residual,
    ]
    feature_names = ['elevation', 'd_elevation', 'azimuth_unwrapped', 'd_azimuth', 'range', 'd_range', 'range_local_residual']
    if 'Doppler (m/s)' in df_noise.columns:
        doppler = df_noise['Doppler (m/s)'].to_numpy(dtype=np.float64)
        feature_columns.extend([doppler, np.diff(doppler, prepend=doppler[0])])
        feature_names.extend(['doppler', 'd_doppler'])
    max_len = min(*(len(col) for col in feature_columns), len(range_clean) - horizon, len(r_range) - horizon)
    features_raw = np.stack([col[:max_len] for col in feature_columns], axis=-1)
    # At feature index i the input contains measurements only through time i.
    # The causal baseline forecasts range at i+horizon using that available history.
    target_clean = range_clean[horizon:horizon + max_len]
    target_baseline = causal_poly_forecast(
        r_range, horizon=horizon, lookback=WINDOW_LENGTH, degree=2
    )[:max_len]
    target_noise = r_range[horizon:horizon + max_len]
    target_error = target_clean - target_baseline
    return features_raw, target_error, target_baseline, target_noise, feature_names


def build_azimuth_windows(df_noise, df_clean, window_length, step_size, horizon=PREDICTION_HORIZON):
    if horizon < 1:
        raise ValueError('Canonical thesis protocol requires horizon >= 1.')
    az_noise = np.unwrap(df_noise['Azimuth (rad)'].to_numpy(dtype=np.float64))
    az_clean = np.unwrap(df_clean['Azimuth (rad)'].to_numpy(dtype=np.float64))
    elevation = df_noise['Elevation (rad)'].to_numpy(dtype=np.float64)
    r_range = df_noise['Range (m)'].to_numpy(dtype=np.float64)
    d_azimuth = np.diff(az_noise, prepend=az_noise[0])
    d_elevation = np.diff(elevation, prepend=elevation[0])
    d_range = np.diff(r_range, prepend=r_range[0])
    doppler = df_noise['Doppler (m/s)'].to_numpy(dtype=np.float64) if 'Doppler (m/s)' in df_noise else np.zeros_like(r_range)
    n = min(len(az_noise), len(az_clean), len(elevation), len(r_range), len(doppler))
    max_start = n - horizon - window_length
    starts = np.arange(0, max_start + 1, step_size, dtype=int) if max_start >= 0 else np.asarray([], dtype=int)
    X, y, pivots, target_indices, clean_targets, noisy_targets = [], [], [], [], [], []
    for start in starts:
        stop = start + window_length
        target_index = stop - 1 + horizon
        pivot = az_noise[start]
        X.append(np.stack([
            az_noise[start:stop] - pivot,
            d_azimuth[start:stop], elevation[start:stop],
            d_elevation[start:stop], d_range[start:stop], doppler[start:stop],
        ], axis=-1))
        y.append([az_clean[target_index] - pivot])
        pivots.append(pivot)
        target_indices.append(target_index)
        clean_targets.append(az_clean[target_index])
        noisy_targets.append(az_noise[target_index])
    return (np.asarray(X), np.asarray(y), np.asarray(pivots), np.asarray(target_indices),
            np.asarray(clean_targets), np.asarray(noisy_targets))


def rolling_median_angular_velocity(azimuth_unwrapped, lookback=KINEMATIC_LOOKBACK):
    delta = wrap_to_pi(np.diff(azimuth_unwrapped, prepend=azimuth_unwrapped[0]))
    robust_velocity = np.zeros_like(delta)
    for index in range(1, len(delta)):
        start = max(1, index - lookback + 1)
        robust_velocity[index] = np.median(delta[start:index + 1])
    return delta, robust_velocity


def build_azimuth_residual_series(df_noise, df_clean):
    """Causal features at t and wrapped residual target for clean azimuth at t+1."""
    azimuth = np.unwrap(pd.to_numeric(df_noise['Azimuth (rad)'], errors='coerce').to_numpy(float))
    clean = np.unwrap(pd.to_numeric(df_clean['Azimuth (rad)'], errors='coerce').to_numpy(float))
    elevation = pd.to_numeric(df_noise['Elevation (rad)'], errors='coerce').to_numpy(float)
    range_m = pd.to_numeric(df_noise['Range (m)'], errors='coerce').to_numpy(float)
    doppler = (pd.to_numeric(df_noise['Doppler (m/s)'], errors='coerce').to_numpy(float)
               if 'Doppler (m/s)' in df_noise.columns else np.zeros_like(range_m))
    n = min(map(len, (azimuth, clean, elevation, range_m, doppler)))
    azimuth, clean = azimuth[:n], clean[:n]
    elevation, range_m, doppler = elevation[:n], range_m[:n], doppler[:n]
    angular_velocity, robust_velocity = rolling_median_angular_velocity(azimuth)
    angular_acceleration = np.diff(angular_velocity, prepend=angular_velocity[0])
    d_elevation = np.diff(elevation, prepend=elevation[0])
    d_range = np.diff(range_m, prepend=range_m[0])
    baseline_next = azimuth + robust_velocity
    target_indices = np.arange(1, n, dtype=int)
    input_indices = target_indices - 1
    features = np.stack([
        np.sin(azimuth[input_indices]), np.cos(azimuth[input_indices]),
        angular_velocity[input_indices], angular_acceleration[input_indices],
        robust_velocity[input_indices], elevation[input_indices],
        d_elevation[input_indices], d_range[input_indices], doppler[input_indices],
    ], axis=-1)
    baseline = baseline_next[input_indices]
    target_residual = wrap_to_pi(clean[target_indices] - baseline)
    return {
        'features': features.astype(np.float32),
        'target_residual': target_residual.astype(np.float32),
        'baseline': baseline.astype(np.float64),
        'clean': clean[target_indices].astype(np.float64),
        'radar': azimuth[target_indices].astype(np.float64),
        'target_indices': target_indices,
        'feature_names': [
            'sin_azimuth', 'cos_azimuth', 'wrapped_angular_velocity',
            'angular_acceleration', 'robust_angular_velocity', 'elevation',
            'd_elevation', 'd_range', 'doppler',
        ],
    }

