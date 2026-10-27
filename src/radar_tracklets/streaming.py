"""Final azimuth chunking, recurrent state management and physical evaluation.

Source functions retained verbatim; this module is not an orbital simulator.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from .preprocessing import wrap_to_pi

SEED = 42
WINDOW_LENGTH = 25
STREAM_CHUNK_LENGTH = 25
STREAM_BATCH_TRAJECTORIES = 8

def fit_residual_scalers(train_sequences):
    scaler_features, scaler_target = StandardScaler(), StandardScaler()
    for series in train_sequences.values():
        scaler_features.partial_fit(series['features'])
        scaler_target.partial_fit(series['target_residual'].reshape(-1, 1))
    return scaler_features, scaler_target


def chunk_sequence(series, scaler_features, scaler_target, chunk_length=STREAM_CHUNK_LENGTH):
    features = scaler_features.transform(series['features']).astype(np.float32)
    target = scaler_target.transform(series['target_residual'].reshape(-1, 1)).astype(np.float32)
    n_chunks = int(np.ceil(len(features) / chunk_length))
    padded_length = n_chunks * chunk_length
    pad = padded_length - len(features)
    X = np.pad(features, ((0, pad), (0, 0))).reshape(n_chunks, chunk_length, features.shape[1])
    y = np.pad(target, ((0, pad), (0, 0))).reshape(n_chunks, chunk_length, 1)
    weights = np.pad(np.ones(len(features), np.float32), (0, pad)).reshape(n_chunks, chunk_length)
    metadata = {}
    for key in ('baseline', 'clean', 'radar', 'target_indices'):
        values = np.asarray(series[key])
        fill = -1 if key == 'target_indices' else np.nan
        metadata[key] = np.pad(values, (0, pad), constant_values=fill).reshape(n_chunks, chunk_length)
    return {'X': X, 'y': y, 'weights': weights, **metadata}


def physical_predictions(pred_scaled, data, target_scaler):
    pred_residual = target_scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()
    baseline = np.asarray(data['baseline']).reshape(-1)
    clean = np.asarray(data['clean']).reshape(-1)
    target_indices = np.asarray(data['target_indices']).reshape(-1)
    weights = np.asarray(data['weights']).reshape(-1)
    valid = (weights > 0) & np.isfinite(baseline) & np.isfinite(clean) & (target_indices >= WINDOW_LENGTH)
    prediction = baseline + pred_residual
    error_mrad = wrap_to_pi(prediction[valid] - clean[valid]) * 1000.0
    return prediction, error_mrad, valid


def reset_states(model_variant):
    for layer in model_variant.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()


def stream_groups(chunk_dict, batch_trajectories, shuffle=False, seed=SEED):
    names = sorted(chunk_dict)
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(names)
    usable = len(names) - len(names) % batch_trajectories
    used_names = names[:usable]
    for group_start in range(0, usable, batch_trajectories):
        group = used_names[group_start:group_start + batch_trajectories]
        common_chunks = min(len(chunk_dict[name]['X']) for name in group)
        yield 'reset', group, None
        for chunk_index in range(common_chunks):
            batch = {key: np.stack([chunk_dict[name][key][chunk_index] for name in group])
                     for key in ('X', 'y', 'weights', 'baseline', 'clean', 'radar', 'target_indices')}
            yield 'batch', group, batch


def stateful_epoch(model_variant, chunk_dict, training, epoch_seed):
    weighted, count = 0.0, 0.0
    for kind, _, batch in stream_groups(chunk_dict, STREAM_BATCH_TRAJECTORIES,
                                        shuffle=training, seed=epoch_seed):
        if kind == 'reset':
            reset_states(model_variant); continue
        model_variant.reset_metrics()
        result = (model_variant.train_on_batch(batch['X'], batch['y'], sample_weight=batch['weights'], return_dict=True)
                  if training else model_variant.test_on_batch(batch['X'], batch['y'], sample_weight=batch['weights'], return_dict=True))
        n = float(np.sum(batch['weights']))
        weighted += float(result['loss']) * n; count += n
    reset_states(model_variant); model_variant.reset_metrics()
    return weighted / count if count else np.nan


def predict_stateful_stream(model_variant, chunk_dict):
    predictions, metadata = [], {key: [] for key in ('y', 'weights', 'baseline', 'clean', 'radar', 'target_indices')}
    used_names = []
    for kind, group, batch in stream_groups(chunk_dict, STREAM_BATCH_TRAJECTORIES, shuffle=False):
        if kind == 'reset':
            reset_states(model_variant); used_names.extend(group); continue
        predictions.append(model_variant.predict_on_batch(batch['X']))
        for key in metadata: metadata[key].append(batch[key])
    reset_states(model_variant)
    output = {key: np.concatenate(values) for key, values in metadata.items()}
    return np.concatenate(predictions), output, sorted(set(used_names))

