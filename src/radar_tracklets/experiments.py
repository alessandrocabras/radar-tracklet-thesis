"""Opt-in private-data workflows. Outputs always go outside the repository."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import uuid
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .evaluation import reconstruct
from .geometry import baseline_names, residual_series
from .preprocessing import wrap_to_pi
from .reporting import (prediction_frame, export_predictions, metrics, plot_errors,
                        seed_summary, select_fallback, apply_fallback)
from .streaming import (chunk_sequence, fit_residual_scalers, physical_predictions,
                        predict_stateful_stream)
from .workflows import read_manifest, read_pair, window_arrays, prepare_windows

REPOSITORY = Path(__file__).resolve().parents[2]


def external_path(value):
    path = Path(value).expanduser().resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError('Private inputs and outputs must be outside the repository')
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def weights_hash(weights):
    return hashlib.sha256(b''.join(w.tobytes() for w in weights)).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def load_pairs(config, split, sort=False):
    names = read_manifest(config['manifest'])[split]
    if sort:
        names = sorted(names)
    return {name: read_pair(config['data_root'], name) for name in names}


def add_diagnostics(frame, pairs):
    parts = []
    for name, group in frame.groupby('trajectory_id', sort=False):
        noisy, clean = pairs[name]
        k = group.target_index.to_numpy(int)
        part = group.copy()
        part['input_index'] = k-1
        if 'Time (s)' in noisy:
            part['time_s'] = noisy['Time (s)'].to_numpy()[k]
        part['profile'] = noisy.noise_profile.to_numpy()[k] if 'noise_profile' in noisy else 'unknown'
        part['input_elevation_deg'] = np.degrees(noisy['Elevation (rad)'].to_numpy()[k-1])
        part['clean_elevation_deg'] = np.degrees(clean['Elevation (rad)'].to_numpy()[k])
        a = noisy['Azimuth (rad)'].to_numpy()
        part['input_increment_abs_mrad'] = 1000*np.abs(wrap_to_pi(a[k-1]-a[k-2]))
        column = {'azimuth': 'Azimuth (rad)', 'elevation': 'Elevation (rad)', 'range': 'Range (m)'}[
            part.coordinate.iloc[0]]
        part['radar'] = noisy[column].to_numpy()[k]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def stream_frame(model, chunks, scaler_y, pairs, retain_state=True):
    pred, data, retained = predict_stateful_stream(model, chunks, retain_state)
    prediction, _, valid = physical_predictions(pred, data, scaler_y)
    frame = prediction_frame('azimuth', prediction[valid], data['clean'].ravel()[valid],
                             data['trajectory_id'].ravel()[valid], data['target_indices'].ravel()[valid])
    return add_diagnostics(frame, pairs), retained


def make_chunks(pairs, sx, sy, baseline='angular_median7'):
    return {name: chunk_sequence(residual_series(noisy, clean, baseline), sx, sy)
            for name, (noisy, clean) in pairs.items()}


def baseline_frame(pairs, baseline):
    names = sorted(pairs)
    names = names[:len(names)//8*8]
    if not names:
        raise ValueError('Baseline comparison requires at least eight trajectories')
    frames = []
    for name in names:
        series = residual_series(*pairs[name], baseline)
        keep = series['target_indices'] >= 25
        frames.append(prediction_frame('azimuth', series['baseline'][keep], series['clean'][keep],
                                      np.repeat(name, keep.sum()), series['target_indices'][keep]))
    return pd.concat(frames, ignore_index=True)


def select_baseline(validation_pairs):
    rows = [dict(candidate=name, **metrics(baseline_frame(validation_pairs, name))) for name in baseline_names()]
    table = pd.DataFrame(rows)
    selected = table[table.candidate.str.startswith('cartesian')].sort_values(['rmse', 'candidate']).iloc[0]
    angular = table.set_index('candidate').loc['angular_median7', 'rmse']
    gain = 1-float(selected.rmse)/float(angular) if float(angular) > 0 else None
    return dict(baseline=selected.candidate, relative_validation_gain=gain,
                required_relative_gain=.01, admitted=bool(gain is not None and gain >= .01),
                selection_split='validation'), table


def save_report(frame, out, prefix, figures=True):
    summary = export_predictions(frame, out, prefix)
    if figures:
        plot_errors(frame, out/f'{prefix}_distribution.pdf')
        worst = frame.assign(squared=frame.error**2).groupby('trajectory_id').squared.mean().idxmax()
        plot_errors(frame, out/f'{prefix}_worst.pdf', worst)
    return summary


def run_canonical(config, out):
    """Frozen evaluation by default; full-array training requires explicit opt-in."""
    import tensorflow as tf
    from .models import build_window_model, LastTimeStep
    coordinate = config['coordinate']
    architecture = config.get('architecture', 'GRU')
    if config.get('execute_training', False):
        if config.get('model'):
            raise ValueError('Choose new training or a frozen model, not both')
        training = config['training']
        data, sx, sy = prepare_windows(config['data_root'], config['manifest'], coordinate)
        model = build_window_model(coordinate, architecture, config.get('seed', 42))
        history = model.fit(data['train_files']['X'], data['train_files']['y'],
                            validation_data=(data['val_files']['X'], data['val_files']['y']),
                            epochs=training['epochs'], batch_size=training['batch_size'], shuffle=True,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                patience=training['patience'], min_delta=training.get('min_delta', 0),
                                restore_best_weights=True),
                                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                patience=training['lr_patience'], factor=.5, min_lr=1e-6)], verbose=2)
        pd.DataFrame(history.history).to_csv(out/'training_history.csv', index=False)
        model.save(out/'best.keras')
        joblib.dump(sx, out/'scaler_X.pkl'); joblib.dump(sy, out/'scaler_y.pkl')
        del data
    else:
        sx, sy = joblib.load(config['scaler_X']), joblib.load(config['scaler_y'])
        model = tf.keras.models.load_model(config['model'], compile=False,
                    custom_objects={'LastTimeStep': LastTimeStep, 'DebrisTracking>LastTimeStep': LastTimeStep})
    pairs = load_pairs(config, 'test_files')
    frames, baseline_frames = [], []
    for name, (noisy, clean) in pairs.items():
        data = window_arrays(coordinate, noisy, clean)
        shape = data['X'].shape
        x = sx.transform(data['X'].reshape(-1, shape[-1])).reshape(shape).astype('float32')
        pred = model.predict(x, batch_size=64, verbose=0)
        physical = reconstruct(coordinate, pred, sy, data.get('baseline'), data.get('pivot'))
        frames.append(prediction_frame(coordinate, physical, data['clean'],
                                      np.repeat(name, len(physical)), data['target_indices']))
        if coordinate == 'range':
            baseline_frames.append(prediction_frame(coordinate, data['baseline'], data['clean'],
                                      np.repeat(name, len(physical)), data['target_indices']))
    frame = add_diagnostics(pd.concat(frames, ignore_index=True), pairs)
    result = save_report(frame, out, 'canonical', config.get('figures', True))
    if baseline_frames:
        save_report(pd.concat(baseline_frames, ignore_index=True), out, 'range_zero_correction', False)
    return result


def run_azimuth(config, out):
    """Focused fit/evaluation or matched fixed-budget seed pairs."""
    import tensorflow as tf
    from .models import build_final_azimuth
    from .training import fit_final_azimuth, fit_fixed_budget
    mode = config['mode']
    training_enabled = config.get('execute_training', False)
    if mode in ('memory20', 'geometry40') and not training_enabled:
        raise ValueError('Fixed-budget training requires execute_training=true; use exports for archived runs')
    seeds = config.get('seeds', [42, 43, 44])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    pairs = {split: load_pairs(config, split, sort=mode == 'geometry40')
             for split in (('train_files', 'val_files') if training_enabled else ())}
    if training_enabled:
        series = {name: residual_series(*pair) for name, pair in pairs['train_files'].items()}
        if mode == 'geometry40':
            sx = StandardScaler().fit(np.concatenate([s['features'] for s in series.values()]))
            sy = StandardScaler().fit(np.concatenate([s['target_residual'] for s in series.values()])[:, None])
        else:
            sx, sy = fit_residual_scalers(series)
        del series
        joblib.dump(sx, out/'scaler_X.pkl'); joblib.dump(sy, out/'scaler_y.pkl')
    else:
        sx, sy = joblib.load(config['scaler_X']), joblib.load(config['scaler_y'])
    write_json(out/'scaling.json', dict(input_features=9, target='wrapped angular residual',
                huber_delta_standardized=2, huber_transition_rad=float(2*sy.scale_[0]),
                target_scaler_source='training angular residuals; shared across paired variants'))
    variants = [('focused', 'angular_median7', True)]
    if mode == 'memory20':
        variants = [('retain', 'angular_median7', True), ('reset', 'angular_median7', False)]
    elif mode == 'geometry40':
        decision, table = select_baseline(pairs['val_files'])
        table.to_csv(out/'baseline_validation_candidates.csv', index=False)
        write_json(out/'baseline_selection.json', decision)
        if not decision['admitted']:
            return dict(status='SKIPPED_BY_VALIDATION', **decision)
        variants = [('angular_control', 'angular_median7', True),
                    ('geometric_residual', decision['baseline'], True)]
    by_variant = {label: [] for label, _, _ in variants}
    validations = {}
    for seed in seeds if training_enabled else []:
        tf.keras.backend.clear_session()
        template = build_final_azimuth(seed)
        initial = template.get_weights()
        initial_hash = weights_hash(initial)
        del template
        for label, baseline, retain in variants:
            folder = out/f'{label}_seed{seed}'
            folder.mkdir()
            chunks = {split: make_chunks(raw, sx, sy, baseline) for split, raw in pairs.items()}
            if mode == 'focused':
                model, history, audit = fit_final_azimuth(chunks['train_files'], chunks['val_files'], sy, seed)
                model.save_weights(folder/'best.weights.h5')
            else:
                model, history, audit = fit_fixed_budget(chunks['train_files'], chunks['val_files'], sy,
                    initial, seed, mode, retain, folder/'best.weights.h5')
            pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
            audit.update(initial_weights_sha256=initial_hash, baseline=baseline, retain_state=retain,
                         seed=seed, protocol=mode, checkpoint_sha256=file_hash(folder/'best.weights.h5'))
            write_json(folder/'selection.json', audit)
            val, _ = stream_frame(model, chunks['val_files'], sy, pairs['val_files'], retain)
            validations[(label, seed)] = val
            save_report(val, folder, 'validation', False)
            del model, chunks
    rule = None
    if mode == 'memory20':
        rule, candidates = select_fallback([(validations['retain', s], validations['reset', s]) for s in seeds])
        candidates.to_csv(out/'fallback_validation_candidates.csv', index=False)
        write_json(out/'fallback_selection.json', dict(rule=rule, selection_split='validation',
            interpretation='output fusion of two separately evaluated networks; not dynamic state resetting'))
    del validations, pairs
    test_pairs = load_pairs(config, 'test_files', sort=mode == 'geometry40')
    for seed in seeds:
        test_frames = {}
        for label, baseline, retain in variants:
            tf.keras.backend.clear_session()
            folder = out/f'{label}_seed{seed}'
            folder.mkdir(exist_ok=True)
            model = build_final_azimuth(seed)
            checkpoint = folder/'best.weights.h5' if training_enabled else config['weights'][str(seed)]
            model.load_weights(checkpoint)
            chunks = make_chunks(test_pairs, sx, sy, baseline)
            frame, retained = stream_frame(model, chunks, sy, test_pairs, retain)
            test_frames[label] = frame
            score = save_report(frame, folder, 'test', config.get('figures', True))
            by_variant[label].append(dict(seed=seed, **score))
            write_json(folder/'support.json', dict(retained=retained,
                omitted=sorted(set(test_pairs)-set(retained)),
                evaluation_start_target_index=25, batch_trajectories=8, chunk_length=25))
            del model, chunks
        if rule is not None:
            save_report(apply_fallback(test_frames['retain'], test_frames['reset'], rule),
                        out, f'fallback_seed{seed}', config.get('figures', True))
    summaries = []
    for label, rows in by_variant.items():
        pd.DataFrame(rows).to_csv(out/f'{label}_seed_metrics.csv', index=False)
        summary = seed_summary(rows); summary['variant'] = label; summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(out/'seed_summary.csv', index=False)
    return dict(status='COMPLETED', mode=mode, seeds=seeds,
                benchmark='original supplied manifest; no visibility filtering')


def run_private(config):
    """Create a unique external run. Failures retain their diagnostic directory."""
    config = dict(config)
    mode = config['mode']
    if mode not in ('canonical', 'focused', 'memory20', 'geometry40'):
        raise ValueError(f'Unknown mode: {mode}')
    for key in ('data_root', 'manifest', 'output_root', 'model', 'scaler_X', 'scaler_y'):
        if config.get(key) is not None:
            config[key] = str(external_path(config[key]))
    if 'weights' in config:
        config['weights'] = {str(k): str(external_path(v)) for k, v in config['weights'].items()}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(config['output_root'])/f'{mode}_{stamp}_{uuid.uuid4().hex[:8]}'
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'configuration.json', config)
    provenance = dict(python=sys.version, platform=platform.platform(),
                       manifest_sha256=file_hash(config['manifest']))
    splits = read_manifest(config['manifest'])
    provenance['split_counts'] = {split: len(names) for split, names in splits.items()}
    provenance['package_versions'] = {name: version(name) for name in
        ('numpy', 'pandas', 'scikit-learn', 'joblib', 'tensorflow', 'keras')}
    provenance['public_source_sha256'] = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    needed = list(splits) if config.get('execute_training', False) else ['test_files']
    provenance['data_sha256'] = {}
    for split in needed:
        for name in splits[split]:
            for kind, filename in (('noise', name), ('clean', name.replace('_noise', '_clean'))):
                provenance['data_sha256'][f'{kind}/{filename}'] = file_hash(Path(config['data_root'])/kind/filename)
    for key in ('model', 'scaler_X', 'scaler_y'):
        if config.get(key):
            provenance[f'{key}_sha256'] = file_hash(config[key])
    if 'weights' in config:
        provenance['weights_sha256'] = {s: file_hash(p) for s, p in config['weights'].items()}
    write_json(out/'provenance.json', provenance)
    try:
        import tensorflow as tf
        provenance['devices'] = [str(device) for device in tf.config.list_physical_devices()]
        write_json(out/'provenance.json', provenance)
        result = run_canonical(config, out) if mode == 'canonical' else run_azimuth(config, out)
        write_json(out/'run_summary.json', result)
    except Exception as exc:
        write_json(out/'failure.json', dict(error=repr(exc), completed=False))
        raise
    print(f'Completed private run: {out}')
    return out
