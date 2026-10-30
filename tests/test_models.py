"""Optional TensorFlow tests; skipped when TensorFlow is not installed."""
import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")
from radar_tracklets.models import build_window_model, build_final_azimuth
from radar_tracklets.streaming import reset_states


@pytest.mark.parametrize("coordinate,features",[("elevation",8),("range",9),("azimuth",6)])
@pytest.mark.parametrize("architecture",["MLP","CNN","GRU","GRU-CNN"])
def test_model_shapes_and_readout(coordinate,features,architecture):
    model = build_window_model(coordinate,architecture)
    assert model.input_shape == (None,25,features)
    assert model.output_shape == (None,1)
    gru = [layer for layer in model.layers if isinstance(layer,tf.keras.layers.GRU)]
    if gru:
        assert gru[0].return_sequences == (architecture=="GRU-CNN")
        assert not gru[0].stateful
    for layer in model.layers:
        if isinstance(layer,tf.keras.layers.Conv1D):
            assert layer.padding=="same" and layer.kernel_size==(5,)


def test_state_reset_and_chunk_invariance():
    model=build_final_azimuth()
    values=np.random.default_rng(42).normal(size=(8,50,9)).astype("float32")
    # Clone with flexible time dimension for the same-weight inference check.
    dynamic=tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(8,None,9)),
        tf.keras.layers.GRU(256,stateful=True,return_sequences=True),
        tf.keras.layers.Dense(64,activation="relu"),tf.keras.layers.Dense(1)])
    dynamic.set_weights(model.get_weights())
    def run(length):
        reset_states(dynamic)
        return np.concatenate([dynamic(values[:,i:i+length],training=False).numpy()
                               for i in range(0,50,length)],axis=1)
    np.testing.assert_allclose(run(25),run(7),atol=1e-5,rtol=1e-5)
    np.testing.assert_allclose(run(25),run(25),atol=1e-6)


def test_canonical_default_and_parameter_counts():
    elevation = build_window_model('elevation')
    range_model = build_window_model('range')
    assert any(isinstance(layer, tf.keras.layers.GRU) for layer in elevation.layers)
    assert not any(isinstance(layer, tf.keras.layers.Conv1D) for layer in elevation.layers)
    assert range_model.count_params()-elevation.count_params() == 768
    assert build_window_model('elevation', 'GRU-CNN').count_params() == 268033
    assert build_window_model('range', 'GRU-CNN').count_params() == 322209


def test_fixed_budget_selection_and_reset_policy(tmp_path):
    from sklearn.preprocessing import StandardScaler
    from radar_tracklets.streaming import chunk_sequence, predict_stateful_stream
    from radar_tracklets.training import fit_fixed_budget, SCHEDULES
    from radar_tracklets.experiments import weights_hash
    rng = np.random.default_rng(9)
    features = rng.normal(size=(29, 9)).astype('float32')
    target = rng.normal(scale=.001, size=29).astype('float32')
    series = dict(features=features, target_residual=target, baseline=np.zeros(29),
                  clean=target.astype(float), radar=np.zeros(29), target_indices=np.arange(1, 30))
    sx = StandardScaler().fit(features); sy = StandardScaler().fit(target[:, None])
    chunks = {f'toy_{i}': chunk_sequence(series, sx, sy) for i in range(8)}
    def factory(seed):
        return build_final_azimuth(seed, units=4, dense_units=3)
    initial = factory(42).get_weights()
    fingerprint = weights_hash(initial)
    model, history, audit = fit_fixed_budget(chunks, chunks, sy, initial, 42, 'memory20',
        retain_state=False, checkpoint_path=tmp_path/'best.weights.h5', model_factory=factory)
    assert len(history) == audit['epochs_completed'] == 20
    assert audit['best_validation_rmse_mrad'] == min(row['val_rmse_mrad'] for row in history)
    assert audit['best_epoch'] == 1+np.argmin([row['val_rmse_mrad'] for row in history])
    assert [row['learning_rate'] for row in history[::5]] == [rate for _, rate in SCHEDULES['memory20']]
    assert weights_hash(initial) == fingerprint
    pred, metadata, names = predict_stateful_stream(model, chunks, retain_state=False)
    assert pred.shape == (16, 25, 1)
    assert set(metadata['trajectory_id'].ravel()) == set(names)
    model.load_weights(tmp_path/'best.weights.h5')
    again, _, _ = predict_stateful_stream(model, chunks, retain_state=False)
    np.testing.assert_allclose(pred, again, atol=1e-6)


@pytest.mark.parametrize('mode', ['canonical', 'focused'])
def test_frozen_private_entry_point(tmp_path, mode):
    import json
    import joblib
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from radar_tracklets.experiments import run_private
    from radar_tracklets.preprocessing import build_azimuth_residual_series
    from radar_tracklets.workflows import window_arrays
    k = np.arange(30, dtype=float)
    data = pd.DataFrame({'Azimuth (rad)': .2+.001*k, 'Elevation (rad)': .1+.0001*k,
                         'Range (m)': 1000+3*k, 'Doppler (m/s)': np.full(30, 3.), 'Time (s)': k})
    for kind in ('noise', 'clean'):
        (tmp_path/kind).mkdir()
        for i in range(8):
            data.to_csv(tmp_path/kind/f'toy_{i}_{kind}.csv', index=False)
    manifest = dict(split_files=dict(train=['unused_train_noise.csv'], val=['unused_val_noise.csv'],
                                    test=[f'toy_{i}_noise.csv' for i in range(8)]))
    (tmp_path/'split.json').write_text(json.dumps(manifest))
    if mode == 'canonical':
        windows = window_arrays('elevation', data, data)
        sx = StandardScaler().fit(windows['X'].reshape(-1, 8))
        sy = StandardScaler().fit(windows['y'])
        model = build_window_model('elevation')
        model.save(tmp_path/'model.keras')
    else:
        series = build_azimuth_residual_series(data, data)
        sx = StandardScaler().fit(series['features'])
        sy = StandardScaler().fit(series['target_residual'][:, None])
        model = build_final_azimuth(42)
        model.save_weights(tmp_path/'model.weights.h5')
    joblib.dump(sx, tmp_path/'sx.pkl'); joblib.dump(sy, tmp_path/'sy.pkl')
    config = dict(mode=mode, coordinate='elevation', data_root=str(tmp_path),
                  manifest=str(tmp_path/'split.json'), output_root=str(tmp_path/'results'),
                  scaler_X=str(tmp_path/'sx.pkl'), scaler_y=str(tmp_path/'sy.pkl'),
                  execute_training=False, figures=False, seeds=[42])
    if mode == 'canonical':
        config['model'] = str(tmp_path/'model.keras')
    else:
        config['weights'] = {'42': str(tmp_path/'model.weights.h5')}
    out = run_private(config)
    assert (out/'run_summary.json').exists()
    exports = list(out.rglob('*_errors.csv'))
    assert len(exports) == 1
    predictions = pd.read_csv(exports[0])
    assert len(predictions) == 40
    assert predictions.target_index.min() == 25
    assert predictions.target_index.max() == 29
    provenance = json.loads((out/'provenance.json').read_text())
    assert provenance['split_counts']['test_files'] == 8
    assert len(provenance['data_sha256']) == 16
    assert provenance['package_versions']['tensorflow'] == '2.18.0'

