"""Toy-array checks for follow-up protocols and private export interfaces."""
import json
import numpy as np
import pandas as pd
import pytest
from radar_tracklets.geometry import cartesian_forecast, residual_series, baseline_names, joint_errors
from radar_tracklets.reporting import (prediction_frame, common_support, export_predictions,
    fallback_rules, fallback_mask, select_fallback, apply_fallback, seed_summary)
from radar_tracklets.workflows import read_manifest, read_pair
from radar_tracklets.experiments import external_path, REPOSITORY, select_baseline
from radar_tracklets.analysis import spatial_metrics


def frame(prediction, clean=None, coordinate='azimuth', indices=None):
    prediction = np.asarray(prediction, float)
    clean = np.zeros_like(prediction) if clean is None else clean
    return prediction_frame(coordinate, prediction, clean, np.repeat('toy', len(prediction)),
                            np.arange(25, 25+len(prediction)) if indices is None else indices)


def toy_pair(n=60):
    k = np.arange(n, dtype=float)
    data = pd.DataFrame({'Azimuth (rad)': .2+.001*k, 'Elevation (rad)': .1+.0001*k,
                         'Range (m)': 1000+3*k, 'Doppler (m/s)': np.full(n, 3.), 'Time (s)': k})
    return data, data.copy()


def test_original_manifest_and_conflicting_schema(tmp_path):
    path = tmp_path/'split.json'
    source = dict(split_files=dict(train=['a_noise.csv'], val=['b_noise.csv'], test=['c_noise.csv']))
    path.write_text(json.dumps(source))
    assert read_manifest(path)['val_files'] == ['b_noise.csv']
    source['train_files'] = ['different_noise.csv']
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='Conflicting'):
        read_manifest(path)


def test_pair_rejects_timestamp_misalignment(tmp_path):
    a, b = toy_pair()
    b['Time (s)'] += .01
    for kind, data in [('noise', a), ('clean', b)]:
        (tmp_path/kind).mkdir()
        data.to_csv(tmp_path/kind/f'toy_{kind}.csv', index=False)
    with pytest.raises(ValueError, match='timestamps differ'):
        read_pair(tmp_path, 'toy_noise.csv')


@pytest.mark.parametrize('name', baseline_names())
def test_baseline_and_features_are_causal_and_clean_independent(name):
    noisy, clean = toy_pair()
    original = residual_series(noisy, clean, name)
    altered = noisy.copy(); altered.iloc[25:] += 1
    changed = residual_series(altered, clean, name)
    np.testing.assert_array_equal(original['features'][:25], changed['features'][:25])
    np.testing.assert_allclose(original['baseline'][:25], changed['baseline'][:25], atol=1e-12)
    changed = residual_series(noisy, clean+1, name)
    np.testing.assert_array_equal(original['features'], changed['features'])
    np.testing.assert_array_equal(original['baseline'], changed['baseline'])


def test_cartesian_linear_extrapolation_and_validation_selection():
    noisy, clean = toy_pair()
    noisy['Azimuth (rad)'] = 0.; noisy['Elevation (rad)'] = 0.
    result = cartesian_forecast(noisy)
    np.testing.assert_allclose(result[24:-1, 0], noisy['Range (m)'].to_numpy()[25:], atol=1e-10)
    decision, candidates = select_baseline({f'toy_{i}': toy_pair() for i in range(8)})
    assert len(candidates) == 10
    assert decision['baseline'].startswith('cartesian_degree')
    assert decision['selection_split'] == 'validation'


def test_common_support_keeps_exact_keys_and_references():
    a, b = common_support(frame([.1, .2, .3]), frame([.4, .5], indices=[26, 27]))
    assert a.target_index.tolist() == b.target_index.tolist() == [26, 27]
    bad = frame([.4, .5], clean=[1, 1], indices=[26, 27])
    with pytest.raises(ValueError, match='references'):
        common_support(a, bad)


def test_metrics_exports_and_no_overwrite(tmp_path):
    data = frame([.001, -.002, .003])
    score = export_predictions(data, tmp_path, 'toy')
    assert score['rmse'] == pytest.approx(np.sqrt(14/3))
    assert score['p99'] == pytest.approx(2.98)
    assert score['maximum'] == pytest.approx(3)
    assert list(pd.read_csv(tmp_path/'toy_errors.csv').target_index) == [25, 26, 27]
    with pytest.raises(FileExistsError):
        export_predictions(data, tmp_path, 'toy')
    with pytest.raises(ValueError, match='prefix'):
        export_predictions(data, tmp_path, '../unsafe')


def test_fallback_is_output_selection_from_causal_gate():
    a, b = frame([.001, .01]), frame([.005, .002])
    for part in (a, b):
        part['input_elevation_deg'] = [20, 70]
        part['input_increment_abs_mrad'] = [1, 1]
    assert len(fallback_rules()) == 26
    np.testing.assert_array_equal(fallback_mask(a, 'elevation_60'), [False, True])
    result = apply_fallback(a, b, 'elevation_60')
    np.testing.assert_allclose(result.prediction, [.001, .002])
    rule, candidates = select_fallback([(a, b), (a.copy(), b.copy())])
    assert len(candidates) == 26
    assert rule == 'elevation_60'


def test_seed_statistics_are_mean_and_sample_sd():
    rows = [dict(rmse=x, mae=x, bias=x, p95=x, p99=x, maximum=x) for x in [1, 2, 3]]
    summary = seed_summary(rows)
    assert summary.iloc[0]['mean'] == 2
    assert summary.iloc[0].sample_sd == 1
    assert summary.iloc[0].seeds == 3


def test_joint_errors_require_predicted_coordinates(tmp_path):
    direction, position = joint_errors(np.array([0.]), np.array([.1]), np.array([1000.]),
                                        np.array([0.]), np.array([0.]), np.array([1000.]))
    assert direction[0] == pytest.approx(100)
    assert position[0] == pytest.approx(2000*np.sin(.05))
    a, e, r = frame([0.]), frame([.1], coordinate='elevation'), frame([1000.], [1000.], 'range')
    table = spatial_metrics(a, e, r, tmp_path/'joint')
    assert table.query("condition == 'all' and quantity == 'direction'").iloc[0].rmse == pytest.approx(100)


def test_private_path_boundary():
    with pytest.raises(ValueError, match='outside'):
        external_path(REPOSITORY/'outputs')


def test_variable_length_groups_are_not_silently_truncated():
    from sklearn.preprocessing import StandardScaler
    from radar_tracklets.streaming import chunk_sequence, stream_groups
    series = [residual_series(*toy_pair(n)) for n in [60]*7+[61]]
    sx = StandardScaler().fit(series[0]['features'])
    sy = StandardScaler().fit(series[0]['target_residual'][:, None])
    chunks = {f'toy_{i}': chunk_sequence(s, sx, sy) for i, s in enumerate(series)}
    with pytest.raises(ValueError, match='equal trajectory lengths'):
        list(stream_groups(chunks, 8))


def test_ukf_loader_declares_units_and_checks_reference(tmp_path):
    pytest.importorskip('matplotlib')
    from radar_tracklets.analysis import ukf_posterior_comparison
    neural = frame([.101, .102], clean=[.1, .1], coordinate='elevation', indices=[50, 51])
    neural['time_s'] = [50., 51.]
    path = tmp_path/'posterior.csv'
    pd.DataFrame({'time[s]': [50., 51.], 'filtered [el, deg]': [.1001, .1002],
                  'measured [el, deg]': [.11, .09], 'real [el, deg]': [.1, .1]}).to_csv(path, index=False)
    kwargs = dict(filtered_column='filtered [el, deg]', measured_column='measured [el, deg]',
                  real_column='real [el, deg]', filtered_unit='rad', measured_unit='rad', real_unit='rad')
    result = ukf_posterior_comparison(neural, path, tmp_path/'comparison', **kwargs)
    assert result.iloc[0].n == result.iloc[1].n == 2
    assert result.iloc[1].rmse == pytest.approx(np.sqrt(.1**2+.2**2)/np.sqrt(2))
    settings = json.loads((tmp_path/'comparison/loader.json').read_text())
    assert not settings['information_sets_matched']
    kwargs['real_unit'] = 'deg'
    with pytest.raises(ValueError, match='references differ'):
        ukf_posterior_comparison(neural, path, tmp_path/'wrong_units', **kwargs)


def test_plot_has_no_title_and_uses_actual_samples(tmp_path, monkeypatch):
    plt = pytest.importorskip('matplotlib.pyplot')
    from radar_tracklets.reporting import plot_errors
    captured = []
    original = plt.subplots
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(plt, 'subplots', capture)
    data = frame([.001, -.002, .003])
    plot_errors(data, tmp_path/'figure.pdf', 'toy')
    figure, axes = captured[0]
    assert all(axis.get_title() == '' for axis in axes)
    assert figure._suptitle is None
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), data.error)
    np.testing.assert_allclose(axes[1].lines[0].get_xdata(), [1, 2, 3])
