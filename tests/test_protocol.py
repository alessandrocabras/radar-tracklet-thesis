"""Toy-array contract tests. Not experimental benchmark data or a simulator."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from radar_tracklets.preprocessing import (
    build_elevation_features_and_target, build_range_features_and_target,
    build_azimuth_windows, build_azimuth_residual_series, causal_poly_forecast)
from radar_tracklets.workflows import window_arrays
from radar_tracklets.streaming import chunk_sequence, physical_predictions, stream_groups
from radar_tracklets.evaluation import pooled_metrics


def toy_frames(n=319):
    # Arbitrary arithmetic arrays, not physically generated trajectories.
    k = np.arange(n, dtype=float)
    frame = pd.DataFrame({"Elevation (rad)": .1 + k*.0001,
                          "Azimuth (rad)": .2 + k*.001,
                          "Range (m)": 1000 + 2*k + .01*k**2,
                          "Doppler (m/s)": 2 + .02*k})
    clean = frame.copy()
    frame["Elevation (rad)"] += .00002*np.sin(k)
    return frame, clean


@pytest.mark.parametrize("coordinate,features", [("elevation",8),("range",9),("azimuth",6)])
def test_window_alignment(coordinate, features):
    noisy, clean = toy_frames()
    data = window_arrays(coordinate, noisy, clean)
    assert data["X"].shape == (294, 25, features)
    assert data["y"].shape == (294, 1)
    np.testing.assert_array_equal(data["target_indices"], np.arange(25,319))
    column = {"elevation":"Elevation (rad)","range":"Range (m)","azimuth":"Azimuth (rad)"}[coordinate]
    np.testing.assert_allclose(data["clean"], clean[column].to_numpy()[25:])
    if coordinate == "range":
        np.testing.assert_allclose(data["y"].ravel()+data["baseline"], data["clean"])
    elif coordinate == "azimuth":
        np.testing.assert_allclose(data["y"].ravel()+data["pivot"], data["clean"])


@pytest.mark.parametrize("coordinate", ["elevation","range","azimuth"])
def test_clean_is_not_a_feature(coordinate):
    noisy, clean = toy_frames()
    altered = clean.copy()
    for column in altered:
        altered[column] += .1
    original = window_arrays(coordinate,noisy,clean)
    changed = window_arrays(coordinate,noisy,altered)
    np.testing.assert_array_equal(original["X"], changed["X"])
    assert not np.array_equal(original["y"], changed["y"])


@pytest.mark.parametrize("coordinate", ["elevation","range","azimuth"])
def test_future_measurements_do_not_change_first_forecast(coordinate):
    noisy, clean = toy_frames()
    altered = noisy.copy()
    altered.iloc[25:] += 10
    original = window_arrays(coordinate,noisy,clean)
    changed = window_arrays(coordinate,altered,clean)
    np.testing.assert_array_equal(original["X"][0], changed["X"][0])
    if coordinate == "range":
        np.testing.assert_allclose(original["baseline"][0], changed["baseline"][0])


def test_polynomial_zero_correction_and_residual_identity():
    noisy, clean = toy_frames()
    baseline = causal_poly_forecast(noisy["Range (m)"].to_numpy())
    np.testing.assert_allclose(baseline[24:-1],clean["Range (m)"].to_numpy()[25:],atol=1e-8)
    data = window_arrays("range",noisy,clean)
    residual_prediction = np.ones(294)
    np.testing.assert_allclose(residual_prediction-data["y"].ravel(),
                               data["baseline"]+residual_prediction-data["clean"],atol=1e-10)


def test_stream_clean_independence_and_future_causality():
    noisy, clean = toy_frames()
    series = build_azimuth_residual_series(noisy,clean)
    other_clean = clean + .01
    np.testing.assert_array_equal(series["features"],build_azimuth_residual_series(noisy,other_clean)["features"])
    altered = noisy.copy(); altered.iloc[41:] += 10
    changed = build_azimuth_residual_series(altered,clean)
    np.testing.assert_array_equal(series["features"][:41], changed["features"][:41])
    np.testing.assert_array_equal(series["baseline"][:41], changed["baseline"][:41])


def test_padding_and_evaluation_support():
    noisy, clean = toy_frames()
    series = build_azimuth_residual_series(noisy,clean)
    sx = StandardScaler().fit(series["features"])
    sy = StandardScaler().fit(series["target_residual"].reshape(-1,1))
    chunked = chunk_sequence(series,sx,sy)
    assert chunked["X"].shape == (13,25,9)
    assert chunked["weights"].sum() == 318
    assert np.count_nonzero(chunked["weights"] == 0) == 7
    prediction, error, mask = physical_predictions(chunked["y"],chunked,sy)
    assert mask.sum() == 294
    np.testing.assert_array_equal(chunked["target_indices"].ravel()[mask],np.arange(25,319))
    assert np.max(np.abs(error)) < 1e-3


def test_fixed_batch_coverage_and_reset_events():
    noisy, clean = toy_frames()
    series = build_azimuth_residual_series(noisy,clean)
    sx = StandardScaler().fit(series["features"])
    sy = StandardScaler().fit(series["target_residual"].reshape(-1,1))
    data = {f"toy_{i:02}":chunk_sequence(series,sx,sy) for i in range(20)}
    events = list(stream_groups(data,8))
    assert sum(kind=="reset" for kind,_,_ in events)==2
    assert sum(kind=="batch" for kind,_,_ in events)==26
    retained = {name for kind,group,_ in events if kind=="reset" for name in group}
    assert len(retained)==16


def test_only_azimuth_uses_circular_errors():
    result = pooled_metrics([2*np.pi-.001],[.001],"azimuth")
    assert result["rmse"] == pytest.approx(2)
    assert pooled_metrics([2*np.pi-.001],[.001],"elevation")["rmse"] > 6000


def test_metrics_pool_samples_not_trajectory_rmses():
    result = pooled_metrics([0,0,0,4],[0,0,0,0],"range")
    assert result["n"]==4
    assert result["rmse"]==2


@pytest.mark.parametrize("coordinate", ["elevation","range","azimuth"])
def test_scalers_training_only_and_preserved_for_checkpoints(tmp_path, coordinate):
    import json
    from radar_tracklets.workflows import prepare_windows
    (tmp_path/"noise").mkdir(); (tmp_path/"clean").mkdir()
    manifest = {"train_files":["toy_train_noise.csv"],
                "val_files":["toy_val_noise.csv"], "test_files":["toy_test_noise.csv"]}
    for split, offset in (("train_files",0),("val_files",100),("test_files",200)):
        noisy, clean=toy_frames(30)
        noisy=noisy+offset; clean=clean+offset
        name=manifest[split][0]
        noisy.to_csv(tmp_path/"noise"/name,index=False)
        clean.to_csv(tmp_path/"clean"/name.replace("_noise","_clean"),index=False)
    path=tmp_path/"private_split.json"; path.write_text(json.dumps(manifest))
    _, sx, sy=prepare_windows(tmp_path,path,coordinate)
    noisy, clean=toy_frames(30)
    if coordinate=="elevation":
        X,y,_=build_elevation_features_and_target(noisy,clean)
    elif coordinate=="range":
        X,y,_,_,_=build_range_features_and_target(noisy,clean)
    else:
        data=window_arrays(coordinate,noisy,clean); X=data["X"].reshape(-1,6); y=data["y"]
    np.testing.assert_allclose(sx.mean_,StandardScaler().fit(X).mean_)
    np.testing.assert_allclose(sy.mean_,StandardScaler().fit(np.asarray(y).reshape(-1,1)).mean_)
    before=sx.mean_.copy()
    sx.mean_+=1
    _, reused, _=prepare_windows(tmp_path,path,coordinate,fitted_scalers=(sx,sy))
    assert reused is sx
    np.testing.assert_array_equal(reused.mean_,before+1)

