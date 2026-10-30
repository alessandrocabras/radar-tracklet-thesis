"""Small, explicit adapters for user-supplied private data.

These adapters preserve supplied trajectory splits and training-only scaling.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .preprocessing import (build_elevation_features_and_target,
                            build_range_features_and_target, build_azimuth_windows,
                            build_azimuth_residual_series)
from .streaming import chunk_sequence


def read_manifest(path):
    """Require explicit trajectory-held-out lists. Never infer a new split."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = ("train_files", "val_files", "test_files")
    if "split_files" in manifest:
        splits = dict(zip(keys, (manifest["split_files"][key] for key in ("train", "val", "test"))))
        if any(key in manifest and manifest[key] != splits[key] for key in keys):
            raise ValueError("Conflicting manifest split representations")
    else:
        splits = {key: manifest[key] for key in keys}
    for key, names in splits.items():
        if not isinstance(names, list) or not names:
            raise ValueError(f"Missing/non-list/empty {key}")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate names in {key}")
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Manifest entries must be filenames, not paths")
    groups = list(splits.values())
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError("Trajectory names overlap across splits")
    return splits


def read_pair(root, name):
    root = Path(root)
    noisy = pd.read_csv(root / "noise" / name)
    clean = pd.read_csv(root / "clean" / name.replace("_noise", "_clean"))
    required = ["Elevation (rad)", "Azimuth (rad)", "Range (m)", "Doppler (m/s)"]
    if len(noisy) != len(clean) or len(noisy) < 26:
        raise ValueError(f"Unequal or insufficient sample counts: {name}")
    for frame in (noisy, clean):
        if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite observables: {name}")
    if "Time (s)" in noisy and "Time (s)" in clean:
        times = noisy["Time (s)"].to_numpy(float)
        if not (np.isfinite(times).all() and np.all(np.diff(times) > 0)):
            raise ValueError(f"Invalid timestamps: {name}")
        reference = clean["Time (s)"].to_numpy(float)
        if not np.allclose(times, reference, atol=1e-9, rtol=0):
            raise ValueError(f"Noisy and clean timestamps differ: {name}")
    elif ("Time (s)" in noisy) != ("Time (s)" in clean):
        raise ValueError(f"Timestamp column is present in only one member: {name}")
    return noisy, clean


def window_arrays(coordinate, noisy, clean):
    """Return raw X/y plus trajectory-local physical evaluation metadata."""
    if coordinate == "azimuth":
        X, y, pivot, indices, target, _ = build_azimuth_windows(noisy, clean, 25, 1)
        return {"X": X, "y": y, "target_indices": indices,
                "clean": target, "pivot": pivot}
    if coordinate == "elevation":
        features, target, _ = build_elevation_features_and_target(noisy, clean, horizon=1)
        baseline = None
    elif coordinate == "range":
        features, target, baseline, _, _ = build_range_features_and_target(noisy, clean, horizon=1)
    else:
        raise ValueError(coordinate)
    starts = np.arange(len(features) - 25 + 1)
    ends = starts + 24
    data = {"X": np.stack([features[i:i+25] for i in starts]),
            "y": target[ends, None], "target_indices": ends + 1,
            "clean": clean["Elevation (rad)" if coordinate == "elevation" else "Range (m)"].to_numpy()[ends+1]}
    if baseline is not None:
        data["baseline"] = baseline[ends]
    return data


def prepare_windows(root, manifest_path, coordinate, fitted_scalers=None):
    """Load private data; fit scalers on training only, with notebook weighting.

    Elevation/range scalers fit raw trajectory rows before overlapping windows;
    canonical azimuth fits window-relative rows including their multiplicity.
    """
    splits = read_manifest(manifest_path)
    records = {}
    scaler_X, scaler_y = fitted_scalers if fitted_scalers is not None else (StandardScaler(), StandardScaler())
    for split, names in splits.items():
        records[split] = []
        for name in names:
            noisy, clean = read_pair(root, name)
            data = window_arrays(coordinate, noisy, clean)
            if split == "train_files" and fitted_scalers is None:
                if coordinate == "elevation":
                    raw_X, raw_y, _ = build_elevation_features_and_target(noisy, clean, horizon=1)
                elif coordinate == "range":
                    raw_X, raw_y, _, _, _ = build_range_features_and_target(noisy, clean, horizon=1)
                else:
                    raw_X, raw_y = data["X"].reshape(-1, 6), data["y"]
                scaler_X.partial_fit(raw_X)
                scaler_y.partial_fit(np.asarray(raw_y).reshape(-1, 1))
            data["trajectory_id"] = np.repeat(name, len(data["y"]))
            records[split].append(data)
    transformed = {}
    for split, items in records.items():
        data = {key: np.concatenate([item[key] for item in items]) for key in items[0]}
        shape = data["X"].shape
        data["X"] = scaler_X.transform(data["X"].reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
        data["y"] = scaler_y.transform(data["y"]).astype(np.float32)
        transformed[split] = data
    return transformed, scaler_X, scaler_y


def prepare_private_stream(root, manifest_path, scaler_X, scaler_y, split="test_files"):
    """Transform with existing private scalers; never refit during evaluation."""
    names = read_manifest(manifest_path)[split]
    result = {}
    for name in names:
        noisy, clean = read_pair(root, name)
        series = build_azimuth_residual_series(noisy, clean)
        result[name] = chunk_sequence(series, scaler_X, scaler_y)
    return result

