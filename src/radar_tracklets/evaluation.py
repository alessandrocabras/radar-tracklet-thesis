"""Physical-unit reconstruction and explicit pooled metrics."""
import numpy as np
from .preprocessing import wrap_to_pi


def reconstruct(coordinate, predicted_scaled, target_scaler, baseline=None, pivot=None):
    values = target_scaler.inverse_transform(
        np.asarray(predicted_scaled).reshape(-1, 1)).ravel()
    if coordinate == "range":
        if baseline is None:
            raise ValueError("A causal range baseline is required")
        values = values + np.asarray(baseline).ravel()
    elif coordinate == "azimuth":
        if pivot is None:
            raise ValueError("Measured window pivots are required")
        values = values + np.asarray(pivot).ravel()
    elif coordinate != "elevation":
        raise ValueError(coordinate)
    return values


def pooled_metrics(prediction, clean, coordinate):
    prediction, clean = np.asarray(prediction).ravel(), np.asarray(clean).ravel()
    if prediction.shape != clean.shape or not len(clean):
        raise ValueError("Expected equal, nonempty evaluation supports")
    if not (np.isfinite(prediction).all() and np.isfinite(clean).all()):
        raise ValueError("Select a finite common support explicitly before evaluation")
    error = prediction - clean
    if coordinate == "azimuth":
        error = wrap_to_pi(error)
    if coordinate not in {"range", "elevation", "azimuth"}:
        raise ValueError(coordinate)
    unit = "m" if coordinate == "range" else "mrad"
    error = error * (1 if coordinate == "range" else 1000)
    absolute = np.abs(error)
    return {"n": int(len(error)), "unit": unit,
            "aggregation": "pooled target errors, not mean trajectory RMSE",
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(absolute.mean()), "bias": float(error.mean()),
            "p95": float(np.quantile(absolute, .95)),
            "p99": float(np.quantile(absolute, .99)),
            "maximum": float(absolute.max())}

