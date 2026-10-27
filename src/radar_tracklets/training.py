"""Explicit new-run final-azimuth fitting; no historical artifacts overwritten."""
import numpy as np
from .models import build_final_azimuth
from .streaming import stateful_epoch, predict_stateful_stream, physical_predictions


def fit_final_azimuth(train_chunks, validation_chunks, target_scaler, seed=42,
                      epochs=60, patience=10, lr_patience=4, min_delta_mrad=.001):
    """Focused Huber-d2 protocol adapted from train_focused_variant.

    Selection uses pooled physical validation RMSE, not normalized Huber
    loss or test metrics. Test data are deliberately not accepted. Both
    scalers must already have been fitted on training data only. Returns
    a best-weight model and in-memory history; writes no files.
    """
    model = build_final_azimuth(seed)
    best_weights, best_epoch = None, None
    best_rmse, wait, lr_wait = np.inf, 0, 0
    history = []
    for epoch in range(epochs):
        train_loss = stateful_epoch(model, train_chunks, True, seed+epoch)
        pred, metadata, retained = predict_stateful_stream(model, validation_chunks)
        _, error, valid = physical_predictions(pred, metadata, target_scaler)
        rmse = float(np.sqrt(np.mean(error**2)))
        if not np.isfinite(rmse):
            raise ValueError("No finite validation score; check stream coverage")
        learning_rate = float(model.optimizer.learning_rate.numpy())
        history.append({"epoch":epoch+1, "train_loss_normalized":train_loss,
                        "val_rmse_mrad":rmse, "learning_rate":learning_rate,
                        "validation_trajectories":len(retained),
                        "validation_targets":int(valid.sum())})
        if rmse < best_rmse-min_delta_mrad:
            best_rmse, best_epoch = rmse, epoch+1
            best_weights = model.get_weights()
            wait, lr_wait = 0, 0
        else:
            wait += 1
            lr_wait += 1
            if lr_wait >= lr_patience:
                model.optimizer.learning_rate.assign(max(1e-6,learning_rate*.5))
                lr_wait = 0
            if wait >= patience:
                break
    if best_weights is None:
        raise ValueError("No validation checkpoint was selected")
    model.set_weights(best_weights)
    return model, history, {"best_epoch":best_epoch,
                            "best_validation_rmse_mrad":best_rmse,
                            "selection_metric":"pooled physical validation RMSE (mrad)"}

