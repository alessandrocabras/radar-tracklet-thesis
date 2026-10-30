"""Explicit new-run final-azimuth fitting; no historical artifacts overwritten."""
import numpy as np
from .models import build_final_azimuth
from .streaming import stateful_epoch, predict_stateful_stream, physical_predictions

SCHEDULES = {
    'memory20': [(0, .001), (5, .0005), (10, .00025), (15, .000125)],
    'geometry40': [(0, .001), (10, .0005), (20, .00025), (30, .000125)],
}


def fit_fixed_budget(train_chunks, validation_chunks, target_scaler,
                     initial_weights, seed, protocol, retain_state=True,
                     checkpoint_path=None, model_factory=build_final_azimuth):
    """Full fixed budget, exact minimum physical validation RMSE; no test input."""
    schedule = SCHEDULES[protocol]
    epochs = 20 if protocol == 'memory20' else 40
    model = model_factory(seed)
    model.set_weights(initial_weights)
    best_rmse, best_weights, best_epoch = np.inf, None, None
    history = []
    for epoch in range(epochs):
        rate = next(rate for start, rate in reversed(schedule) if epoch >= start)
        model.optimizer.learning_rate.assign(rate)
        loss = stateful_epoch(model, train_chunks, True, seed+epoch, retain_state)
        pred, data, retained = predict_stateful_stream(model, validation_chunks, retain_state)
        _, error, valid = physical_predictions(pred, data, target_scaler)
        score = float(np.sqrt(np.mean(error**2)))
        if not np.isfinite(score):
            raise ValueError('No finite physical validation RMSE')
        history.append(dict(epoch=epoch+1, train_loss_normalized=loss,
                            val_rmse_mrad=score, learning_rate=rate,
                            validation_trajectories=len(retained), validation_targets=int(valid.sum())))
        if score < best_rmse:
            best_rmse, best_epoch, best_weights = score, epoch+1, model.get_weights()
            if checkpoint_path is not None:
                model.save_weights(checkpoint_path)
        print(f'{protocol} seed={seed} retain={retain_state} epoch={epoch+1}/{epochs} '
              f'validation_RMSE={score:.6f} mrad', flush=True)
    model.set_weights(best_weights)
    return model, history, dict(best_epoch=best_epoch, best_validation_rmse_mrad=best_rmse,
                                epochs_completed=epochs, early_stopping=False,
                                learning_rate_schedule=schedule,
                                selection_metric='pooled physical validation RMSE (mrad)')


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

