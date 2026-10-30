"""Explicit model families; no claim of reproducing historical tuned weights."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FEATURE_DIMENSIONS = {"elevation": 8, "range": 9, "azimuth": 6}


@keras.utils.register_keras_serializable(package="RadarTracklets")
class LastTimeStep(layers.Layer):
    def call(self, inputs):
        return inputs[:, -1, :]

    def get_config(self):
        return super().get_config()


def build_window_model(coordinate, architecture="GRU", seed=42):
    """Canonical stateless GRU; other families require explicit selection."""
    tf.keras.utils.set_random_seed(seed)
    inputs = layers.Input(shape=(25, FEATURE_DIMENSIONS[coordinate]))
    if architecture == "MLP":
        x = layers.Flatten()(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
    elif architecture == "CNN":
        x = layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
        x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
    elif architecture == "GRU":
        x = layers.GRU(256, stateful=False, return_sequences=False)(inputs)
        x = layers.Dense(64, activation="relu")(x)
    elif architecture == "GRU-CNN":
        x = layers.GRU(256, stateful=False, return_sequences=True)(inputs)
        widths = (64, 64, 32) if coordinate == "range" else (32, 64, 32)
        for width in widths:
            x = layers.Conv1D(width, 5, padding="same")(x)
            x = layers.LeakyReLU(negative_slope=0.1)(x)
        x = LastTimeStep()(x) if coordinate == "range" else layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128 if coordinate == "range" else 64)(x)
        x = layers.LeakyReLU(negative_slope=0.1)(x)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    model = keras.Model(inputs, layers.Dense(1)(x))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_final_azimuth(seed=42, units=256, dense_units=64):
    """Focused stateful_dense_huber_d2: fixed eight-trajectory batch."""
    tf.keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        layers.Input(batch_shape=(8, 25, 9)),
        layers.GRU(units, stateful=True, return_sequences=True),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.Huber(delta=2.0))
    return model

