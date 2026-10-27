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

