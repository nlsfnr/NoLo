import haiku as hk
import jax.numpy as jnp

from . import nn


def test_model() -> None:
    @hk.testing.transform_and_run
    def fn() -> None:
        model = nn.Model(
            vocab_size=10,
            n_layers=2,
            model_dim=32,
            n_heads=4,
            t_mlp_layers=[8, 32, 32],
            mlp_size=64,
            dropout=0.1,
            name="model",
        )
        indices = jnp.zeros((2, 3), dtype=jnp.int32)
        x = model.embed(indices)
        assert x.shape == (2, 3, 32)
        t = jnp.ones((2, 3))
        out = model(x, t, is_training=True)
        assert out.shape == (2, 3, 10)

    fn()  # type: ignore
