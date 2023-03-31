import haiku as hk
import jax

from .nn import Model


def test_model_call() -> None:
    @hk.testing.transform_and_run
    def fn() -> None:
        model = Model(
            num_layers=2,
            vocabulary_size=128,
            model_dim=8,
            num_heads=2,
            pos_emb_portion=0.5,
            t_emb_dim=8,
            hidden_dim=32,
            dropout=0.0,
        )
        rng = jax.random.PRNGKey(0)
        indices = jax.random.randint(rng, (4, 16), 0, 128)
        xs = model.embed(indices)
        assert xs.shape == (4, 16, 8)
        ts = jax.random.uniform(rng, (4,))
        ys = model(xs, ts, is_training=False)
        assert ys.shape == (4, 16, 128)

    fn()  # type: ignore
