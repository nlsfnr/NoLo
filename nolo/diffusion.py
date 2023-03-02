from functools import partial
from typing import Iterator, Optional, Protocol, Tuple, runtime_checkable

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, Numeric

from . import data, nn
from .common import expand_dims, plot


@runtime_checkable
class SamplingConfig(nn.ModelConfig, data.DataConfig, Protocol):
    pass


def lerp(start: Numeric, end: Numeric, t: Numeric) -> Array:
    return jnp.asarray(start + (end - start) * t)


def alpha_schedule(t: Array) -> Array:
    # return jnp.cos(t * jnp.pi / 2) ** 2
    return (1 - t) ** 2


def pertubation_kernel(
    x0: Array,  # B S D
    t: Array,  # B S
    noise: Array,  # B S D
) -> Array:
    alpha = expand_dims(alpha_schedule(t), x0)
    return x0 * alpha**0.5 + noise * (1 - alpha) ** 0.5


def sampling_step(
    xt: Array,  # B S D
    logits: Array,  # B S V
    embeddings: Array,  # V D
    t: Array,  # B S
    next_t: Array,  # B S
) -> Tuple[Array, Array]:
    alpha = expand_dims(alpha_schedule(t), xt)
    print(alpha)
    weights = jax.nn.softmax(logits, axis=-1)
    x0 = jnp.einsum("v d, b s v -> b s d", embeddings, weights)
    plot(xt)
    noise = (xt - alpha**0.5 * x0) / (1 - alpha) ** 0.5
    plot(noise)
    next_xt = pertubation_kernel(x0, next_t, noise)
    return x0, next_xt


def sample(
    params: ArrayTree,
    config: SamplingConfig,
    steps: int,
    seed: int,
    sequence_length: Optional[int] = None,
) -> Iterator[Tuple[Array, Array, str]]:
    def sample_step(
        xt: Array,
        t: Array,
        next_t: Array,
    ) -> Tuple[Array, Array, Array]:
        model = nn.Model.from_config(config)
        logits = model(xt, t[None, None], False)
        # TODO: Simplify sampling by eliminating score_interpolation.
        # scores = score_interpolation(logits, model.embeddings, xt, t)
        # x0, xt = sampling_step(xt, scores, t, next_t)
        x0, xt = sampling_step(xt, logits, model.embeddings, t, next_t)
        return xt, x0, logits

    rng = jax.random.PRNGKey(seed)
    sequence_length = sequence_length or config.sequence_length
    tokenizer = data.get_tokenizer(config)
    sample_step_hk = hk.without_apply_rng(hk.transform(sample_step))
    fn = partial(sample_step_hk.apply, params)
    ts = jnp.linspace(1, 0, steps + 1)
    xt = jax.random.normal(rng, (1, sequence_length, config.model_dim))
    for t, next_t in zip(ts[:-1], ts[1:]):
        xt, x0, logits = fn(xt, t, next_t)
        indices = logits.argmax(axis=-1)[0]
        text = tokenizer.decode(indices)
        yield xt, x0, text
