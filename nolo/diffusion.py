from functools import partial
from typing import Iterator, Optional, Protocol, Tuple, runtime_checkable

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, PRNGKey, Numeric
from einops import rearrange

from . import data, nn


@runtime_checkable
class SamplingConfig(nn.ModelConfig, data.DataConfig, Protocol):
    pass


def lerp(start: Numeric, end: Numeric, t: Numeric) -> Array:
    return jnp.asarray(start + (end - start) * t)


def noise_scale(x: Array) -> Array:
    return lerp(0.0001, 1., x)


def expand_dims(x: Array, reference: Array) -> Array:
    while x.ndim < reference.ndim:
        x = x[..., None]
    return x


def pertubation_kernel(
    x0: Array,  # B S D
    t: Array,  # B S
    noise: Array,  # B S D
) -> Array:
    # Eq. 29 in https://arxiv.org/abs/2011.13456, we assume that σ(0) is negligible.
    return x0 + noise * expand_dims(t, x0)  # B S D


def sampling_step(
    xt: Array,  # B S D
    scores: Array,  # B S D
    t: Array,  # B S
    next_t: Array,  # B S
) -> Array:
    # Eq.43 in https://arxiv.org/abs/2011.13456
    scale = expand_dims(noise_scale(t), xt)
    next_scale = expand_dims(noise_scale(next_t), xt)
    return xt + 0.5 * (scale ** 2 - next_scale ** 2) * scores  # B S D


def score_interpolation(
    logits: Array,  # B S V
    embeddings: Array,  # V D
    xt: Array,  # B S D
) -> Array:
    weights = jax.nn.softmax(logits, axis=-1)
    # TODO: Brute fore for correctness, optimize later.
    e_scores = (rearrange(embeddings, 'v d -> 1 1 v d')
                - rearrange(xt, 'b s d -> b s 1 d'))  # B S V D
    scores = jnp.einsum('b s v d, b s v -> b s d', e_scores, weights)  # B S D
    return scores


def sample(
    params: ArrayTree,
    config: SamplingConfig,
    steps: int,
    seed: int,
    sequence_length: Optional[int] = None,
) -> Iterator[Tuple[Array, str]]:
    def sample_step(
        xt: Array,
        t: Array,
        next_t: Array,
    ) -> Tuple[Array, Array]:
        model = nn.Model.from_config(config)
        logits = model(xt, t[None, None], False)
        scores = score_interpolation(logits, model.embeddings, xt)
        xt = sampling_step(xt, scores, t, next_t)
        return xt, logits

    rng = jax.random.PRNGKey(seed)
    sequence_length = sequence_length or config.sequence_length
    tokenizer = data.get_tokenizer(config)
    sample_step_hk = hk.without_apply_rng(hk.transform(sample_step))
    fn = partial(sample_step_hk.apply, params)
    ts = jnp.linspace(1, 0, steps + 1)
    xt = jax.random.normal(rng, (1, sequence_length, config.model_dim))
    for t, next_t in zip(ts[:-1], ts[1:]):
        xt, logits = fn(xt, t, next_t)
        indices = logits.argmax(axis=-1)[0]
        text = tokenizer.decode(indices)
        yield xt, text
