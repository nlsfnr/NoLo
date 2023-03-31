from functools import partial
from typing import Iterator, NamedTuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from chex import ArrayTree
from jax import Array

from . import data, nn
from .common import Config, expand_dims


def alpha_schedule(t: Array) -> Array:
    return (1. - t) ** 2


def pertubation_kernel(
    x0: Array,  # B S D
    t: Array,  # B S
    noise: Array,  # B S D
) -> Array:
    alpha = expand_dims(alpha_schedule(t), x0)
    return x0 * alpha**0.5 + noise * (1 - alpha) ** 0.5


def x0_pred(
    logits: Array,  # B S V
    embeddings: Array,  # V D
) -> Array:  # B S D
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("v d, b s v -> b s d", embeddings, weights)


def noise_pred(
    x0: Array,  # B S D
    xt: Array,  # B S D
    t: Array,  # B S
) -> Array:  # B S D
    alpha = expand_dims(alpha_schedule(t), x0)
    noise = (xt - alpha**0.5 * x0) / (1 - alpha) ** 0.5
    return noise


class SamplingStepRV(NamedTuple):
    x0_pred: Array
    noise_pred: Array
    next_xt: Array


def sampling_step(
    xt: Array,  # B S D
    logits: Array,  # B S V
    embeddings: Array,  # V D
    t: Array,  # B S
    next_t: Array,  # B S
    noise: Optional[Array] = None,  # B S D
) -> SamplingStepRV:
    x0 = x0_pred(logits, embeddings)
    noise = noise or noise_pred(x0, xt, t)
    next_xt = pertubation_kernel(x0, next_t, noise)
    return SamplingStepRV(x0, noise, next_xt)


class SampleYV(NamedTuple):
    x0_pred: Array
    noise_pred: Array
    next_xt: Array
    logits: Array
    t: Array
    next_t: Array
    alpha: Array
    next_alpha: Array
    text: str


def sample(
    params: ArrayTree,
    config: Config,
    steps: int,
    seed: int,
    sequence_length: Optional[int] = None,
    xt: Optional[Array] = None,
) -> Iterator[SampleYV]:
    class _SampleStepRV(NamedTuple):
        x0_pred: Array
        noise_pred: Array
        next_xt: Array
        logits: Array

    def sample_step(
        xt: Array,
        t: Array,
        next_t: Array,
    ) -> _SampleStepRV:
        model = nn.Model.from_config(config)
        logits = model(xt, t[None, None], False)
        out = sampling_step(
            xt=xt,
            logits=logits,
            embeddings=model.embeddings,
            t=t,
            next_t=next_t,
        )
        return _SampleStepRV(*out, logits)

    rngs = hk.PRNGSequence(seed)
    sequence_length = sequence_length or config.data.length
    tokenizer = data.tokenizer_from_config(config)
    sample_step_hk = hk.without_apply_rng(hk.transform(sample_step))
    fn = partial(sample_step_hk.apply, params)
    ts = jnp.linspace(1, 0, steps + 1)
    xt = xt or jax.random.normal(
        next(rngs), (1, sequence_length, config.model.model_dim)
    )
    for t, next_t in zip(ts[:-1], ts[1:]):
        out: _SampleStepRV
        out = fn(xt, t, next_t)
        xt = out.next_xt
        indices = out.logits.argmax(axis=-1)[0]
        text = tokenizer.decode(indices)
        yield SampleYV(
            x0_pred=out.x0_pred,
            noise_pred=out.noise_pred,
            next_xt=xt,
            logits=out.logits,
            t=t,
            next_t=next_t,
            alpha=alpha_schedule(t),
            next_alpha=alpha_schedule(next_t),
            text=text,
        )
