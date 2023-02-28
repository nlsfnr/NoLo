from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import Array, ArrayTree, PRNGKey

from . import data, nn


@runtime_checkable
class TrainingConfig(nn.ModelConfig, data.DataConfig, Protocol):
    lr_min: float
    lr_max: float
    lr_warmup_steps: int
    lr_decay_steps: Optional[int]
    gradient_accumulation_steps: int
    gradient_clip_norm: float
    adam_b1: float
    adam_b2: float
    telemetry_interval: int
    checkpoint_interval: int


@dataclass
class Telemetry:
    step: int
    loss: float
    params: ArrayTree
    opt_state: optax.MultiStepsState
    rngs: hk.PRNGSequence
    config: TrainingConfig
    seed: int


def train(
    config: TrainingConfig,
    batches: Iterable[np.ndarray],
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    step: int,
    rngs: hk.PRNGSequence,
    seed: int,
) -> Iterator[Telemetry]:
    train_step_fn = jax.jit(partial(train_step, config=config), static_argnums=(4,))
    for batch in batches:
        collect_telemetry = step % config.telemetry_interval == 0
        out = train_step_fn(
            params, opt_state, next(rngs), jnp.asarray(batch), collect_telemetry
        )
        params, opt_state, telemetry_data = out
        if collect_telemetry:
            yield Telemetry(
                step=step,
                loss=telemetry_data["loss"],
                params=params,
                opt_state=opt_state,
                rngs=rngs,
                config=config,
                seed=seed,
            )
        step += 1


def train_step(
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    rng: PRNGKey,
    batch: Array,
    collect_telemetry: bool,
    *,
    config: TrainingConfig,
) -> Tuple[ArrayTree, ArrayTree, Telemetry]:
    optimizer = get_optimizer(config)
    grad_hk = hk.transform(
        partial(loss_fn, config=config, collect_telemetry=collect_telemetry)
    )
    grad_fn = jax.grad(grad_hk.apply, has_aux=True)
    gradients, telemetry = grad_fn(params, rng, batch)
    updates, opt_state = optimizer.update(gradients, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, telemetry


def loss_fn(
    indices: Array,
    collect_telemetry: bool,
    *,
    config: TrainingConfig,
) -> Tuple[Array, Dict[str, Any]]:
    model = nn.Model.from_config(config)
    x0 = model.embed(indices)
    B, S, D = x0.shape
    noise = jax.random.normal(hk.next_rng_key(), (B, S, D))
    t = jax.random.uniform(hk.next_rng_key(), (B, 1))
    xt = x0 + noise * t[:, :, None]
    logits = model(xt, t, is_training=True)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, indices))
    return loss, (dict(loss=loss) if collect_telemetry else dict())


def get_optimizer(config: TrainingConfig) -> optax.MultiSteps:
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(config.lr_min, config.lr_max, config.lr_warmup_steps),
            (
                optax.cosine_decay_schedule(config.lr_max, config.lr_decay_steps)
                if config.lr_decay_steps is not None
                else optax.constant_schedule(config.lr_max)
            ),
        ],
        [config.lr_warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.scale_by_adam(config.adam_b1, config.adam_b2),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )
    return optax.MultiSteps(optimizer, config.gradient_accumulation_steps)
