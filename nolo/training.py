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
import jmp

from . import data, diffusion, nn


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
    label_smoothing: float
    # AMP
    use_half_precision: bool
    loss_scale_period: Optional[int]
    initial_loss_scale_log2: Optional[int]
    # Telemetry and checkpoints
    telemetry_interval: int
    checkpoint_interval: int
    plot_interval: int


@dataclass
class Telemetry:
    step: int
    loss: float
    params: ArrayTree
    opt_state: optax.MultiStepsState
    loss_scale: jmp.LossScale
    rngs: hk.PRNGSequence
    config: TrainingConfig
    seed: int


def train(
    config: TrainingConfig,
    batches: Iterable[np.ndarray],
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    loss_scale: jmp.LossScale,
    step: int,
    rngs: hk.PRNGSequence,
    seed: int,
) -> Iterator[Telemetry]:
    train_step_fn = jax.jit(partial(train_step, config=config), static_argnums=(5,))
    losses = []
    for batch in batches:
        collect_telemetry = step % config.telemetry_interval == 0
        out = train_step_fn(
            params, opt_state, loss_scale, next(rngs), jnp.asarray(batch), collect_telemetry
        )
        params, opt_state, loss_scale, telemetry_data = out
        losses.append(float(telemetry_data["loss"]))
        if collect_telemetry:
            yield Telemetry(
                step=step,
                loss=float(jnp.mean(jnp.asarray(losses))),
                params=params,
                opt_state=opt_state,
                loss_scale=loss_scale,
                rngs=rngs,
                config=config,
                seed=seed,
            )
            losses.clear()
        step += 1


def train_step(
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    loss_scale: jmp.LossScale,
    rng: PRNGKey,
    batch: Array,
    collect_telemetry: bool,
    *,
    config: TrainingConfig,
) -> Tuple[ArrayTree, ArrayTree, jmp.LossScale, Telemetry]:
    optimizer = get_optimizer(config)
    grad_hk = hk.transform(
        partial(loss_fn, config=config, collect_telemetry=collect_telemetry)
    )
    grad_fn = jax.grad(grad_hk.apply, has_aux=True)
    gradients, telemetry = grad_fn(params, rng, batch, loss_scale)
    gradients = loss_scale.unscale(gradients)
    gradients_finite = jmp.all_finite(gradients)
    loss_scale = loss_scale.adjust(gradients_finite)
    updates, new_opt_state = optimizer.update(gradients, opt_state)
    new_params = optax.apply_updates(params, updates)
    # Only actually update the params and opt_state if all gradients were finite
    opt_state, params = jmp.select_tree(
        gradients_finite,
        (new_opt_state, new_params),
        (opt_state, params))
    return params, opt_state, loss_scale, telemetry


def loss_fn(
    indices: Array,
    loss_scale: jmp.LossScale,
    collect_telemetry: bool,
    *,
    config: TrainingConfig,
) -> Tuple[Array, Dict[str, Any]]:
    model = nn.Model.from_config(config)
    x0 = model.embed(indices)
    B, S, D = x0.shape
    noise = jax.random.normal(hk.next_rng_key(), (B, S, D))
    t = jax.random.uniform(hk.next_rng_key(), (B,))
    xt = diffusion.pertubation_kernel(x0, t, noise)
    logits = model(xt, t, is_training=True)
    labels = jax.nn.one_hot(indices, logits.shape[-1])
    smoothed_labels = optax.smooth_labels(labels, config.label_smoothing)
    losses = optax.softmax_cross_entropy(logits, smoothed_labels)
    loss = jnp.mean(losses)
    telemetry = (
        dict(loss=loss)  # TODO: Add logits, labels, etc.
        if collect_telemetry
        else dict(loss=loss)
    )
    return loss_scale.scale(loss), telemetry


def get_optimizer(config: TrainingConfig) -> optax.MultiSteps:
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(config.lr_min, config.lr_max, config.lr_warmup_steps),
            (
                optax.cosine_decay_schedule(
                    config.lr_max,
                    config.lr_decay_steps,
                    alpha=config.lr_min / config.lr_max,
                )
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
