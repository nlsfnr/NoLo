from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Tuple, TypeVar, Union

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from chex import ArrayTree, PRNGKey
from einops import rearrange
from jax import Array

from . import diffusion, nn
from .common import Config, get_logger

logger = get_logger()

ArrayTreeT = TypeVar("ArrayTreeT", bound=ArrayTree)
T = TypeVar("T")


@dataclass
class TrainStep:
    step: int
    has_updated: bool
    loss: float
    sample_losses: Optional[Array]
    gradients_finite: bool
    loss_scale_log2: float
    loss_density: Optional[Array]
    gradients: Optional[ArrayTree]
    params: Optional[ArrayTree]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Save:
    path: Path
    step: int
    config: Config
    params: ArrayTree
    loss_scale: jmp.LossScale
    loss_density: Array
    opt_state: optax.MultiStepsState
    rng_key: PRNGKey
    seed: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EndOfTraining:
    timestamp: datetime = field(default_factory=datetime.now)


Event = Union[TrainStep, Save, EndOfTraining]


class StopTraining(Exception):
    pass


class Trainer(threading.Thread):
    def __init__(
        self,
        *,
        batch_queue: Union[queue.Queue[np.ndarray], queue.Queue[Array]],
        event_queue: queue.Queue[Event],
        config: Config,
        seed: int,
        termination_event: Optional[threading.Event] = None,
        timeout: float = 0.1,
        rng_key: Optional[PRNGKey] = None,
        params: Optional[ArrayTree] = None,
        opt_state: Optional[optax.MultiStepsState] = None,
        loss_scale: Optional[jmp.LossScale] = None,
        loss_density: Optional[Array] = None,
        step: Optional[int] = None,
        save_frequency: Optional[int] = None,
        save_directory: Optional[Path] = None,
        log_gradients_frequency: Optional[int] = None,
        log_params_frequency: Optional[int] = None,
        log_sample_losses_frequency: Optional[int] = None,
        log_param_size_on_init: bool = True,
    ) -> None:
        super().__init__()
        self.batch_queue = batch_queue
        self.event_queue = event_queue
        self.config = config
        self.seed = seed
        self.termination_event = termination_event or threading.Event()
        self.timeout = timeout
        self.rng_key = rng_key or jax.random.PRNGKey(seed)
        self.params = params or nn.Model.get_params(config, seed + 1)
        self.opt_state = opt_state or _get_optimizer(config).init(self.params)
        self.step = step or 0
        self.loss_scale = loss_scale or _get_loss_scale(config, self.step)
        # Initialize loss density to log(vocab_size) for each bin. This is the
        # expected cross-entropy loss for random guessing.
        self.loss_density = loss_density or jnp.full(
            (config.time_warping.bins,), jnp.log(config.model.vocabulary_size)
        )
        self.save_frequency = save_frequency
        self.save_directory = save_directory
        self.log_gradients_frequency = log_gradients_frequency
        self.log_params_frequency = log_params_frequency
        self.log_param_size_on_init = log_param_size_on_init
        self.log_sample_losses_frequency = log_sample_losses_frequency
        self._exception: Optional[Exception] = None
        self._policy = _set_amp_policy(self.config)
        # Two training functions, one which returns gradients and one which does not.
        self._train_step = jax.pmap(
            partial(_train_step, config=self.config, axis_name="device"),
            axis_name="device",
            static_broadcasted_argnums=(0, 1),

        )

    def run(self) -> None:
        logger.info(f"Starting training loop at step {self.step}.")
        try:
            self.loop()
        except StopTraining:
            logger.info(f"Training loop terminated at step {self.step}.")
        except Exception as e:
            logger.exception(f"Training loop failed at step {self.step}.")
            self._exception = e
            raise

    def join(self, timeout: Optional[float] = None) -> None:
        super().join(timeout)
        if self._exception is not None:
            raise self._exception

    def terminate(self, timeout: Optional[float] = None) -> Trainer:
        logger.info("Terminating training thread.")
        self.termination_event.set()
        self.join(timeout)
        return self

    def save_and_terminate(self, timeout: Optional[float] = None) -> Trainer:
        self._save()
        return self.terminate(timeout)

    def emit_end_of_training_event(self) -> Trainer:
        if self.is_alive():
            raise RuntimeError(
                "Cannot emit EndOfTraining event while running."
                " Call terminate() or save_and_terminate() first."
            )
        self.event_queue.put(EndOfTraining())
        return self

    def __enter__(self) -> Trainer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self.is_alive():
            self.terminate()
        self.join()

    def loop(self) -> Trainer:
        self.params = _broadcast_to_devices(self.params)
        self.opt_state = _broadcast_to_devices(self.opt_state)
        self.loss_scale = _broadcast_to_devices(self.loss_scale)
        self.loss_density = _broadcast_to_devices(self.loss_density)
        # Set the policy again, as it is bound to the current thread
        self._policy = _set_amp_policy(self.config)
        while True:
            with_gradients = (
                self.log_gradients_frequency is not None
                and self.step % self.log_gradients_frequency == 0
            )
            with_params = (
                self.log_params_frequency is not None
                and self.step % self.log_params_frequency == 0
            )
            with_sample_losses = (
                self.log_sample_losses_frequency is not None
                and self.step % self.log_sample_losses_frequency == 0
            )
            self.train_step(
                self._fetch_batch(),
                with_gradients=with_gradients,
                with_sample_losses=with_sample_losses,
                with_params=with_params,
            )

    def train_step(
        self,
        batch: Array,
        *,
        with_gradients: bool,
        with_sample_losses: bool,
        with_params: bool,
    ) -> Trainer:
        """Performs one training step. Notably, this does not mean that the parameters will be
        updated. This is because we use a multi-step optimizer, which means that we accumulate
        gradients over multiple steps before updating the parameters."""
        # Split the batch across devices.
        device_count = jax.device_count()
        batch = self._policy.cast_to_compute(batch)
        batch = rearrange(batch, "(d b) ... -> d b ...", d=device_count)
        # Get a new RNG key for each device
        self.rng_key, subkey = jax.random.split(self.rng_key)
        subkeys = jax.random.split(subkey, device_count)
        # Run the training step.
        retval: _TrainStepRV
        retval = self._train_step(
            with_gradients,
            with_sample_losses,
            indices=batch,
            params=self.params,
            opt_state=self.opt_state,
            loss_scale=self.loss_scale,
            loss_density=self.loss_density,
            rng_key=subkeys,
        )
        (
            self.params,
            self.opt_state,
            self.loss_scale,
            self.loss_density,
            loss,
            sample_losses,
            gradients,
            gradients_finite,
            has_updated,
        ) = retval
        # Emit the event.
        gffd = _get_from_first_device
        self._emit_event(
            TrainStep(
                step=self.step,
                has_updated=bool(gffd(has_updated)),
                loss=float(jnp.mean(loss)),
                sample_losses=_concat_from_devices(sample_losses),
                gradients=gffd(gradients),
                params=gffd(self.params) if with_params else None,
                gradients_finite=bool(gffd(gradients_finite)),
                loss_scale_log2=round(
                    float(gffd(jnp.log2(self.loss_scale.loss_scale)))
                ),
                loss_density=gffd(self.loss_density),
            )
        )
        # If this is a gradient-accumulation step, don't update the step count.
        if not has_updated.all():
            return self
        # Update the step count.
        self.step += 1
        # Save the model.
        if (
            self.save_frequency is not None
            and self.save_directory is not None
            and self.step % self.save_frequency == 0
        ):
            self._save()
        return self

    def _save(self) -> Trainer:
        """Emits a save event."""
        if self.save_directory is None:
            return self
        gffd = _get_from_first_device
        self._emit_event(
            Save(
                path=self.save_directory,
                step=self.step,
                config=self.config,
                params=gffd(self.params),
                loss_scale=gffd(self.loss_scale),
                loss_density=gffd(self.loss_density),
                opt_state=gffd(self.opt_state),
                rng_key=self.rng_key,
                seed=self.seed,
            )
        )
        return self

    def _emit_event(
        self,
        event: Event,
    ) -> Trainer:
        """Emits an event."""
        while True:
            if self.termination_event.is_set():
                raise StopTraining("Termination event set when emitting event.")
            try:
                self.event_queue.put(event, timeout=self.timeout)
                break
            except queue.Full:
                pass
        return self

    def _fetch_batch(
        self,
    ) -> Array:
        """Fetches a batch from the batch queue."""
        cpu = jax.devices("cpu")[0]
        while True:
            if self.termination_event.is_set():
                raise StopTraining("Termination event set when fetching batch.")
            try:
                batch = self.batch_queue.get(timeout=self.timeout)
                batch_jnp = jnp.asarray(batch, dtype=jnp.int32)
                batch_jnp = jax.device_put(batch_jnp, device=cpu)
                return batch_jnp
            except queue.Empty:
                pass


class _TrainStepRV(NamedTuple):
    params: ArrayTree
    opt_state: optax.MultiStepsState
    loss_scale: jmp.LossScale
    loss_density: Array
    loss: Array
    sample_losses: Optional[Array]
    gradients: Optional[ArrayTree]
    gradients_finite: Array
    has_updated: Array


def _train_step(
    with_gradients: bool,
    with_sample_losses: bool,
    *,
    indices: Array,
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    loss_scale: jmp.LossScale,
    loss_density: Array,
    rng_key: PRNGKey,
    config: Config,
    axis_name: str,
) -> _TrainStepRV:
    loss_fn = hk.transform(partial(_loss_fn, config=config,
                                   with_sample_losses=with_sample_losses)).apply
    grad_fn = jax.grad(loss_fn, has_aux=True)
    optimizer = _get_optimizer(config)
    loss_aux: _LossFnAux
    gradients, loss_aux = grad_fn(
        params,
        rng_key,
        indices=indices,
        loss_scale=loss_scale,
        loss_density=loss_density,
    )
    gradients = jax.lax.pmean(gradients, axis_name=axis_name)
    gradients = loss_scale.unscale(gradients)
    gradients_finite = jmp.all_finite(gradients)
    loss_scale = loss_scale.adjust(gradients_finite)
    updates, new_opt_state = optimizer.update(gradients, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_loss_density = jax.lax.pmean(loss_aux.loss_density, axis_name=axis_name)
    opt_state, params, loss_density = jmp.select_tree(
        gradients_finite,
        (new_opt_state, new_params, new_loss_density),
        (opt_state, params, loss_density),
    )
    return _TrainStepRV(
        params=params,
        opt_state=opt_state,
        loss_scale=loss_scale,
        loss_density=loss_density,
        loss=loss_aux.loss,
        sample_losses=loss_aux.sample_losses,
        gradients=gradients if with_gradients else None,
        gradients_finite=gradients_finite,
        has_updated=optimizer.has_updated(opt_state),
    )


class _LossFnAux(NamedTuple):
    loss: Array
    loss_density: Array
    sample_losses: Optional[Array]


def _loss_fn(
    *,
    indices: Array,
    loss_scale: jmp.LossScale,
    loss_density: Array,
    config: Config,
    with_sample_losses: bool,
) -> Tuple[Array, _LossFnAux]:
    # Sample the timestep proportionally to the loss density.
    timesteps = jax.random.choice(
        hk.next_rng_key(),
        a=len(loss_density),
        shape=indices.shape[:-1],
        replace=True,
        p=loss_density / jnp.sum(loss_density),
    )
    ts = timesteps / len(loss_density)
    model = nn.Model.from_config(config)
    x = model.embed(indices)
    noise = jax.random.normal(hk.next_rng_key(), x.shape)
    inputs = diffusion.pertubation_kernel(x, ts, noise)
    logits = model(inputs, ts, is_training=True)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits, indices)
    sample_losses = token_losses.mean(axis=-1)
    # Update the loss_density. Notably, if two timesteps are equal, this only considers
    # one of them. This can be fixed later.
    new_loss_density = loss_density.at[timesteps].set(sample_losses)
    alpha = config.time_warping.alpha
    loss_density = alpha * loss_density + (1 - alpha) * new_loss_density
    loss = jnp.mean(sample_losses)
    aux = _LossFnAux(
        loss=loss,
        loss_density=loss_density,
        sample_losses=sample_losses if with_sample_losses else None,
    )
    return loss_scale.scale(loss), aux


def _get_optimizer(
    config: Config,
) -> optax.MultiSteps:
    cfg = config.optimizer
    # Assemble the lr schedule
    parts = [optax.linear_schedule(cfg.lr_min, cfg.lr_max, cfg.lr_warmup_steps)]
    if cfg.lr_decay_steps is None:
        parts += [optax.constant_schedule(cfg.lr_max)]
    else:
        parts += [
            (
                optax.cosine_decay_schedule(
                    cfg.lr_max, cfg.lr_decay_steps, alpha=cfg.lr_min / cfg.lr_max
                )
            )
        ]
    lr_schedule = optax.join_schedules(parts, [cfg.lr_warmup_steps])
    # Assemble the optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.gradient_clip_norm),
        # Adam + weight decay = AdamW
        optax.scale_by_adam(b1=cfg.adam_b1, b2=cfg.adam_b2, eps=cfg.adam_eps),
        optax.add_decayed_weights(weight_decay=cfg.weight_decay),
        # We want gradient descent not ascent, so we negate the learning rate
        optax.scale_by_schedule(lambda step: -lr_schedule(step)),
    )
    return optax.MultiSteps(optimizer, _batch_size_schedule(config))


def _batch_size_schedule(config: Config) -> Callable[[Array], Array]:
    step_gas_pairs = tuple(config.optimizer.gradient_accumulation_steps)
    if not all(isinstance(s, int) and isinstance(g, int) for s, g in step_gas_pairs):
        raise TypeError(
            f"Expected gradient_accumulation_steps to be a sequence of (int, int) pairs, got "
            f"{step_gas_pairs}"
        )
    if not all(s >= 0 and g > 0 for s, g in step_gas_pairs):
        raise ValueError(
            f"Expected gradient_accumulation_steps to be a sequence of (int, int) pairs with "
            f"non-negative steps and positive gas, got {step_gas_pairs}"
        )
    pairs = sorted(step_gas_pairs, key=lambda x: x[0])
    steps, gass = map(jnp.array, zip(*pairs))
    return lambda step: jnp.max(jnp.where(steps <= step, gass, 1))


def _get_loss_scale(
    config: Config,
    step: int,
) -> jmp.LossScale:
    if not config.mixed_precision.enable:
        return jmp.NoOpLossScale()
    return jmp.DynamicLossScale(
        loss_scale=jnp.asarray(
            2**config.mixed_precision.initial_scale_log2,
            dtype=jnp.float32,
        ),
        counter=jnp.asarray(step, dtype=jnp.int32),
        period=config.mixed_precision.scale_period,
    )


def _set_amp_policy(config: Config) -> jmp.Policy:
    full = jnp.dtype(jnp.float32)
    half = jnp.dtype(jnp.float16 if config.mixed_precision.enable else jnp.float32)
    policy = jmp.Policy(param_dtype=full, compute_dtype=full, output_dtype=half)
    hk.mixed_precision.set_policy(hk.LayerNorm, policy)
    policy = jmp.Policy(param_dtype=full, compute_dtype=half, output_dtype=half)
    hk.mixed_precision.set_policy(nn.Block, policy)
    hk.mixed_precision.set_policy(nn.MultiHeadAttention, policy)
    hk.mixed_precision.set_policy(nn.FeedForward, policy)
    hk.mixed_precision.set_policy(hk.Embed, policy)
    hk.mixed_precision.set_policy(hk.Linear, policy)
    policy = jmp.Policy(param_dtype=full, compute_dtype=half, output_dtype=full)
    hk.mixed_precision.set_policy(nn.Model, policy)
    return policy


def _broadcast_to_devices(obj: T) -> T:
    """Broadcasts a tree of arrays to all devices."""
    device_count = jax.device_count()

    def fn(x: Array) -> Array:
        x = jax.device_put(x, jax.devices("cpu")[0])
        x = jnp.broadcast_to(x, (device_count, *x.shape)) if isinstance(x, Array) else x
        return jax.pmap(lambda x: x, axis_name="batch")(x)

    return jax.tree_util.tree_map(fn, obj)


def _get_from_first_device(obj: T) -> T:
    """Gets a tree of arrays from the first device, putting it on the CPU."""
    cpu = jax.devices("cpu")[0]
    fn = lambda x: jax.device_put(x[0], cpu) if isinstance(x, Array) else x
    return jax.tree_util.tree_map(fn, obj)


def _concat_from_devices(obj: T) -> T:
    """Concatenates a tree of arrays from all devices."""
    cpu = jax.devices("cpu")[0]
    fn = lambda x: jnp.concatenate(jax.device_put(x, cpu)) if isinstance(x, Array) else x
    return jax.tree_util.tree_map(fn, obj)
