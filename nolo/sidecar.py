import contextlib
import io
import os
import pickle
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import optax
import yaml
from chex import ArrayTree, PRNGKey
from jax import Array
from wandb.sdk.wandb_run import Run as WandbRun

import wandb

from .common import Config, get_logger
from .training import EndOfTraining, Event, Save, TrainStep

logger = get_logger()


def accumulate_gac_steps(
    *,
    events: Iterable[Event],
) -> Iterable[Event]:
    """Accumulate gradient accumulation steps. This way, only the last
    step of the accumulation is propagated, with the mean loss over all
    of its sub-steps."""
    losses = []
    sample_losses = []
    gradients = None
    params = None
    gradients_finite = True
    for event in events:
        if isinstance(event, EndOfTraining):
            yield event
            return
        if not isinstance(event, TrainStep):
            yield event
            continue
        losses.append(event.loss)  # Take the mean of the losses.
        if event.sample_losses is not None:
            sample_losses.append(event.sample_losses)
        gradients_finite = gradients_finite and event.gradients_finite  # All finite.
        gradients = event.gradients or gradients  # Only keep the last gradients.
        params = event.params or params  # Only keep the last params.
        if not event.has_updated:
            continue
        yield TrainStep(
            step=event.step,
            has_updated=True,
            loss=statistics.mean(losses),
            sample_losses=jnp.concatenate(sample_losses) if sample_losses else None,
            loss_density=event.loss_density,
            gradients_finite=gradients_finite,
            loss_scale_log2=event.loss_scale_log2,
            gradients=gradients,
            params=params,
            timestamp=event.timestamp,
        )
        losses = []
        sample_losses = []
        gradients = None
        params = None
        gradients_finite = True


def log_losses(
    *,
    events: Iterable[Event],
    frequency: int,
    log_fn: Callable[[str], None] = logger.info,
) -> Iterable[Event]:
    """Log the mean loss and standard deviation over the last `frequency`
    steps."""
    losses = []
    for event in events:
        if isinstance(event, EndOfTraining):
            yield event
            return
        if not isinstance(event, TrainStep):
            yield event
            continue
        losses.append(event.loss)
        if event.step % frequency == 0 and losses:
            mean, std = statistics.mean(losses), statistics.pstdev(losses)
            items = (
                f"Step: {event.step:>6}",
                f"Loss: {mean:0.6f} ± {std:0.6f}",
            )
            log_fn(" | ".join(items))
            losses = []
        if not event.gradients_finite:
            logger.info(
                f"Step: {event.step:>6} | Non-finite gradients, "
                f"loss scale (log2): {event.loss_scale_log2}"
            )
        yield event


def log_time_per_step(
    *,
    events: Iterable[Event],
    frequency: int,
    percentiles: Iterable[int],
    log_fn: Callable[[str], None] = logger.info,
) -> Iterable[Event]:
    """Log the time per step over the last `frequency` steps. Concretely, log
    the given percentiles of the time between each step."""
    if frequency < 100:
        raise ValueError(f"Expected frequency to be at least 100, got {frequency}")
    percentiles = tuple(percentiles)
    timestamps = []
    for event in events:
        if isinstance(event, EndOfTraining):
            yield event
            return
        if not isinstance(event, TrainStep):
            yield event
            continue
        timestamps.append(event.timestamp)
        if (
            event.has_updated
            and event.step % frequency == 0
            and len(timestamps) >= max(percentiles)
        ):
            deltas = [b - a for a, b in zip(timestamps[:-1], timestamps[1:])]
            deltas_seconds = [delta.total_seconds() for delta in deltas]
            points = statistics.quantiles(deltas_seconds, n=101, method="inclusive")
            points = [points[p] for p in percentiles]
            points_str = ", ".join(
                f"{p}%: {t:0.4f}s" for p, t in zip(percentiles, points)
            )
            items = (
                f"Step: {event.step:>6}",
                "s/step: " + points_str,
            )
            log_fn(" | ".join(items))
        yield event


def detect_anomalies(
    events: Iterable[Event],
) -> Iterable[Event]:
    """Detect anomalies in the gradients and loss such as NaNs and infs."""
    for event in events:
        if isinstance(event, EndOfTraining):
            yield event
            return
        if not isinstance(event, TrainStep):
            yield event
            continue
        if np.isnan(event.loss):
            logger.error(f"Loss is NaN at step {event.step}")
        if np.isinf(event.loss):
            logger.error(f"Loss is infinite at step {event.step}")
        if event.gradients is not None:
            non_finites = _find_anomalies(event.gradients)
            for key, reason in non_finites:
                logger.error(f"Gradient {key} {reason} at step {event.step}")
        yield event


def _find_anomalies(
    x: ArrayTree,
    prefix: str = "",
) -> Iterable[Tuple[str, str]]:
    if isinstance(x, (np.ndarray, Array)):
        if np.isnan(x).any():
            yield prefix, "contains NaNs"
        if np.isinf(x).any():
            yield prefix, "contains infinities"
        if np.std(x) == 0:
            yield prefix, "has zero std"
    elif isinstance(x, dict):
        for key, value in x.items():
            yield from _find_anomalies(value, prefix + f"{key}.")
    elif isinstance(x, Iterable):
        for i, value in enumerate(x):
            yield from _find_anomalies(value, prefix + f"{i}.")
    else:
        raise TypeError(f"Unexpected type {type(x)}")


@contextlib.contextmanager
def atomic_open(path: Path, mode: str = "w") -> Generator[io.IOBase, None, None]:
    f = tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False)
    try:
        yield f  # type: ignore
        f.close()
        os.replace(f.name, path)
    finally:
        if os.path.exists(f.name):
            os.remove(f.name)


def save_to_directory(
    *,
    events: Iterable[Event],
) -> Iterable[Event]:
    """Save the parameters, optimizer state, etc. to the directory specified
    in the `Save` event."""
    for event in events:
        if isinstance(event, EndOfTraining):
            yield event
            return
        if not isinstance(event, Save):
            yield event
            continue
        path = Path(event.path)
        path.mkdir(parents=True, exist_ok=True)
        event.config.to_yaml(path / "config.yaml")
        with atomic_open(path / "params.pkl", "wb") as f:
            pickle.dump(event.params, f)
        with atomic_open(path / "opt_state.pkl", "wb") as f:
            pickle.dump(event.opt_state, f)
        with atomic_open(path / "loss_density.pkl", "wb") as f:
            pickle.dump(event.loss_density, f)
        with atomic_open(path / "rng_key.pkl", "wb") as f:
            pickle.dump(event.rng_key, f)
        with atomic_open(path / "step.txt", "w") as f:
            f.write(str(event.step))
        with atomic_open(path / "seed.txt", "w") as f:
            f.write(str(event.seed))
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Step: {event.step:>6} | Saved model to {path}")
        yield event


@dataclass
class LoadResultForInference:
    config: Config
    params: ArrayTree
    loss_density: Array


@dataclass
class LoadResult(LoadResultForInference):
    step: int
    seed: int
    rng_key: PRNGKey
    opt_state: optax.MultiStepsState


def load_from_directory_for_inference(
    *,
    path: Path,
) -> LoadResultForInference:
    """Load the parameters and config from the directory specified in the `Load` event."""
    config = Config.from_yaml(path / "config.yaml")
    with open(path / "params.pkl", "rb") as f:
        params = pickle.load(f)
        assert isinstance(params, dict)
    with open(path / "loss_density.pkl", "rb") as f:
        loss_density = pickle.load(f)
        assert isinstance(loss_density, Array)
        assert loss_density.ndim == 1
    return LoadResultForInference(
        config=config,
        params=params,
        loss_density=loss_density,
    )


def load_from_directory(
    *,
    path: Path,
) -> LoadResult:
    """Load the parameters, optimiser state, config etc. from the directory
    specified."""
    config = Config.from_yaml(path / "config.yaml")
    with open(path / "params.pkl", "rb") as f:
        params = pickle.load(f)
        assert isinstance(params, dict)
    with open(path / "opt_state.pkl", "rb") as f:
        opt_state = pickle.load(f)
        assert isinstance(opt_state, optax.MultiStepsState)
    with open(path / "loss_density.pkl", "rb") as f:
        loss_density = pickle.load(f)
        assert isinstance(loss_density, Array)
        assert loss_density.ndim == 1
    with open(path / "rng_key.pkl", "rb") as f:
        rng_key = pickle.load(f)
    with open(path / "step.txt") as f:
        step = int(f.read().strip())
    with open(path / "seed.txt") as f:
        seed = int(f.read().strip())
    return LoadResult(
        config=config,
        step=step,
        seed=seed,
        rng_key=rng_key,
        params=params,
        opt_state=opt_state,
        loss_density=loss_density,
    )


def log_to_wandb(
    *,
    events: Iterable[Event],
    run: WandbRun,
) -> Iterable[Event]:
    """Log the parameters, gradients, loss, etc. to wandb."""
    for event in events:
        if isinstance(event, EndOfTraining):
            yield event
            return
        if isinstance(event, Save):
            event.path.mkdir(parents=True, exist_ok=True)
            with atomic_open(event.path / "wandb-run.yaml", "w") as f:
                yaml.dump(dict(id=run.id, project=run.project, group=run.group), f)
        elif isinstance(event, TrainStep):
            data: Dict[str, Any] = dict(loss=event.loss)
            if event.sample_losses is not None:
                hist = wandb.Histogram(np_histogram=np.histogram(event.sample_losses, bins=64))
                data["sample_losses"] = hist
            if event.gradients is not None and event.gradients_finite:
                tuples = _to_histograms(event.gradients, "/")
                data["gradients"] = dict(tuples)
            if event.params is not None:
                tuples = _to_histograms(event.params, "/")
                data["params"] = dict(tuples)
            if event.loss_density is not None:
                hist = np.asarray(event.loss_density)
                bin_edges = np.linspace(0, 1, num=len(hist) + 1)
                data["loss_density"] = wandb.Histogram(np_histogram=(hist, bin_edges))
            run.log(data, step=event.step)
        yield event


def load_wandb_run(
    *,
    path: Path,
) -> WandbRun:
    with open(path / "wandb-run.yaml") as f:
        run_data = dict(yaml.safe_load(f))
    run = wandb.init(**run_data, resume="must")
    assert isinstance(run, WandbRun)
    return run


def new_wandb_run(
    *,
    project: Optional[str] = None,
    tags: Iterable[str] = [],
    group: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None,
) -> WandbRun:
    run = wandb.init(
        project=project,
        group=group,
        tags=tuple(tags),
        name=name,
        notes=notes,
    )
    assert isinstance(run, WandbRun)
    return run


def _to_histograms(
    x: ArrayTree,
    prefix: str = "",
    bins: int = 64,
) -> Iterable[Tuple[str, wandb.Histogram]]:
    if isinstance(x, (np.ndarray, Array)):
        x = np.asarray(x)
        x = x.flatten()
        hist = np.histogram(x, bins=bins)
        yield prefix, wandb.Histogram(np_histogram=hist)
    elif isinstance(x, dict):
        for key, value in x.items():
            key = str(key).replace("/", ".").replace(" ", "_")
            yield from _to_histograms(value, prefix=f"{prefix}.{key}", bins=bins)
    elif isinstance(x, Iterable):
        for i, value in enumerate(x):
            yield from _to_histograms(value, prefix=f"{prefix}.{i}", bins=bins)
    else:
        raise TypeError(f"Expected x to be an array dict or iterable, got {type(x)}")
