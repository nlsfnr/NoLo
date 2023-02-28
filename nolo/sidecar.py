"""Auxiliary functions for training. Tightly coupled to the training module,
the reason this is in a separate file is solely to reduce visual clutter and
keep training.py focussed on machine learning."""
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator

import haiku as hk
import jax.numpy as jnp
import optax
import yaml
from chex import ArrayTree

from . import common, data, nn
from .training import Telemetry, TrainingConfig, get_optimizer, train

logger = logging.getLogger(__name__)


def train_new(
    config: TrainingConfig,
    seed: int,
) -> Iterator[Telemetry]:
    def model_fn() -> None:
        model = nn.Model.from_config(config)
        x = model.embed(jnp.zeros((1, config.sequence_length), dtype=jnp.int32))
        model(x, jnp.zeros((1, config.sequence_length)), is_training=True)

    rngs = hk.PRNGSequence(seed)
    params = hk.transform(model_fn).init(next(rngs))
    params_n = hk.data_structures.tree_size(params)
    params_mb = round(hk.data_structures.tree_bytes(params) / 1e6, 2)
    logger.info(f'Model parameters: {params_n:,} ({params_mb:.2f} MB)')
    opt_state = get_optimizer(config).init(params)
    batches = data.get_batches(config, seed)
    return train(config, batches, params, opt_state, 0, rngs, seed)


def save_checkpoint(
    path: Path,
    config: common.Config,
    params: ArrayTree,
    opt_state: optax.MultiStepsState,
    rngs: hk.PRNGSequence,
    step: int,
    seed: int,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config.to_yaml(path / "config.yaml")
    with open(path / "params.pkl", "wb") as f:
        pickle.dump(params, f)
    with open(path / "opt_state.pkl", "wb") as f:
        pickle.dump(opt_state, f)
    with open(path / "rngs.pkl", "wb") as f:
        pickle.dump(rngs, f)
    with open(path / "other.yaml", "w") as f:
        yaml.dump(dict(step=step, seed=seed), f)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
) -> Dict[str, Any]:
    config = common.Config.from_yaml(path / "config.yaml")
    with open(path / "params.pkl", "rb") as f:
        params = pickle.load(f)
    with open(path / "opt_state.pkl", "rb") as f:
        opt_state = pickle.load(f)
    with open(path / "rngs.pkl", "rb") as f:
        rngs_internal_state = pickle.load(f).internal_state
    rngs = hk.PRNGSequence(0)
    rngs.replace_internal_state(rngs_internal_state)
    with open(path / "other.yaml", "r") as f:
        other = yaml.safe_load(f)
    logger.info(f"Loaded checkpoint from {path}")
    return dict(
        config=config,
        params=params,
        opt_state=opt_state,
        rngs=rngs,
        step=other["step"],
        seed=other["seed"],
    )


def train_from_checkpoint(
    path: Path,
) -> Iterator[Telemetry]:
    checkpoint = load_checkpoint(path)
    config = checkpoint["config"]
    assert isinstance(config, TrainingConfig)
    params = checkpoint["params"]
    opt_state = checkpoint["opt_state"]
    rngs = checkpoint["rngs"]
    step = checkpoint["step"]
    seed = checkpoint["seed"]
    batches = data.get_batches(config, skip=step, seed=seed)
    return train(config, batches, params, opt_state, step, rngs, seed)


def log(telemetry: Iterator[Telemetry]) -> Iterator[Telemetry]:
    for t in telemetry:
        logger.info(f"{t.step:6d} | loss={t.loss:.4f}")
        yield t


def save_checkpoints(
    telemetry: Iterator[Telemetry],
    path: Path,
) -> Iterator[Telemetry]:
    path.mkdir(parents=True, exist_ok=True)
    for t in telemetry:
        if t.step % t.config.checkpoint_interval == 0:
            save_checkpoint(
                path,
                t.config,  # type: ignore
                t.params,
                t.opt_state,
                t.rngs,
                t.step,
                t.seed,
            )
        yield t