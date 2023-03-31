#!/usr/bin/env python3
import queue
import sys
import threading
from itertools import islice
from pathlib import Path
from pprint import pformat
from typing import Iterator, Optional, Tuple

import click
import numpy as np
import optax
from chex import ArrayTree, PRNGKey

sys.path.append(".")

import nolo  # noqa: E402

logger = nolo.get_logger()


@click.group()
@click.option("--debug", is_flag=True)
def cli(
    debug: bool,
) -> None:
    nolo.setup_logging()
    nolo.set_debug(debug)


# fmt: off
@cli.command("train")
@click.option("--config-path", "-c", type=Path, default=None,
              help="Path to the config file.")
@click.option("--seed", "-s", type=int, default=None,
              help="Random seed.")
@click.option("--load-from", "-l", type=Path, default=None,
              help="Path to the checkpoint to load from. Default None.")
@click.option("--save-frequency", "-sf", type=int, default=250,
              help="Save frequency. Default 250.")
@click.option("--save-directory", "-sd", type=Path, default=None,
              help="Path to the directory to save checkpoints to. Default None.")
@click.option("--log-frequency", "-lf", type=int, default=10,
              help="Log frequency. Default 10.")
@click.option("--log-time-per-step-frequency", "-tf", type=int, default=100,
              help="Log time per step frequency. Default 100.")
@click.option("--log-time-per-step-percentiles", "-tp", type=int, multiple=True,
              default=(1, 50, 99), help="Percentiles to log for time per step. Default "
              "(1, 50, 99).")
@click.option("--log-gradients-frequency", "-gf", type=int, default=100,
              help="Log frequency for gradients. Default 100.")
@click.option("--log-params-frequency", "-pf", type=int, default=100,
              help="Log frequency for parameters. Default 100.")
@click.option("--log-sample-losses-frequency", "-slf", type=int, default=10,
              help="Log frequency for per-sample losses. Default 10.")
@click.option("--wandb-disable", "-wd", is_flag=True,
              help="Disable wandb.")
@click.option("--wandb-project", "-wp", type=str, default="NoLo",
              help="Wandb project name. Default NoLo.")
@click.option("--wandb-group", "-wg", type=str, default="default",
              help="Wandb group name. Default default.")
@click.option("--wandb-name", "-wn", type=str, default=None,
              help="Wandb run name. Default None.")
@click.option("--wandb-tags", "-wt", type=str, multiple=True, default=[],
              help="Wandb tags. Default []")
@click.option("--data-buffer", "-db", type=int, default=1000,
              help="Data buffer size. Default 1000.")
@click.option("--event-buffer", "-eb", type=int, default=10,
              help="Event buffer size. Default 10.")
@click.option("--detect-anomalies", "-da", is_flag=True,
              help="Detect anomalies.")
# fmt: on
def train_new(
    config_path: Optional[Path],
    seed: Optional[int],
    load_from: Optional[Path],
    save_frequency: int,
    save_directory: Optional[Path],
    log_frequency: int,
    log_time_per_step_frequency: int,
    log_time_per_step_percentiles: Tuple[int, ...],
    log_gradients_frequency: int,
    log_params_frequency: int,
    log_sample_losses_frequency: int,
    wandb_disable: bool,
    wandb_project: Optional[str],
    wandb_group: Optional[str],
    wandb_name: Optional[str],
    wandb_tags: Tuple[str, ...],
    data_buffer: int,
    event_buffer: int,
    detect_anomalies: bool,
) -> None:
    rng_key: Optional[PRNGKey]
    params: Optional[ArrayTree]
    opt_state: Optional[optax.MultiStepsState]
    if config_path is None and load_from is not None:
        # Either load a checkpoint...
        cp = nolo.load_from_directory(path=load_from)
        config = cp.config
        step = cp.step
        seed = cp.seed
        rng_key = cp.rng_key
        params = cp.params
        opt_state = cp.opt_state
        if wandb_disable:
            run = None
        else:
            try:
                run = nolo.load_wandb_run(path=load_from)
            except FileNotFoundError:
                run = nolo.new_wandb_run(
                    project=wandb_project,
                    tags=wandb_tags,
                    group=wandb_group,
                    name=wandb_name,
                    notes=pformat(config.to_dict()),
                )
    elif config_path is not None and load_from is None:
        # ...or create a new one
        if seed is None:
            raise ValueError("Seed must be specified if loading from config")
        config = nolo.Config.from_yaml(config_path)
        rng_key = params = opt_state = None
        step = 0
        run = (
            None
            if wandb_disable
            else nolo.new_wandb_run(
                project=wandb_project,
                tags=wandb_tags,
                group=wandb_group,
                name=wandb_name,
                notes=pformat(config.to_dict()),
            )
        )
    else:
        raise ValueError("Must specify either config-path or load-from")

    batch_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=data_buffer)
    trainer_event_queue: queue.Queue[nolo.Event] = queue.Queue(maxsize=event_buffer)
    trainer_termination_event = threading.Event()
    batch_termination_event = threading.Event()

    def batches_fn() -> Iterator[np.ndarray]:
        assert isinstance(seed, int)
        batches = nolo.batches_from_config(config, seed)
        # TODO: Make this compatible with gradient accumulation
        return islice(batches, step, None)

    batch_queue_thread = nolo.IteratorAsQueue(
        iterator_fn=batches_fn,
        queue=batch_queue,
        termination_event=batch_termination_event,
    )

    def sidecar_fn() -> None:
        events = nolo.queue_as_iterator(trainer_event_queue)
        events = nolo.log_time_per_step(
            events=events,
            frequency=log_time_per_step_frequency,
            percentiles=log_time_per_step_percentiles,
        )
        events = nolo.accumulate_gac_steps(events=events)
        events = nolo.log_losses(events=events, frequency=log_frequency)
        events = (
            nolo.log_to_wandb(events=events, run=run) if run is not None else events
        )
        if detect_anomalies:
            events = nolo.detect_anomalies(events=events)
        events = nolo.save_to_directory(events=events)
        for event in events:
            del event

    sidecar_thread = nolo.ReraisingThread(target=sidecar_fn)

    trainer_thread = nolo.Trainer(
        batch_queue=batch_queue,
        event_queue=trainer_event_queue,
        config=config,
        seed=seed,
        termination_event=trainer_termination_event,
        rng_key=rng_key,
        params=params,
        opt_state=opt_state,
        step=step,
        save_frequency=save_frequency,
        save_directory=save_directory,
        log_gradients_frequency=log_gradients_frequency,
        log_params_frequency=log_params_frequency,
        log_sample_losses_frequency=log_sample_losses_frequency,
    )

    with batch_queue_thread, sidecar_thread, trainer_thread:
        try:
            batch_queue_thread.join()
            trainer_thread.join()
            sidecar_thread.join()
        except KeyboardInterrupt:
            try:
                logger.info("Keyboard interrupt received. Graceful shutdown...")
                batch_termination_event.set()
                batch_queue_thread.join()
                logger.info("Batch queue thread joined.")
                trainer_thread.save_and_terminate().emit_end_of_training_event()
                logger.info("Trainer thread joined.")
                sidecar_thread.join()
                logger.info("Sidecar thread joined.")
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received again. Hard shutdown...")
                batch_termination_event.set()
                trainer_termination_event.set()
        finally:
            logger.info("Training ended.")


if __name__ == "__main__":
    cli()
