#!/usr/bin/env python3
from pathlib import Path
from functools import partial
from typing import Optional, Iterator, Callable
import click
import logging

import nolo

logger = logging.getLogger(__name__)


@click.group('NoLo')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
def cli(debug: bool,
        ) -> None:
    nolo.setup_logging(log_level="INFO", log_to_stdout=True, logfile=None)
    nolo.set_debug(debug)


@cli.command('samples')
@click.option('--config', '-c', type=Path, required=True)
@click.option('--number', '-n', type=int, default=10)
@click.option('--seed', '-s', type=int, default=None)
def cli_show_samples(config: Path,
                     number: int,
                     seed: Optional[int],
                     ) -> None:
    cfg = nolo.Config.from_yaml(config)
    assert isinstance(cfg, nolo.DataConfig)
    batches = nolo.get_batches(cfg, seed=seed or 0)
    tokenizer = nolo.get_tokenizer(cfg)
    samples = (sample for batch in batches for sample in batch)
    for _ in range(number):
        click.echo(click.style(" ".join(map(str, next(samples))), fg="blue"))
        click.echo(click.style(tokenizer.decode(next(samples), skip_special_tokens=False), fg="green"))
        click.echo()


@cli.command('tokenizer')
@click.option('--config', '-c', type=Path, required=True)
def cli_tokenizer(config: Path) -> None:
    cfg = nolo.Config.from_yaml(config)
    assert isinstance(cfg, nolo.DataConfig)
    nolo.create_tokenizer(cfg)


def _add_sidecar_processors(train_fn: Callable[[], Iterator[nolo.Telemetry]],
                            out: Optional[Path],
                            ) -> Iterator[nolo.Telemetry]:
    # Buffer the telemetry stream to mitigate the impact of slow I/O etc. in
    # downstream processors.
    telemetry = nolo.buffer(train_fn, 10)
    telemetry = nolo.log_to_stderr(telemetry)
    if out is None:
        logger.warning("No output path specified, checkpoints will not be saved")
    else:
        telemetry = nolo.save_checkpoints(telemetry, out)
        telemetry = nolo.log_to_csv(telemetry, out / "telemetry.csv")
        telemetry = nolo.plot_from_csv(telemetry,
                                       out / "telemetry.csv",
                                       out / "telemetry.png")
    return telemetry


@cli.command('train')
@click.option('--config', '-c', type=Path, required=True)
@click.option('--seed', '-s', type=int, required=True)
@click.option('--out', '-o', type=Path, default=None)
def cli_train(config: Path,
              seed: int,
              out: Optional[Path],
              ) -> None:
    cfg = nolo.Config.from_yaml(config)
    assert isinstance(cfg, nolo.TrainingConfig)
    train_fn = partial(nolo.train_new, cfg, seed=seed)
    telemetry = _add_sidecar_processors(train_fn, out)
    for _ in telemetry:
        pass


@cli.command('resume')
@click.option('--checkpoint', '-c', type=Path, required=True)
@click.option('--out', '-o', type=Path, default=None)
def cli_resume(checkpoint: Path,
               out: Optional[Path],
               ) -> None:
    train_fn = partial(nolo.train_from_checkpoint, checkpoint)
    telemetry = _add_sidecar_processors(train_fn, out)
    for _ in telemetry:
        pass


@cli.command('sample')
@click.option('--checkpoint', '-c', type=Path, required=True)
@click.option('--steps', type=int, default=100,
              help='Number of diffusion steps to take')
@click.option('--sequence-length', type=int, default=None,
              help='Length of the generated sequence')
@click.option('--seed', '-s', type=int, required=True)
def cli_sample(checkpoint: Path,
               steps: int,
               sequence_length: Optional[int],
               seed: int,
               ) -> None:
    cp = nolo.load_checkpoint(checkpoint, for_inference=True)
    params = cp["params"]
    config = cp["config"]
    it = nolo.sample(params, config, steps=steps, sequence_length=sequence_length, seed=seed)
    for xt, x0, text in it:
        click.echo(text)
        msg = (f"xt: {xt.mean():.4f} +/- {xt.std():.4f}"
               f"\nx0: {x0.mean():.4f} +/- {x0.std():.4f}")
        click.echo(click.style(msg, fg="blue"))
        click.echo()


if __name__ == "__main__":
    cli()
