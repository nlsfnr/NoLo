#!/usr/bin/env python3
from pathlib import Path
from typing import Optional
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
    batches = nolo.get_batches(cfg, seed=seed)
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
    telemetry = nolo.train_new(cfg, seed=seed)
    telemetry = nolo.log(telemetry)
    if out is None:
        logger.warning("No output path specified, checkpoints will not be saved")
    else:
        telemetry = nolo.save_checkpoints(telemetry, out)
    for _ in telemetry:
        pass


@cli.command('resume')
@click.option('--checkpoint', '-c', type=Path, required=True)
@click.option('--out', '-o', type=Path, default=None)
def cli_resume(checkpoint: Path,
               out: Optional[Path],
               ) -> None:
    telemetry = nolo.train_from_checkpoint(checkpoint)
    telemetry = nolo.log(telemetry)
    if out is None:
        logger.warning("No output path specified, checkpoints will not be saved")
    else:
        telemetry = nolo.save_checkpoints(telemetry, out)
    for _ in telemetry:
        pass


if __name__ == "__main__":
    cli()
