#!/usr/bin/env python3
from pathlib import Path
from typing import Optional
import click

import nolo


@click.group('NoLo')
def cli() -> None:
    nolo.setup_logging(log_level="INFO", log_to_stdout=True, logfile=None)


@cli.command('samples')
@click.option('--config', '-c', type=click.Path(exists=True), required=True)
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
@click.option('--config', '-c', type=click.Path(exists=True), required=True)
def cli_tokenizer(config: Path) -> None:
    cfg = nolo.Config.from_yaml(config)
    assert isinstance(cfg, nolo.DataConfig)
    nolo.create_tokenizer(cfg)


if __name__ == "__main__":
    cli()
