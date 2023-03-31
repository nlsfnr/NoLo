#!/usr/bin/env python3
import sys
from pathlib import Path

import click
from jax import Array

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
@cli.command("gen")
@click.option("--load-from", "-l", type=Path, required=True,
              help="Path to the checkpoint to load from.")
@click.option("--length", "-n", type=int, default=32,
              help="Length of the generated text. Default 32.")
@click.option("--steps", "-i", type=int, default=128,
              help="Number of diffusion steps to run the model. Default 128.")
@click.option("--seed", "-s", type=int, default=0,
              help="Random seed. Default 0.")
# fmt: on
def cli_generate(
    load_from: Path,
    length: int,
    steps: int,
    seed: int,
) -> None:
    cp = nolo.load_from_directory_for_inference(path=load_from)
    config, params = cp.config, cp.params
    it = nolo.sample(
        params=params,
        config=config,
        seed=seed,
        steps=steps,
        sequence_length=length,
    )
    for i, rv in enumerate(it):
        print(i, rv.text)


# fmt: off
@cli.command("dbg")
@click.option("--load-from", "-l", type=Path, required=True,
              help="Path to the checkpoint to load from.")
@click.option("--text", "-t", type=str, required=True,
              help="Text to use as input.")
@click.option("--steps", "-i", type=int, default=128,
              help="Number of diffusion steps to run the model. Default 128.")
@click.option("--seed", "-s", type=int, default=0,
              help="Random seed. Default 0.")
# fmt: on
def cli_dbg(
    load_from: Path,
    text: int,
    steps: int,
    seed: int,
) -> None:
    del text
    cp = nolo.load_from_directory_for_inference(path=load_from)
    config, params = cp.config, cp.params
    it = nolo.sample(
        params=params,
        config=config,
        seed=seed,
        steps=steps,
    )

    def _print_stats(x: Array, name: str) -> None:
        print(
            f"name {name:<10}, min {x.min():.3f}, max {x.max():.3f}, "
            f"mean {x.mean():.3f}, std {x.std():.3f}"
        )

    for i, rv in enumerate(it):
        print(i, rv.text)
        _print_stats(rv.x0_pred, "x0_pred")
        _print_stats(rv.noise_pred, "noise_pred")
        _print_stats(rv.next_xt, "next_xt")
        print(f"alpha: {rv.alpha:.3f}, next_alpha: {rv.next_alpha:.3f}")


if __name__ == "__main__":
    cli()
