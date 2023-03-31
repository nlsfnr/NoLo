from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import yaml
from jax import Array

__logger = logging.getLogger("NoLo")


def get_logger() -> logging.Logger:
    return __logger


logger = get_logger()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s|%(levelname)s|%(name)s] %(message)s",
    )


def set_debug(debug: bool) -> None:
    jax.config.update("jax_debug_nans", debug)
    jax.config.update("jax_debug_infs", debug)
    jax.config.update("jax_disable_jit", debug)
    if debug:
        logger.warn("Running in debug mode")


class Config(Dict[str, Any]):
    """A dictionary with syntax similar to that of JavaScript objects. I.e.
    instead of d['my_key'], we can simply say d.my_key."""

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError:
            if key not in self:
                raise KeyError(f"Key '{key}' not found in config {self}")
            return self[key]

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Config:
        return Config(**{k: cls._from_obj(v) for k, v in d.items()})

    @classmethod
    def _from_obj(cls, o: Any) -> Any:
        if isinstance(o, dict):
            return cls.from_dict(o)
        if isinstance(o, list):
            return [cls._from_obj(x) for x in o]
        return o

    def to_dict(self) -> Dict[str, Any]:
        def _to_obj(x: Any) -> Any:
            if isinstance(x, Config):
                return {k: _to_obj(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to_obj(i) for i in x]
            return x

        obj = _to_obj(self)
        assert isinstance(obj, dict)
        return obj

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        with open(path) as fh:
            d = dict(yaml.safe_load(fh))
        return cls.from_dict(d)

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as fh:
            yaml.dump(self.to_dict(), fh)


def scaled_l2_norm(x: Array, axis: int = -1) -> Array:
    d = x.shape[axis]
    return d**0.5 * x / jnp.linalg.norm(x, axis=axis, keepdims=True)


def expand_dims(x: Array, reference: Union[int, Array]) -> Array:
    if isinstance(reference, Array):
        reference = reference.ndim
    while x.ndim < reference:
        x = x[..., None]
    return x
