from __future__ import annotations

import inspect
import itertools
import logging
import sys
import threading
from pathlib import Path
from queue import Full, Queue
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import yaml
from chex import Array

logger = logging.getLogger(__name__)


T = TypeVar("T")


def set_debug(debug: bool) -> None:
    """Set debug mode for JAX."""
    jax.config.update("jax_debug_nans", debug)
    jax.config.update("jax_debug_infs", debug)
    jax.config.update("jax_disable_jit", debug)
    if debug:
        logger.warn("Running in debug mode")


def buffer(fn: Callable[[], Iterator[T]], size: int) -> Iterator[T]:
    """Buffer an iterator into a queue of size `size`."""
    queue: Queue[Union[T, object]] = Queue(maxsize=size)
    sentinel = object()
    terminate = threading.Event()

    def producer() -> None:
        for item in fn():
            try:
                queue.put(item, timeout=0.1)
            except Full:
                pass
            if terminate.is_set():
                return
        queue.put(sentinel)

    thread = threading.Thread(target=producer)
    try:
        thread.start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item  # type: ignore
    except Exception:
        terminate.set()
        raise
    finally:
        thread.join()


def chunks(it: Iterable[T], size: int) -> Iterator[List[T]]:
    """Yield successive n-sized chunks from iterable."""
    iterator = iter(it)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


class Config(Dict[str, Any]):
    """A dictionary with syntax similar to that of JavaScript objects. I.e.
    instead of d['my_key'], we can simply say d.my_key."""

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            if key not in self:
                raise e
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


def setup_logging(
    log_level: str,
    log_to_stdout: bool,
    logfile: Optional[Path],
) -> None:
    """Setup logging to stdout and/or a file."""
    handlers: List[logging.Handler] = []
    handlers.append(logging.StreamHandler(sys.stdout if log_to_stdout else sys.stderr))
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s|%(name)s|%(levelname)s] %(message)s",
        handlers=handlers,
    )


def expand_dims(x: Array, reference: Union[int, Array]) -> Array:
    """Expand the dimensions of `x` to match the dimensions of `reference`."""
    if isinstance(reference, Array):
        reference = reference.ndim
    while x.ndim < reference:
        x = x[..., None]
    return x


def l2_norm(x: Array, axis: int = -1) -> Array:
    return x / jnp.linalg.norm(x, axis=axis, keepdims=True)


def plot(
    x: Array,
    title: Optional[str] = None,
    out_dir: Path = Path("plots/"),
) -> Array:
    """Plot a {1, 2, 3}D array."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if title is None:
        title = get_callstack_name()
    old_x = x
    if not x.ndim in (1, 2, 3):
        raise ValueError(f"Can't plot {x.ndim}-dimensional array")
    if x.ndim == 3:
        if x.shape[0] == 1:
            # Remove batch dimension
            x = x[0]
        elif x.shape[2] not in (1, 3):
            # Otherwise, either grey-scale or RGB
            raise ValueError(f"Can't plot {x.shape[2]}-channel image")
    if x.ndim == 1:
        plt.plot(x)
        plt.grid()
    else:
        plt.imshow(x, cmap="gray" if x.ndim == 2 else None)
        plt.axis("off")
    enhanced_title = f"{title}\n{tuple(x.shape)} {x.mean():.6f} {x.std():.6f}"
    plt.title(enhanced_title)
    plt.tight_layout()
    plt.savefig(out_dir / f'{title.replace(" ", "-").replace("/", "-")}.png')
    plt.close()
    return old_x


def get_callstack_name(depth: int = 2) -> str:
    """Get the name of the function that called this function and the line number of the call."""
    caller = inspect.stack()[depth]
    return f"{caller.function}:{caller.lineno}"
