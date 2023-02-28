from __future__ import annotations

import itertools
import logging
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Iterable, Iterator, List, Optional, TypeVar, Union

import yaml

T = TypeVar("T")


def buffer(it: Iterator[T], size: int) -> Iterator[T]:
    """Buffer an iterator into a queue of size `size`."""
    queue: Queue[Union[T, object]] = Queue(maxsize=size)
    sentinel = object()

    def producer() -> None:
        for item in it:
            queue.put(item)
        queue.put(sentinel)

    thread = threading.Thread(target=producer)
    thread.start()
    try:
        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item  # type: ignore
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
    handlers: List[logging.Handler] = []
    handlers.append(logging.StreamHandler(sys.stdout if log_to_stdout else sys.stderr))
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s|%(name)s|%(levelname)s] %(message)s",
        handlers=handlers,
    )
