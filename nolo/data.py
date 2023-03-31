import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    TypeVar,
    Union,
)

import datasets
import numpy as np
import tokenizers
from tokenizers import Tokenizer

from .common import Config, get_logger

T = TypeVar("T")
logger = get_logger()


RNGLike = Union[int, np.random.Generator]


def _to_rng(rng_like: RNGLike) -> np.random.Generator:
    return np.random.default_rng(rng_like) if isinstance(rng_like, int) else rng_like


def _stream_huggingface_dataset(
    *args: str,
    **kwargs: str,
) -> Iterable[Dict[str, Any]]:
    dataset = datasets.load_dataset(*args, **kwargs, streaming=True)
    return (dict(sample) for sample in dataset)


def _load_huggingdace_tokenizer(
    *args: str,
    **kwargs: str,
) -> Tokenizer:
    if len(args) == 1 and args[0].strip().lower().endswith(".json"):
        logger.info(f"Loading tokenizer from JSON file: {args[0]}")
        return Tokenizer.from_file(*args, **kwargs)
    logger.info(f"Loading tokenizer from HuggingFace: {args}, {kwargs}")
    return tokenizers.Tokenizer.from_pretrained(*args, **kwargs)


def load_huggingface_dataset(
    args: Iterable[str],
    kwargs: Mapping[str, str],
    load_dataset_fn: Callable[
        ..., Iterable[Dict[str, Any]]
    ] = _stream_huggingface_dataset,
    repeat_forever: bool = False,
) -> Iterable[str]:
    if isinstance(args, str):
        raise TypeError(f"Expected args to be a sequence of str, got {args}")
    args = list(args)
    kwargs = dict(kwargs)
    key = kwargs.pop("key", "text")
    for epoch in itertools.count():
        logger.info(
            f"Streaming dataset from HuggingFace: {args}, {kwargs} (epoch: {epoch})"
        )
        dataset = load_dataset_fn(*args, **kwargs)
        samples = (sample[key] for sample in dataset)
        samples = (sample for sample in samples if sample.strip())
        yield from samples
        if not repeat_forever:
            break


def merge_datasets(
    datasets: Iterable[Iterable[T]],
    weights: Iterable[float],
    rng: RNGLike,
) -> Iterable[T]:
    p = np.array(list(weights), dtype=float)
    p = p / p.sum()
    rng = _to_rng(rng)
    iterators = [iter(dataset) for dataset in datasets]
    while True:
        index = rng.choice(len(iterators), p=p)
        yield next(iterators[index])


def tokenizer_from_config(
    config: Config,
) -> Tokenizer:
    return load_huggingface_tokenizer(
        config.tokenizer.args,
        config.tokenizer.kwargs,
    )


def load_huggingface_tokenizer(
    args: Iterable[str],
    kwargs: Mapping[str, str],
    load_tokenizer_fn: Callable[..., Tokenizer] = _load_huggingdace_tokenizer,
) -> Tokenizer:
    return load_tokenizer_fn(*args, **kwargs)


def tokenize_samples(
    samples: Iterable[str],
    tokenizer: Tokenizer,
    batch_size: int = 1000,
) -> Iterable[Iterable[int]]:
    def tokenizer_fn(texts: Sequence[str]) -> Sequence[Sequence[int]]:
        return [enc.ids for enc in tokenizer.encode_batch(texts)]

    batched_indices = (tokenizer_fn(chunk) for chunk in chunks(samples, batch_size))
    indices = (index for batch in batched_indices for index in batch)
    yield from indices


def chain_and_split(
    arrays: Iterable[Iterable[T]],
    length: int,
) -> Iterable[Iterable[T]]:
    ids = (id for array in arrays for id in array)
    yield from chunks(ids, length)


def chunks(
    iterable: Iterable[T],
    size: int,
    drop_last: bool = False,
) -> Iterable[Sequence[T]]:
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, size))
        if not chunk:
            break
        if drop_last and len(chunk) < size:
            break
        yield chunk


def shuffle(
    xs: Iterable[T],
    buffer_size: int,
    rng: RNGLike,
) -> Iterable[T]:
    if buffer_size <= 0:
        raise ValueError(f"Expected buffer_size > 0, got {buffer_size}")
    rng = _to_rng(rng)
    buffer: List[T] = list(itertools.islice(xs, buffer_size))
    for x in xs:
        index = rng.integers(len(buffer))
        buffer[index], x = x, buffer[index]
        yield x
    yield from rng.permutation(np.array(buffer), 0)


def batches_from_config(
    config: Config,
    rng: Union[int, np.random.Generator],
    extra_length: int = 0,
) -> Iterable[np.ndarray]:
    rng = _to_rng(rng)
    datasets = (
        load_huggingface_dataset(
            args=ds.args,
            kwargs=ds.kwargs,
            repeat_forever=True,
        )
        for ds in config.dataset
    )
    # Shuffle the samples of each dataset. This prevents the samples from
    # low-frequency datasets to be almost sequential due to a too-small buffer
    # size for the final shuffle.
    datasets = (
        shuffle(
            xs=ds,
            buffer_size=config.data.per_dataset_shuffle_buffer_size,
            rng=rng,
        )
        for ds in datasets
    )
    weights = (ds.weight for ds in config.dataset)
    dataset = merge_datasets(datasets=datasets, weights=weights, rng=rng)
    tokenizer = load_huggingface_tokenizer(
        args=config.tokenizer.args,
        kwargs=config.tokenizer.kwargs,
    )
    indices = tokenize_samples(dataset, tokenizer)
    indices = chain_and_split(arrays=indices, length=config.data.length + extra_length)
    # Shuffle the individual chunks
    indices = shuffle(
        xs=indices,
        buffer_size=config.data.shuffle_buffer_size,
        rng=rng,
    )
    batches = chunks(indices, size=config.data.batch_size)
    arrays = (np.array(batch, dtype=np.int32) for batch in batches)
    return arrays
