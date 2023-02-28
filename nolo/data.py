from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import datasets
import numpy as np
from torch.utils.data import DataLoader

import tokenizers
from tokenizers import Tokenizer  # type: ignore

from .common import chunks

logger = getLogger(__name__)


SPECIAL_TOKENS = (
    PAD_TOKEN := "<pad>",
    BOS_TOKEN := "<bos>",
    EOS_TOKEN := "<eos>",
)


# E.g. (['wikitext', 'wikitext-103-raw-v1], {'language': 'en'})
# Or  [[scientific_papers, pubmed], {split: train}, 'article']
HFDatasetIdent = Union[
    Tuple[List[str], Dict[str, str]], Tuple[List[str], Dict[str, str], str]
]


@runtime_checkable
class DataConfig(Protocol):
    # Datasets
    datasets: List[HFDatasetIdent]
    dataset_weights: List[float]
    # Batches and sequence
    batch_size: int
    sequence_length: int
    min_tokens_per_sequence: int
    # Tokenizer
    tokenizer_path: Path
    tokenizer_max_training_samples: Optional[int]
    tokenizer_min_token_frequency: int
    vocab_size: int


def load_dataset(
    ident: HFDatasetIdent,
    repeat_forever: bool = False,
) -> datasets.IterableDataset:
    logger.info(f"Streaming dataset {ident}")
    if ident == "dummy":
        return datasets.IterableDataset.from_generator(
            lambda: iter([{"text": "This is dummy text!"}] * 10)
        )
    else:
        args, kwargs, key = ident if len(ident) == 3 else (*ident, "text")
        dataset = datasets.load_dataset(*args, **kwargs, streaming=True)
        dataset = dataset.map(lambda x: dict(text=x[key]))
    if isinstance(dataset, dict):
        raise ValueError(
            f"Expected dataset, got dict. Maybe you forgot to specify the split?"
        )
    assert isinstance(dataset, datasets.IterableDataset)
    return dataset


def get_batches(
    config: DataConfig,
    seed: int,
    skip: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """Yield batches of tokenized sequences."""
    assert isinstance(config, DataConfig), "Config is missing required attributes"
    if not len(config.datasets) == len(config.dataset_weights):
        raise ValueError(
            "datasets and dataset_weights must be the same length, "
            f"got {len(config.datasets)} and {len(config.dataset_weights)}"
        )
    tokenizer = get_tokenizer(config)

    def fn(ident: HFDatasetIdent) -> datasets.IterableDataset:
        dataset = load_dataset(ident)
        # Tokenize samples
        dataset = dataset.map(
            lambda x: dict(
                input_ids=[enc.ids for enc in tokenizer.encode_batch(x["text"])]
            ),
            batched=True,
        )
        # Only keep the input ids
        dataset = dataset.select_columns("input_ids")
        # Split samples into chunks with max size `sequence_length`
        dataset = dataset.map(
            lambda x: dict(
                input_ids=list(
                    (
                        chunk
                        for sample in x["input_ids"]
                        for chunk in chunks(sample, config.sequence_length)
                    )
                )
            ),
            batched=True,
        )
        # Filter chunks that are too short
        dataset = dataset.filter(
            lambda x: len(x["input_ids"]) >= config.min_tokens_per_sequence
        )
        # Shuffle the chunks
        dataset = dataset.shuffle(generator=np.random.default_rng(seed))
        # Add padding
        pad_id = tokenizer.token_to_id(PAD_TOKEN)
        dataset = dataset.map(
            lambda x: dict(
                input_ids=np.pad(
                    x["input_ids"],
                    (0, config.sequence_length - len(x["input_ids"])),
                    constant_values=pad_id,
                )
            ),
        )
        return dataset

    weights = np.asarray(config.dataset_weights)
    weights = weights / weights.sum()
    dataset = datasets.interleave_datasets(
        [fn(ident) for ident in config.datasets], weights, seed=seed
    )
    if skip is not None:
        logger.info(f"Skipping {skip} batches")
        dataset = dataset.skip(skip)
    collate_fn = lambda x: np.stack([s["input_ids"] for s in x], axis=0)
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, collate_fn=collate_fn
    )
    return iter(dataloader)


def get_tokenizer(config: DataConfig) -> Tokenizer:
    assert isinstance(config, DataConfig), "Config is missing required attributes"
    tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
    assert isinstance(tokenizer, Tokenizer)
    return tokenizer


def create_tokenizer(config: DataConfig) -> Tokenizer:
    """Create a tokenizer from a config."""
    assert isinstance(config, DataConfig), "Config is missing required attributes"
    logger.info("Creating tokenizer")

    def fn(ident: HFDatasetIdent) -> datasets.IterableDataset:
        dataset = load_dataset(ident)
        dataset.map(lambda x: x["text"])
        return dataset

    weights = np.asarray(config.dataset_weights)
    weights = weights / weights.sum()
    dataset = datasets.interleave_datasets(
        [fn(ident) for ident in config.datasets],
        weights,
    )
    texts: Iterator[str] = (x["text"] for x in dataset)
    if config.tokenizer_max_training_samples is not None:
        texts = islice(texts, config.tokenizer_max_training_samples)
    tokenizer = tokenizers.SentencePieceBPETokenizer()  # type: ignore
    tokenizer.train_from_iterator(
        texts,
        special_tokens=list(SPECIAL_TOKENS),
        vocab_size=config.vocab_size,
        min_frequency=config.tokenizer_min_token_frequency,
    )
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(  # type: ignore
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[
            (BOS_TOKEN, tokenizer.token_to_id(BOS_TOKEN)),
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            (PAD_TOKEN, tokenizer.token_to_id(PAD_TOKEN)),
        ],
    )
    path = Path(config.tokenizer_path)
    logger.info(f"Saving tokenizer to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer
