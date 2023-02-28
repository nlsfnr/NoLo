from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from tokenizers import Tokenizer  # type: ignore

from . import data
from .common import Config


@pytest.fixture
def cfg(tmp_path: Path) -> data.DataConfig:
    cfg = Config(
        datasets=["dummy"],
        dataset_weights=[1.0],
        batch_size=2,
        sequence_length=8,
        min_tokens_per_sequence=2,
        tokenizer_path=tmp_path / "tokenizer.json",
        tokenizer_max_training_samples=None,
        tokenizer_min_token_frequency=1,
        vocab_size=100,
    )
    assert isinstance(cfg, data.DataConfig), "Config is missing required attributes"
    return cfg


@pytest.fixture
def tokenizer(cfg: data.DataConfig) -> Tokenizer:
    return data.create_tokenizer(cfg)


def test_get_batches(cfg: data.DataConfig, tokenizer: Tokenizer) -> None:
    del tokenizer
    batches = data.get_batches(cfg)
    assert isinstance(batches, Iterator)
    assert isinstance(next(batches), np.ndarray)
