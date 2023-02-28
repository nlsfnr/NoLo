# Noisy Lookahead

# Datasets

All datasets are streamed from the HuggingFace hub, i.e. they are never on disk at once.

```yaml
[
  [[wikitext, wikitext-103-v1], {split: train}],  # 0.7 GB
  [[c4, en], {split: train}],  # 305 GB
  # Bookcorpus: Samples are short, might not reach min_tokens_per_sequence
  [[bookcorpus], {split: train}],  # 5.7 GB
  [[scientific_papers, arxiv], {split: train}, 'article'],  # 11.5 GB
  [[scientific_papers, pubmed], {split: train}, 'article'],  # 6.7 GB
]
```
