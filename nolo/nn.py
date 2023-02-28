from __future__ import annotations

from functools import partial
from typing import List, Optional, Protocol, runtime_checkable

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array
from einops import rearrange, repeat


@runtime_checkable
class ModelConfig(Protocol):
    vocab_size: int
    n_layers: int
    model_dim: int
    n_heads: int
    t_mlp_layers: List[int]
    dropout: float
    mlp_size: int


def rotary_pos_emb(
    x: Array,  # B H S D
) -> Array:
    dim = x.shape[-1]
    seq = x.shape[-2]
    # Near eq. 15 in https://arxiv.org/abs/2104.09864, equivalent to those
    # in https://arxiv.org/abs/1706.03762
    ts = jnp.arange(0, dim, 2, dtype=jnp.float32)  # D/2
    inv_freqs = 10_000 ** (-ts / dim)  # D/2
    grid = jnp.einsum("s, d -> s d", jnp.arange(seq), inv_freqs)  # S D/2
    # Eq. 34 in https://arxiv.org/abs/2104.09864
    sin_embs = repeat(jnp.sin(grid), "s d -> 1 s (d 2)")  # B S D
    cos_embs = repeat(jnp.cos(grid), "s d -> 1 s (d 2)")  # B S D
    # Pairwise swap with alternating signs
    x1, x2 = x[..., ::2], x[..., 1::2]  # [x1, x3, x5, ...], [x2, x4, x6, ...]
    x1x2 = jnp.stack([-x2, x1], axis=-1)  # [[-x2, x1], [-x4, x3], ...]
    xs = rearrange(x1x2, "... d two -> ... (d two)", two=2)  # [-x2, x1, -x4, x3, ...]
    out = x * cos_embs + xs * sin_embs
    return out


class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        n_heads: int,
        model_size: int,
        dropout: float,
        use_rotary: bool,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = n_heads
        self.model_size = model_size
        self.dropout = dropout
        self.key_size = model_size // n_heads
        self.use_rotary = use_rotary

    def __call__(
        self,
        x: Array,  # B L V
        is_training: bool,
    ) -> Array:
        chex.assert_rank(x, 3)
        # Projections
        projection = partial(hk.Linear, with_bias=False)
        q_proj = projection(self.model_size, name="q_proj")
        k_proj = projection(self.model_size, name="k_proj")
        v_proj = projection(self.model_size, name="v_proj")
        o_proj = projection(self.model_size, name="o_proj")
        # Q, K, V
        q = q_proj(x) / x.shape[-1] ** 0.5  # B L H K
        q = rearrange(q, "b l (h k) -> b h l k", h=self.num_heads)
        k = k_proj(x)  # B L H K
        k = rearrange(k, "b l (h k) -> b h l k", h=self.num_heads)
        v = v_proj(x)  # B L H V
        v = rearrange(v, "b l (h v) -> b h l v", h=self.num_heads)
        if self.use_rotary:
            q = rotary_pos_emb(q)
            k = rotary_pos_emb(k)
        # Attention weights
        l: Array = jnp.einsum("b h i k, b h j k -> b h i j", q, k)  # B H L L
        if is_training:
            l = hk.dropout(hk.next_rng_key(), self.dropout, l)
        a = jax.nn.softmax(l, axis=-1)  # B H L L
        # Attention output
        y = jnp.einsum("b h i j, b h j v -> b h i v", a, v)  # B H L V
        y = rearrange(y, "b h l v -> b l (h v)")  # B L (H V)
        return o_proj(y)  # B L M


class AffineConditioning(hk.Module):
    def __init__(
        self,
        dropout: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.dropout = dropout

    def __call__(
        self,
        x: Array,  # B S D
        t_emb: Array,  # B S E
        is_training: bool,
    ) -> Array:
        n_channels = x.shape[-1]
        scale = hk.Linear(n_channels, with_bias=False, name="scale")(t_emb)  # B S D
        offset = hk.Linear(n_channels, with_bias=False, name="offset")(t_emb)  # B S D
        if is_training:
            scale = hk.dropout(hk.next_rng_key(), self.dropout, scale)
            offset = hk.dropout(hk.next_rng_key(), self.dropout, offset)
        x = hk.LayerNorm(-1, False, False)(x)  # B S D
        x = x * scale + offset  # B S D
        return x


class EncoderBlock(hk.Module):
    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        mlp_size: int,
        dropout: float,
        use_rotary: bool,
        use_affine: bool,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.mlp_size = mlp_size
        self.dropout = dropout
        self.use_rotary = use_rotary
        self.use_affine = use_affine

    def __call__(
        self,
        x: Array,
        t_emb: Array,
        is_training: bool,
    ) -> Array:
        mha_ln = (
            AffineConditioning(self.dropout, name="affine")
            if self.use_affine
            else hk.LayerNorm(-1, True, False, name="mha_ln")
        )
        mha = MultiHeadAttention(
            n_heads=self.n_heads,
            model_size=self.model_dim,
            dropout=self.dropout,
            use_rotary=self.use_rotary,
            name="mha",
        )
        mlp_ln = hk.LayerNorm(-1, True, False, name="mlp_ln")
        mlp = hk.Sequential(
            [
                hk.Linear(self.mlp_size, with_bias=False),
                jax.nn.gelu,
                hk.Linear(self.model_dim, with_bias=False),
            ]
        )
        # Multi-head attention
        h = (
            mha_ln(x)
            if isinstance(mha_ln, hk.LayerNorm)
            else mha_ln(x, t_emb, is_training)  # type: ignore
        )
        h = mha(h, is_training=is_training)
        if is_training:
            h = hk.dropout(hk.next_rng_key(), self.dropout, h)
        x = x + h
        # Multi-layer perceptron
        h = mlp_ln(x)
        h = mlp(h)
        if is_training:
            h = hk.dropout(hk.next_rng_key(), self.dropout, h)
        x = x + h
        return x


class Model(hk.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        model_dim: int,
        n_heads: int,
        t_mlp_layers: List[int],
        dropout: float = 0.1,
        mlp_size: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.n_layers = n_layers
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.t_mlp_layers = t_mlp_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.mlp_size = mlp_size

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        name: Optional[str] = None,
    ) -> Model:
        assert isinstance(config, ModelConfig)
        return cls(
            vocab_size=config.vocab_size,
            n_layers=config.n_layers,
            model_dim=config.model_dim,
            n_heads=config.n_heads,
            t_mlp_layers=config.t_mlp_layers,
            dropout=config.dropout,
            mlp_size=config.mlp_size,
            name=name,
        )

    @property
    @hk.transparent
    def embeddings(self):
        raw_embeddings = hk.get_parameter(
            "embeddings",
            [self.vocab_size, self.model_dim],
            init=hk.initializers.RandomNormal(),
        )
        embeddings = raw_embeddings / jnp.linalg.norm(
            raw_embeddings, axis=-1, keepdims=True
        )
        return embeddings

    def embed(
        self,
        indices: Array,
    ) -> Array:
        return jnp.take(self.embeddings, indices, axis=0)

    def __call__(
        self,
        x: Array,
        t: Array,
        is_training: bool,
    ) -> Array:
        assert x.ndim == 3  # B S D
        assert t.ndim == 2  # B S
        # T Embedding
        t_emb_dim, *t_mlp_layers = self.t_mlp_layers
        frequencies = 0.5 * jnp.pi * 2.0 ** jnp.arange(0, t_emb_dim, dtype=jnp.float32)
        t_emb = jnp.sin(jnp.einsum("b s, d -> b s d", t, frequencies))  # B S D
        for i, mlp_size in enumerate(t_mlp_layers):
            t_emb = hk.Linear(mlp_size, name=f"t_mlp_{i}")(t_emb)  # S D
            t_emb = jax.nn.gelu(t_emb)
        # Transformer
        for i in range(self.n_layers):
            block = EncoderBlock(
                model_dim=self.model_dim,
                n_heads=self.n_heads,
                mlp_size=self.mlp_size or self.model_dim * 4,
                dropout=self.dropout,
                use_rotary=i == 0,
                use_affine=i == 0,
                name=f"encoder_block_{i}",
            )
            x = block(x, t_emb, is_training=is_training)
        # TODO: Can we share parameters between embeddings and the out projection?
        x = hk.LayerNorm(-1, True, False, name="out_ln")(x)
        logits = hk.Linear(self.vocab_size, name="out_proj")(x)
        return logits
