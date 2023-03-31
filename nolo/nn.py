from __future__ import annotations

from functools import partial, wraps
from typing import Callable, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
from chex import ArrayTree, PRNGKey
from einops import rearrange, repeat
from jax import Array

from .common import Config, expand_dims, get_logger, scaled_l2_norm

_SMALL_INIT = hk.initializers.VarianceScaling(0.02)

logger = get_logger()


def full_precision(fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    @wraps(fn)
    def inner(x: Array) -> Array:
        return fn(x.astype(jnp.float32)).astype(x.dtype)

    return inner


def rotary_pos_emb(
    x: Array,  # B H S D
) -> Array:
    dim, seq = x.shape[-1], x.shape[-2]
    # Near eq. 15 in https://arxiv.org/abs/2104.09864, equivalent to those
    # in https://arxiv.org/abs/1706.03762
    ts = jnp.arange(0, dim, 2, dtype=jnp.float32)  # D/2
    inv_freqs = 10_000 ** (-ts / dim)  # D/2
    grid = jnp.einsum("s, d -> s d", jnp.arange(seq), inv_freqs)  # S D/2
    # Eq. 34 in https://arxiv.org/abs/2104.09864
    sin_embs = repeat(jnp.sin(grid), "s d -> 1 s (d 2)")  # B S D
    sin_embs = sin_embs.astype(x.dtype)
    cos_embs = repeat(jnp.cos(grid), "s d -> 1 s (d 2)")  # B S D
    cos_embs = cos_embs.astype(x.dtype)
    # Pairwise swap with alternating signs
    x1, x2 = x[..., ::2], x[..., 1::2]  # [x1, x3, x5, ...], [x2, x4, x6, ...]
    x1x2 = jnp.stack([-x2, x1], axis=-1)  # [[-x2, x1], [-x4, x3], ...]
    xs = rearrange(x1x2, "... d two -> ... (d two)", two=2)  # [-x2, x1, -x4, x3, ...]
    out = x * cos_embs + xs * sin_embs
    return out


class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        pos_emb_portion: float,
        name: str,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.pos_emb_portion = pos_emb_portion

    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
    ) -> Array:
        # Constants
        D, H = x.shape[-1], self.num_heads
        if D % H != 0:
            raise ValueError(f"Dimension {D} must be divisible by number of heads {H}")
        K = D // H
        # Projections
        projection = partial(hk.Linear, with_bias=False)
        q_proj = projection(K * H, name="q_proj")
        k_proj = projection(K * H, name="k_proj")
        v_proj = projection(K * H, name="v_proj")
        o_proj = projection(D, name="o_proj", w_init=_SMALL_INIT)
        # Q, K, V
        p = int(K * self.pos_emb_portion)
        q: Array = q_proj(x) / K**0.5  # B L H K
        q = rearrange(q, "b l (h k) -> b h l k", h=H)
        q = jnp.concatenate([rotary_pos_emb(q[..., :p]), q[..., p:]], axis=-1)
        k: Array = k_proj(x)  # B L H K
        k = rearrange(k, "b l (h k) -> b h l k", h=H)
        k = jnp.concatenate([rotary_pos_emb(k[..., :p]), k[..., p:]], axis=-1)
        v: Array = v_proj(x)  # B L H V
        v = rearrange(v, "b l (h v) -> b h l v", h=H)
        # Attention weights
        l: Array = jnp.einsum("b h i k, b h j k -> b h i j", q, k)  # B H L L
        _apply_mask = lambda l_, m_: (l_ if m_ is None else jnp.where(m_, l_, -jnp.inf))
        with jax.debug_infs(False):
            l = hk.remat(_apply_mask)(l, mask)
            a = full_precision(jax.nn.softmax)(l)  # B H L L
        # Attention output
        y: Array = jnp.einsum("b h i j, b h j v -> b h i v", a, v)  # B H L V
        y = rearrange(y, "b h l v -> b l (h v)")  # B L (H V)
        o = o_proj(y)  # B L M
        return o


class FeedForward(hk.Module):
    def __init__(
        self,
        hidden_dim: int,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.hidden_dim = hidden_dim

    def __call__(
        self,
        x: Array,
    ) -> Array:
        model_dim = x.shape[-1]
        # Projections
        projection = partial(hk.Linear, with_bias=False)
        in_proj = projection(self.hidden_dim, name="in_proj")
        out_proj = projection(model_dim, name="out_proj", w_init=_SMALL_INIT)
        # Feed-forward
        y = in_proj(x)
        y = hk.remat(jax.nn.gelu)(y)
        y = out_proj(y)
        return y


class Block(hk.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        hidden_dim: int,
        pos_emb_portion: float,
        dropout: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.pos_emb_portion = pos_emb_portion
        self.dropout = dropout

    def __call__(
        self,
        x: Array,
        is_training: bool,
        mask: Optional[Array] = None,
    ) -> Array:
        mha = MultiHeadAttention(
            num_heads=self.num_heads, pos_emb_portion=self.pos_emb_portion, name="mha"
        )
        mha_ln = hk.remat(hk.RMSNorm(-1, name="mha_norm"))
        ff = FeedForward(self.hidden_dim, name="ff")
        ff_ln = hk.remat(hk.RMSNorm(-1, name="ff_norm"))
        # Multi-head attention
        y = mha(mha_ln(x), mask)
        y = hk.dropout(hk.next_rng_key(), self.dropout, y) if is_training else y
        x = x + y
        # Feed-forward
        z = ff(ff_ln(x))
        z = hk.dropout(hk.next_rng_key(), self.dropout, z) if is_training else z
        out = x + z
        return out


def sine_encoding(
    ts: Array,
    dims: int,
    dtype: jnp.dtype,
) -> Array:
    if not dims % 2 == 0:
        raise ValueError(f"Expected even number of dimensions, got {dims}")
    freqs = jnp.pi * 2 ** jnp.arange(dims // 2, dtype=jnp.float32)
    xs = [jnp.sin(ts[..., None] * freqs), jnp.cos(ts[..., None] * freqs)]
    out = jnp.concatenate(xs, axis=-1)
    return out.astype(dtype)


class Model(hk.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        vocabulary_size: int,
        model_dim: int,
        num_heads: int,
        pos_emb_portion: float,
        t_emb_dim: int,
        hidden_dim: int,
        dropout: float,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.num_layers = num_layers
        self.vocabulary_size = vocabulary_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.pos_emb_portion = pos_emb_portion
        self.t_emb_dim = t_emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    @classmethod
    def from_config(cls, config: Config) -> Model:
        cfg = config.model
        return cls(
            num_layers=int(cfg.num_layers),
            vocabulary_size=int(cfg.vocabulary_size),
            model_dim=int(cfg.model_dim),
            num_heads=int(cfg.num_heads),
            pos_emb_portion=float(cfg.pos_emb_portion),
            t_emb_dim=int(cfg.t_emb_dim),
            hidden_dim=int(cfg.hidden_dim),
            dropout=float(cfg.dropout),
        )

    @property
    def embedding(self) -> hk.Embed:
        return hk.Embed(
            self.vocabulary_size,
            self.model_dim,
            name="embedding",
        )

    @property
    def embeddings(self) -> Array:
        return self.embedding.embeddings

    def embed(
        self,
        indices: Array,
    ) -> Array:
        return scaled_l2_norm(self.embedding(indices))

    def __call__(
        self,
        x: Array,
        t: Array,
        is_training: bool,
        mask: Optional[Array] = None,
    ) -> Array:
        t_proj = hk.Linear(
            self.model_dim,
            with_bias=False,
            w_init=_SMALL_INIT,
            name="t_proj",
        )
        blocks = [
            Block(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                pos_emb_portion=self.pos_emb_portion,
                dropout=self.dropout,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]
        out_ln = hk.RMSNorm(-1, name="out_norm")
        logit_proj = hk.Linear(
            self.vocabulary_size,
            with_bias=False,
            w_init=_SMALL_INIT,
            name="logit_proj",
        )
        # Execution
        t = expand_dims(t, len(x.shape) - 1)
        te = sine_encoding(t, self.t_emb_dim, x.dtype)
        x = x + t_proj(te)
        for block in blocks:
            x = block(x, is_training, mask)
        return logit_proj(out_ln(x))

    @classmethod
    def get_params(
        cls,
        config: Config,
        rng_or_seed: Union[int, PRNGKey],
        log_size: bool = True,
    ) -> ArrayTree:
        def fn() -> None:
            model = cls.from_config(config)
            indices = jnp.zeros((1, 1), dtype=jnp.int32)
            ts = jnp.zeros((1,))
            xs = model.embed(indices)
            model(xs, ts, False)

        rng = (
            jax.random.PRNGKey(rng_or_seed)
            if isinstance(rng_or_seed, int)
            else rng_or_seed
        )
        params = hk.transform(fn).init(rng)
        if log_size:
            params_n = hk.data_structures.tree_size(params)
            params_mb = round(hk.data_structures.tree_bytes(params) / 1e6, 2)
            logger.info(f"Model parameters: {params_n:,} ({params_mb:.2f} MB)")
        return params
