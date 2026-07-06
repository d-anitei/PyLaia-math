from typing import Callable, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer that also returns self-attention weights."""

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._resolve_activation(activation)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    @staticmethod
    def _resolve_activation(
        activation: Union[str, Callable[[Tensor], Tensor]]
    ) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if isinstance(activation, str):
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")
        return activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = src
        if self.norm_first:
            sa_out, attn_weights = self._sa_block(
                self.norm1(x),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            x = x + sa_out
            x = x + self._ff_block(self.norm2(x))
            return x, attn_weights

        sa_out, attn_weights = self._sa_block(
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = self.norm1(x + sa_out)
        x = self.norm2(x + self._ff_block(x))
        return x, attn_weights

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return self.dropout1(x), attn_weights

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
