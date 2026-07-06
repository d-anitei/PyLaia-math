import torch
import torch.nn as nn
from typing import Any, Optional, Callable, Union
import torch.nn.functional as F
import torch.nn.init as init

from torch import Tensor

import warnings
from torch.nn import MultiheadAttention
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm

import math

class FocusedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(FocusedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = Dropout(dropout)
            
    def split_heads(self, x):
        x = x.transpose(0,1)
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None, need_weights=True, average_attn_weights=True):             
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)        

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        T = 0.5
        attn_probs = F.softmax(attn_scores/T, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        output = self.W_o(self.combine_heads(attn_output)).transpose(0,1)        
        if average_attn_weights:
            attn_probs = attn_probs.mean(dim=1)

        output = self.dropout(output)

        if need_weights:
            return output, attn_probs
        return output

class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerDecoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerDecoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerDecoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerDecoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()        

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)

        # self.cross_attn_width = MultiheadAttention(d_model, nhead, dropout=dropout,
        #                                     bias=bias, batch_first=batch_first,
        #                                     **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.norm4 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        # self.dropout4 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = self._get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        # self._initialize_weights()        
        self.gradients = None

    def _initialize_weights(self):
        # Example initialization for multi-head attention weights
        for p in self.self_attn.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)  # Xavier uniform initialization

        for p in self.cross_attn.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)  # Xavier uniform initialization

        # # Example initialization for feedforward neural network weights
        # for p in self.linear1.parameters():
        #     if p.dim() > 1:
        #         init.kaiming_uniform_(p, a=torch.nn.functional.gelu)  # Kaiming uniform initialization

        # # Example initialization for biases
        # for p in self.parameters():
        #     if p.dim() == 1:
        #         init.constant_(p, 0)  # Initialize biases to zero

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def _get_activation_fn(self, activation: str) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def forward(
            self,
            src: Tensor,
            conv_src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        is_causal = False
        x = src
        x_conv = conv_src
        if self.norm_first:        
            x = x + self._sa_block(
                self.norm1(x),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )[0]
            x = x + self._ca_block(
                self.norm1(x),
                self.norm2(x_conv),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )[0]
            x = x + self._ff_block(self.norm3(x))
            attn_weights = None
        else:
            x_attn_sa, _ = self._sa_block(
                x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
            x = self.norm1(x + x_attn_sa)                                   

            x_attn, attn_weights = self._ca_block(
                x,
                x_conv,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

            x = self.norm2(x + x_attn)
            
            x = self.norm3(x + self._ff_block(x))

        return x, attn_weights

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        # x, x_attn_out = self.self_attn(x, x, x,
        #                     need_weights=True, average_attn_weights=True)[:2]
        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            average_attn_weights=False,
        )[:2]
        return self.dropout1(x), None #x_attn_out
    
    # cross-attention block
    def _ca_block(
        self,
        x: Tensor,
        x_conv: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        # x_out, x_attn_out = self.cross_attn(x, x_conv, x_conv,
        #                     need_weights=True, average_attn_weights=False)[:2]

        x_out, x_attn_out = self.cross_attn(
            x,
            x_conv,
            x_conv,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )[:2]
        
        # x, x_attn_out = self.cross_attn(x, x_conv, x_conv)[:2]
        return self.dropout2(x_out), x_attn_out

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
