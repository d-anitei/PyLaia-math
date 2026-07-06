from itertools import count
from typing import Mapping, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from laia.common.types import Param2d, ParamNd
from laia.data import PaddedTensor
from laia.models.htr import ConvBlock
from laia.nn import ImagePoolingSequencer
from .custom_resnet import CustomResnet

import math

from .custom_transformer_decoder_layer import TransformerDecoderLayer
from .custom_transformer_encoder_layer import TransformerEncoderLayer
from torch import Tensor

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.5, max_height: int = 1000, max_width: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings for the height (rows)
        position_h = torch.arange(max_height).unsqueeze(1)
        div_term_h = torch.exp(torch.arange(0, d_model // 2, 2) * (-math.log(10000.0) / (d_model // 2)))
        pe_h = torch.zeros(max_height, d_model // 2)
        pe_h[:, 0::2] = torch.sin(position_h * div_term_h)
        pe_h[:, 1::2] = torch.cos(position_h * div_term_h)

        # Compute positional encodings for the width (columns)
        position_w = torch.arange(max_width).unsqueeze(1)
        div_term_w = torch.exp(torch.arange(0, d_model // 2, 2) * (-math.log(10000.0) / (d_model // 2)))
        pe_w = torch.zeros(max_width, d_model // 2)
        pe_w[:, 0::2] = torch.sin(position_w * div_term_w)
        pe_w[:, 1::2] = torch.cos(position_w * div_term_w)

        # Combine height and width encodings
        pe = torch.zeros(max_height, max_width, d_model)
        pe[:, :, :d_model // 2] = pe_h.unsqueeze(1).expand(-1, max_width, -1)
        pe[:, :, d_model // 2:] = pe_w.unsqueeze(0).expand(max_height, -1, -1)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, height, width, embedding_dim]``
        """
        batch_size, height, width, d_model = x.shape
        if d_model != self.pe.shape[-1]:
            raise ValueError("Input embedding dimension must match positional encoding dimension.")
        
        x = x + self.pe[:height, :width].unsqueeze(0)
        return self.dropout(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.5, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LaiaCRNN(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        num_output_labels: int,
        cnn_num_features: Sequence[int],
        cnn_kernel_size: Sequence[Param2d],
        cnn_stride: Sequence[Param2d],
        cnn_dilation: Sequence[Param2d],
        cnn_activation: Sequence[Type[nn.Module]],
        cnn_poolsize: Sequence[Param2d],
        cnn_dropout: Sequence[float],
        cnn_batchnorm: Sequence[bool],
        image_sequencer: str,
        rnn_units: int,
        rnn_layers: int,
        rnn_dropout: float,
        lin_dropout: float,
        rnn_type: Union[nn.LSTM, nn.GRU, nn.RNN] = nn.LSTM,
        context_dropout: Optional[float] = None,
        inplace: bool = False,
        vertical_text: bool = False,
        use_masks: bool = False,
        cnn_backend: str = "vgg",
        resnet_dimension: int = 34,
        resnet_weights: Optional[str] = "DEFAULT",
        resnet_pool: Optional[Mapping[int, Tuple[int, int]]] = None,
        context_backend: str = "rnn",
        transformer_encoder_layers: int = 8,
        transformer_cross_attention_layers: int = 4,
        transformer_model_size: int = 768,
        transformer_num_heads: int = 12,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if context_dropout is None:
            context_dropout = rnn_dropout
        self._context_dropout = context_dropout
        self._lin_dropout = lin_dropout
        self.cross_attention_recalibration_layer = 1

        ni = num_input_channels      

        # Add sequencer module to convert an image into a sequence
        self.sequencer = ImagePoolingSequencer(
            sequencer=image_sequencer, columnwise=not vertical_text
        )

        if hasattr(cnn_backend, "value"):
            cnn_backend = cnn_backend.value
        self.cnn_backend = cnn_backend

        if self.cnn_backend == "resnet":
            self.conv = CustomResnet(
                resnet_dimension,
                weights=resnet_weights,
                pool=resnet_pool,
            )
            ni = self.conv.output_channels
        elif self.cnn_backend == "vgg":
            conv_blocks = []
            for i, nh, ks, st, di, f, ps, dr, bn in zip(
                count(),
                cnn_num_features,
                cnn_kernel_size,
                cnn_stride,
                cnn_dilation,
                cnn_activation,
                cnn_poolsize,
                cnn_dropout,
                cnn_batchnorm,
            ):
                conv_blocks.append(
                    ConvBlock(
                        in_channels=ni,
                        out_channels=nh,
                        kernel_size=ks,
                        stride=st,
                        dilation=di,
                        activation=f,
                        poolsize=ps,
                        dropout=dr,
                        batchnorm=bn,
                        inplace=inplace,
                        use_masks=use_masks,
                    )
                )
                ni = nh

            self.conv = nn.Sequential(*conv_blocks)
        else:
            raise ValueError(f"Unknown CNN backend: {cnn_backend}")

        if hasattr(context_backend, "value"):
            context_backend = context_backend.value
        self.context_backend = context_backend

        if self.context_backend == "rnn":
            self.rnn = rnn_type(
                ni * self.sequencer.fix_size,
                rnn_units,
                rnn_layers,
                dropout=rnn_dropout,
                bidirectional=True,
                batch_first=False,
            )
            self.rnn.flatten_parameters()
            self.linear = nn.Linear(2 * rnn_units, num_output_labels)
        elif self.context_backend == "transformer":
            pool_size = int((ni * self.sequencer.fix_size) / transformer_model_size)
            if pool_size < 1:
                raise ValueError(
                    "The transformer context backend requires "
                    "`cnn_features * sequencer_size >= transformer_model_size`."
                )

            self.avg_pool = nn.AvgPool1d(pool_size, stride=pool_size)
            self.pos_encoder = PositionalEncoding(
                transformer_model_size,
                transformer_dropout,
            )
            self.encoder_layers = nn.ModuleList(
                TransformerEncoderLayer(
                    d_model=transformer_model_size,
                    nhead=transformer_num_heads,
                    dim_feedforward=transformer_model_size * 4,
                    dropout=transformer_dropout,
                    activation="gelu",
                )
                for _ in range(transformer_encoder_layers)
            )
            self.decoder_layers = nn.ModuleList(
                TransformerDecoderLayer(
                    d_model=transformer_model_size,
                    nhead=transformer_num_heads,
                    dim_feedforward=transformer_model_size * 4,
                    dropout=transformer_dropout,
                    activation="gelu",
                )
                for _ in range(transformer_cross_attention_layers)
            )
            self.linear = nn.Linear(transformer_model_size, num_output_labels)
            self.linear_struct = nn.Linear(transformer_model_size, 156)
        else:
            raise ValueError(f"Unknown context backend: {context_backend}")

        self.gradients = None

    def dropout(
        self, x: Union[torch.Tensor, PaddedTensor, PackedSequence], p: float
    ) -> Union[torch.Tensor, PaddedTensor, PackedSequence]:
        if 0.0 < p < 1.0:
            cls = None
            if isinstance(x, PaddedTensor):
                cls = PaddedTensor
                x, xs = x.data, x.sizes
            elif isinstance(x, PackedSequence):
                cls = PackedSequence
                x, xs = x.data, x.batch_sizes
            d = F.dropout(x, p=p, training=self.training)
            return cls(d, xs) if cls is not None else d
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def forward(
        self, x: Union[torch.Tensor, PaddedTensor]
    ) -> Union[torch.Tensor, PackedSequence]:
        raw_sizes = x.sizes.clone() if isinstance(x, PaddedTensor) else None
        x_f = x.data.clone() if isinstance(x, PaddedTensor) else x.clone()
        x = self.conv(x)
        conv_sizes = x.sizes.clone() if isinstance(x, PaddedTensor) else None
        x_c = x.data.clone() if isinstance(x, PaddedTensor) else x.clone()

        x = self.sequencer(x)

        x = self.dropout(x, p=self._context_dropout)

        # rnn context backend
        if self.context_backend == "rnn": 
            x, _ = self.rnn(x)
            x = self.dropout(x, p=self._lin_dropout)
            logits = (
                PackedSequence(
                    self.linear(x.data),
                    x.batch_sizes,
                )
                if isinstance(x, PackedSequence)
                else self.linear(x)
            )
            return logits, None, None
        
        # transformer context backend
        if isinstance(x, PackedSequence):
            x, sequence_lengths = pad_packed_sequence(x)
        else:
            sequence_lengths = None

        x = x.permute(1, 0, 2)
        x = self.avg_pool(x)
        x = x.permute(1, 0, 2)
        if sequence_lengths is not None:
            sequence_lengths = torch.clamp(
                sequence_lengths,
                min=1,
                max=x.size(0),
            )

        x = self.pos_encoder(x)
        x_conv = x.clone()

        attn_out = None
        for encoder_layer in self.encoder_layers:
            x, _ = encoder_layer(x)
        for layer_idx, cross_attention_layer in enumerate(self.decoder_layers):
            x, cross_attn_out = cross_attention_layer(x, x_conv)
            if layer_idx == self.cross_attention_recalibration_layer:
                attn_out = cross_attn_out

        x = self.dropout(x, p=self._lin_dropout)
        logits = self.linear(x)
        if sequence_lengths is not None:
            logits = pack_padded_sequence(
                logits,
                sequence_lengths.cpu().tolist(),
                enforce_sorted=False,
            )
        return logits, attn_out, [x_f, x_c, raw_sizes, conv_sizes]

    @staticmethod
    def get_conv_output_size(
        size: Param2d,
        cnn_kernel_size: Sequence[ParamNd],
        cnn_stride: Sequence[ParamNd],
        cnn_dilation: Sequence[ParamNd],
        cnn_poolsize: Sequence[ParamNd],
    ) -> Tuple[Union[torch.LongTensor, int]]:
        size_h, size_w = size
        for ks, st, di, ps in zip(
            cnn_kernel_size, cnn_stride, cnn_dilation, cnn_poolsize
        ):
            size_h = ConvBlock.get_output_size(
                size_h, kernel_size=ks[0], dilation=di[0], stride=st[0], poolsize=ps[0]
            )
            size_w = ConvBlock.get_output_size(
                size_w, kernel_size=ks[1], dilation=di[1], stride=st[1], poolsize=ps[1]
            )
        return size_h, size_w

    def get_self_conv_output_size(
        self, size: Param2d
    ) -> List[Union[torch.Tensor, int]]:
        xs = size.clone()
        for l in self.conv:
            xs = l.get_batch_output_size(xs)
        return xs
