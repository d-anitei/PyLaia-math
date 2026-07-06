from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from laia.data import PaddedTensor


class CustomResnet(nn.Module):
    SUPPORTED_DIMENSIONS: Tuple[int, ...] = (18, 34, 50, 101, 152)
    STRIDED_LAYER_INDICES: Tuple[int, ...] = (5, 6, 7)
    DEFAULT_POOL: Dict[int, Tuple[int, int]] = {
        5: (2, 1),
        6: (2, 1),
        7: (2, 1),
    }
    _MODEL_NAMES: Dict[int, str] = {
        18: "resnet18",
        34: "resnet34",
        50: "resnet50",
        101: "resnet101",
        152: "resnet152",
    }
    _OUTPUT_CHANNELS: Dict[int, int] = {
        18: 512,
        34: 512,
        50: 2048,
        101: 2048,
        152: 2048,
    }

    def __init__(
        self,
        dimension: int,
        weights: Optional[str] = "DEFAULT",
        pool: Optional[
            Union[Sequence[int], Mapping[int, Tuple[int, int]]]
        ] = None,
    ):
        super().__init__()

        if dimension not in self.SUPPORTED_DIMENSIONS:
            supported = ", ".join(str(dim) for dim in self.SUPPORTED_DIMENSIONS)
            raise ValueError(
                f"Unsupported ResNet dimension {dimension}. "
                f"Expected one of: {supported}."
            )

        self.dimension = dimension
        self.pool = self._normalize_pool(pool)
        self.output_channels = self._OUTPUT_CHANNELS[dimension]
        self.resnet = self._build_backbone(dimension, weights, self.pool)

    @classmethod
    def _build_backbone(
        cls,
        dimension: int,
        weights: Optional[str],
        pool: Mapping[int, Tuple[int, int]],
    ) -> nn.Sequential:
        backbone = cls._load_torchvision_resnet(dimension, weights)
        cls._replace_input_conv(backbone)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        cls._configure_pooling(backbone, pool)
        return backbone

    @classmethod
    def _normalize_pool(
        cls,
        pool: Optional[Union[Sequence[int], Mapping[int, Tuple[int, int]]]],
    ) -> Dict[int, Tuple[int, int]]:
        if pool is None:
            return dict(cls.DEFAULT_POOL)

        if isinstance(pool, Mapping):
            normalized = {}
            for layer_idx, stride in pool.items():
                if not isinstance(stride, Sequence) or len(stride) != 2:
                    raise ValueError(
                        f"Invalid pooling stride {stride} for layer {layer_idx}. "
                        "Expected a pair of positive integers."
                    )
                normalized[layer_idx] = (int(stride[0]), int(stride[1]))
        else:
            normalized = {layer_idx: (2, 1) for layer_idx in pool}

        for layer_idx, stride in normalized.items():
            if layer_idx not in cls.STRIDED_LAYER_INDICES:
                supported = ", ".join(str(idx) for idx in cls.STRIDED_LAYER_INDICES)
                raise ValueError(
                    f"Unsupported ResNet pooling layer {layer_idx}. "
                    f"Expected one of: {supported}."
                )
            if stride[0] < 1 or stride[1] < 1:
                raise ValueError(
                    f"Invalid pooling stride {stride} for layer {layer_idx}. "
                    "Expected a pair of positive integers."
                )

        return normalized

    @classmethod
    def _load_torchvision_resnet(
        cls, dimension: int, weights: Optional[str]
    ) -> nn.Module:
        try:
            from torchvision import models
        except ImportError:
            try:
                return torch.hub.load(
                    "pytorch/vision:v0.10.0",
                    cls._MODEL_NAMES[dimension],
                    weights=weights,
                )
            except TypeError:
                return torch.hub.load(
                    "pytorch/vision:v0.10.0",
                    cls._MODEL_NAMES[dimension],
                    pretrained=weights == "DEFAULT",
                )

        model_fn = getattr(models, cls._MODEL_NAMES[dimension])
        if weights is None:
            return model_fn(weights=None)

        try:
            weights_enum = getattr(models, f"ResNet{dimension}_Weights")
            weights_value = weights_enum.DEFAULT if weights == "DEFAULT" else weights
            return model_fn(weights=weights_value)
        except (AttributeError, TypeError):
            return model_fn(pretrained=weights == "DEFAULT")

    @staticmethod
    def _replace_input_conv(backbone: nn.Module) -> None:
        original_conv = backbone.conv1
        grayscale_conv = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=(1, 1),
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            grayscale_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
            if original_conv.bias is not None:
                grayscale_conv.bias.copy_(original_conv.bias)

        backbone.conv1 = grayscale_conv

    @staticmethod
    def _configure_pooling(
        backbone: nn.Sequential, pool: Mapping[int, Tuple[int, int]]
    ) -> None:
        for layer_idx, stride in pool.items():
            first_block = backbone[layer_idx][0]

            for conv_name in ("conv1", "conv2"):
                conv = getattr(first_block, conv_name, None)
                if conv is not None and conv.stride == (2, 2):
                    conv.stride = stride

            if first_block.downsample is not None:
                first_block.downsample[0].stride = stride

    def forward(self, x: Union[Tensor, PaddedTensor]) -> Union[Tensor, PaddedTensor]:
        x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        out = self.resnet(x)

        return (
            out
            if xs is None
            else PaddedTensor.build(out, self.get_batch_output_size(xs, self.dimension))
        )

    def get_batch_output_size(
        self, xs: torch.Tensor, dim: Optional[int] = None
    ) -> torch.Tensor:
        return self.get_output_size(
            xs, self.dimension if dim is None else dim, self.pool
        )

    @staticmethod
    def get_output_size(
        size: Union[torch.Tensor, int],
        dim: int,
        pool: Optional[
            Union[Sequence[int], Mapping[int, Tuple[int, int]]]
        ] = None,
    ) -> Union[torch.Tensor, int]:
        if dim not in CustomResnet.SUPPORTED_DIMENSIONS:
            supported = ", ".join(str(dim) for dim in CustomResnet.SUPPORTED_DIMENSIONS)
            raise ValueError(
                f"Unsupported ResNet dimension {dim}. Expected one of: {supported}."
            )

        pool = CustomResnet._normalize_pool(pool)
        size = (size + 1) // 2  # After maxpool.
        for layer_idx in CustomResnet.STRIDED_LAYER_INDICES:
            size = CustomResnet._apply_stride(size, pool.get(layer_idx, (2, 2)))
        return size

    @staticmethod
    def _apply_stride(
        size: Union[torch.Tensor, int], stride: Tuple[int, int]
    ) -> Union[torch.Tensor, int]:
        if not isinstance(size, torch.Tensor):
            return (size + stride[0] - 1) // stride[0]

        size = size.clone()
        if size.ndim == 1 and size.numel() == 2:
            size[0] = (size[0] + stride[0] - 1) // stride[0]
            size[1] = (size[1] + stride[1] - 1) // stride[1]
            return size

        if size.ndim > 1:
            size[:, 0] = (size[:, 0] + stride[0] - 1) // stride[0]
            size[:, 1] = (size[:, 1] + stride[1] - 1) // stride[1]
            return size

        size = (size + stride[0] - 1) // stride[0]
        return size
