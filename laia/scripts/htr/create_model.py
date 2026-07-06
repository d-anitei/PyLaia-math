#!/usr/bin/env python3
from typing import Any, Dict, List, Optional

import jsonargparse
import torch
import torch.nn as nn
from jsonargparse.typing import NonNegativeInt
from pytorch_lightning import seed_everything

import laia.common.logging as log
from laia.common.arguments import CommonArgs, CreateCRNNArgs
from laia.common.saver import ModelSaver
from laia.models.htr import LaiaCRNN
from laia.models.htr.custom_resnet import CustomResnet
from laia.scripts.htr import common_main
from laia.utils import SymbolsTable


def run(
    syms: str,
    fixed_input_height: Optional[NonNegativeInt] = 0,
    adaptive_pooling: str = "avgpool-16",
    common: CommonArgs = CommonArgs(),
    crnn: CreateCRNNArgs = CreateCRNNArgs(),
    save_model: bool = False,
) -> LaiaCRNN:
    seed_everything(common.seed)

    crnn.num_output_labels = len(SymbolsTable(syms))
    if crnn is not None:
        if fixed_input_height:
            if crnn.cnn_backend == crnn.CNNBackend.resnet:
                conv_output_size = CustomResnet.get_output_size(
                    torch.tensor([fixed_input_height, fixed_input_height]),
                    crnn.resnet_dimension,
                    crnn.resnet_pool,
                )
            else:
                conv_output_size = LaiaCRNN.get_conv_output_size(
                    size=(fixed_input_height, fixed_input_height),
                    cnn_kernel_size=crnn.cnn_kernel_size,
                    cnn_stride=crnn.cnn_stride,
                    cnn_dilation=crnn.cnn_dilation,
                    cnn_poolsize=crnn.cnn_poolsize,
                )
            fixed_size_after_conv = conv_output_size[1 if crnn.vertical_text else 0]
            if hasattr(fixed_size_after_conv, "item"):
                fixed_size_after_conv = fixed_size_after_conv.item()
            assert (
                fixed_size_after_conv > 0
            ), "The image size is too small for the CNN architecture"
            crnn.image_sequencer = f"none-{fixed_size_after_conv}"
        else:
            crnn.image_sequencer = adaptive_pooling
        if crnn.context_backend == crnn.ContextBackend.rnn:
            crnn.rnn_type = getattr(nn, crnn.rnn_type)
        if crnn.cnn_backend == crnn.CNNBackend.vgg:
            crnn.cnn_activation = [getattr(nn, act) for act in crnn.cnn_activation]

    model = LaiaCRNN(**vars(crnn))
    log.info(
        "Model has {} parameters",
        sum(param.numel() for param in model.parameters()),
    )
    if save_model:
        ModelSaver(common.train_path, common.model_filename).save(
            LaiaCRNN, **vars(crnn)
        )
    return model


def get_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = jsonargparse.ArgumentParser(
        parse_as_dict=True,
        description=(
            "Create a model for HTR composed of a CNN encoder, a configurable "
            "context modeler, and a final linear layer. A dropout layer can "
            "precede each convolutional block, the context modeler, and the "
            "final linear layer."
        ),
    )
    parser.add_argument(
        "--config", action=jsonargparse.ActionConfigFile, help="Configuration file"
    )
    parser.add_argument(
        "syms",
        type=str,
        help=(
            "Mapping from strings to integers. "
            "The CTC symbol must be mapped to integer 0"
        ),
    )
    parser.add_argument(
        "--fixed_input_height",
        type=NonNegativeInt,
        default=0,
        help=(
            "Height of the input images. If 0, a variable height model "
            "will be used (see `adaptive_pooling`). This will be used to compute the "
            "model output height at the end of the convolutional layers"
        ),
    )
    parser.add_argument(
        "--adaptive_pooling",
        type=str,
        default="avgpool-16",
        help=(
            "Use our custom adaptive pooling layers. This option allows training with"
            " variable height images. Takes into account the size of each individual"
            " image within the bach (before padding). (allowed: {avg,max}pool-N)"
        ),
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=True,
        help="Whether to save the model to a file",
    )

    parser.add_class_arguments(CommonArgs, "common")
    parser.add_function_arguments(log.config, "logging")
    parser.add_class_arguments(CreateCRNNArgs, "crnn")

    args = parser.parse_args(argv, with_meta=False)

    args["common"] = CommonArgs(**args["common"])
    args["crnn"] = CreateCRNNArgs(**args["crnn"])

    return args


def main():
    args = get_args()
    args = common_main(args)
    run(**args)


if __name__ == "__main__":
    main()
