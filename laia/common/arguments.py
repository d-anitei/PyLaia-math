import inspect
from dataclasses import dataclass, field, make_dataclass
from enum import Enum
from os.path import join
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pytorch_lightning as pl
import torch
from jsonargparse.typing import (
    ClosedUnitInterval,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    restricted_number_type,
)
from packaging import version

GeNeg1Int = restricted_number_type(None, int, (">=", -1))


class Monitor(str, Enum):
    va_loss = "va_loss"
    va_cer = "va_cer"
    va_wer = "va_wer"


@dataclass
class CommonArgs:
    """Common arguments

    Args:
        seed: Seed for random number generators
        train_path: Save any files in this location
        model_filename: Filename of the model
        experiment_dirname: Directory name of the experiment.
            It will be created inside the `train_path`
        checkpoint: Any of:
            1 - None: load the best checkpoint with respect to the `monitor`,
                checkpoints will be searched in the `experiment_dirname` directory
            2 - A filepath: the filepath to the checkpoint (e.g. "/tmp/model.ckpt")
            3 - A filename: the filename of the checkpoint to load inside the
                `experiment_dirname` directory (e.g. "model.ckpt")
            4 - A glob pattern: e.g. "epoch=*.ckpt" will load the checkpoint with
                the highest epoch number, globbing will be done inside the
                `experiment_dirname` directory
        monitor: Metric to monitor for early stopping and checkpointing.
    """

    seed: int = 74565  # 0x12345
    train_path: str = ""
    model_filename: str = "model"
    experiment_dirname: str = "experiment"
    monitor: Monitor = Monitor.va_cer
    checkpoint: Optional[str] = None

    def __post_init__(self):
        self.experiment_dirpath = join(self.train_path, self.experiment_dirname)


@dataclass
class CreateCRNNArgs:
    """Create LaiaCRNN arguments

    Args:
        num_input_channels: Number of channels of the input images
        vertical_text: Whether the text is written vertically
        cnn_backend: CNN encoder backend. Use `vgg` for the configurable
            ConvBlock stack or `resnet` for a pretrained ResNet encoder.
        cnn_num_features: VGG backend only. Number of features in each
            convolutional layer.
        cnn_kernel_size: VGG backend only. Kernel size of each convolutional layer.
             (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])
        cnn_stride: VGG backend only. Stride of each convolutional layer.
             (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])
        cnn_dilation: VGG backend only. Spacing between each convolutional
            layer kernel elements.
             (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])
        cnn_activation: VGG backend only. Type of activation function in each
            convolutional layer. From `torch.nn`
        cnn_poolsize: VGG backend only. MaxPooling size after each convolutional layer.
             (e.g. [n,n,...] or [[h1,w1],[h2,w2],...])
        cnn_dropout: VGG backend only. Dropout probability at the input of each
            convolutional layer.
        cnn_batchnorm: VGG backend only. Whether to do batch normalization before
            the activation in each convolutional layer.
        resnet_dimension: ResNet depth. Only used when `cnn_backend=resnet`.
            Available values: 18, 34, 50, 101, 152.
        resnet_weights: Torchvision weights selector. Use `DEFAULT` for pretrained
            ImageNet weights or None for randomly initialized weights.
        resnet_pool: Optional mapping from ResNet sequential layer indices to
            stride pairs. If None, layers 5, 6, and 7 use [2, 1].
        use_masks: VGG backend only. Whether to apply a zero mask after each
            convolution and non-linear activation.
        context_backend: Context modeler backend. Use `rnn` for the original
            recurrent model or `transformer` for Transformer context layers.
        context_dropout: Dropout probability before the context modeler.
        rnn_layers: RNN context backend only. Number of recurrent layers.
        rnn_units: RNN context backend only. Number of units in each recurrent layer.
        rnn_dropout: RNN context backend only. Dropout probability between
            recurrent layers.
        rnn_type: RNN context backend only. Type of recurrent layer. From `torch.nn`.
        transformer_encoder_layers: Number of Transformer encoder layers. Only
            used when `context_backend=transformer`.
        transformer_cross_attention_layers: Number of Transformer encoder-like
            layers enhanced with cross-attention. These layers are stacked on top
            of the regular `transformer_encoder_layers` and are only used when
            `context_backend=transformer`.
        transformer_model_size: Feature dimension used inside Transformer layers
            (`d_model`). Only used when `context_backend=transformer`.
        transformer_num_heads: Number of attention heads in Transformer layers.
            Only used when `context_backend=transformer`.
        transformer_dropout: Transformer context backend only. Dropout
            probability inside Transformer context layers.
        lin_dropout: Dropout probability at the input of the final linear layer
    """

    class CNNBackend(str, Enum):
        vgg = "vgg"
        resnet = "resnet"

    class ContextBackend(str, Enum):
        rnn = "rnn"
        transformer = "transformer"

    num_input_channels: PositiveInt = 1
    vertical_text: bool = False
    cnn_backend: CNNBackend = CNNBackend.vgg
    cnn_num_features: List[PositiveInt] = field(
        default_factory=lambda: [16, 16, 32, 32]
    )
    cnn_kernel_size: List[Union[PositiveInt, List[PositiveInt]]] = field(
        default_factory=lambda: [3] * 4
    )
    cnn_stride: List[Union[PositiveInt, List[PositiveInt]]] = field(
        default_factory=lambda: [1] * 4
    )
    cnn_dilation: List[Union[PositiveInt, List[PositiveInt]]] = field(
        default_factory=lambda: [1] * 4
    )
    cnn_activation: List[str] = field(default_factory=lambda: ["LeakyReLU"] * 4)
    cnn_poolsize: List[Union[NonNegativeInt, List[NonNegativeInt]]] = field(
        default_factory=lambda: [2, 2, 2, 0]
    )
    cnn_dropout: List[ClosedUnitInterval] = field(default_factory=lambda: [0.0] * 4)
    cnn_batchnorm: List[bool] = field(default_factory=lambda: [False] * 4)
    resnet_dimension: PositiveInt = 34
    resnet_weights: Optional[str] = "DEFAULT"
    resnet_pool: Optional[Dict[int, List[PositiveInt]]] = None
    use_masks: bool = False
    context_backend: ContextBackend = ContextBackend.rnn
    context_dropout: ClosedUnitInterval = 0.5
    rnn_layers: PositiveInt = 3
    rnn_units: PositiveInt = 256
    rnn_dropout: ClosedUnitInterval = 0.5
    rnn_type: str = "LSTM"
    transformer_encoder_layers: PositiveInt = 8
    transformer_cross_attention_layers: NonNegativeInt = 4
    transformer_model_size: PositiveInt = 768
    transformer_num_heads: PositiveInt = 12
    transformer_dropout: ClosedUnitInterval = 0.1
    lin_dropout: ClosedUnitInterval = 0.5

    def __post_init__(self):
        if self.cnn_backend == self.CNNBackend.resnet:
            self._validate_resnet_args()
        elif self.cnn_backend != self.CNNBackend.vgg:
            raise ValueError(f"Unknown CNN backend: {self.cnn_backend}")

        dimensions = map(
            len,
            (
                self.cnn_num_features,
                self.cnn_kernel_size,
                self.cnn_stride,
                self.cnn_dilation,
                self.cnn_activation,
                self.cnn_poolsize,
                self.cnn_dropout,
                self.cnn_batchnorm,
            ),
        )
        if len(set(dimensions)) != 1:
            raise ValueError("Wrong cnn layer dimensions")
        self.cnn_kernel_size = self.parse_parameter(self.cnn_kernel_size)
        self.cnn_stride = self.parse_parameter(self.cnn_stride)
        self.cnn_dilation = self.parse_parameter(self.cnn_dilation)
        self.cnn_poolsize = self.parse_parameter(self.cnn_poolsize)
        if not all(hasattr(torch.nn, act) for act in set(self.cnn_activation)):
            raise ValueError(
                f"Could not find all cnn activations {self.cnn_activation} in `torch.nn`"
            )
        if self.context_backend == self.ContextBackend.transformer:
            self._validate_transformer_args()
        elif self.context_backend != self.ContextBackend.rnn:
            raise ValueError(f"Unknown context backend: {self.context_backend}")
        else:
            self._validate_rnn_args()

        if self.context_backend == self.ContextBackend.rnn and not hasattr(torch.nn, self.rnn_type):
            raise ValueError(f"Could not find RNN type {self.rnn_type} in `torch.nn`")

    def _validate_resnet_args(self):
        if self.num_input_channels != 1:
            raise ValueError("`num_input_channels` must be 1 when `cnn_backend=resnet`")
        if self.use_masks:
            raise ValueError("`use_masks` is only supported when `cnn_backend=vgg`")

        defaults = self._vgg_defaults()
        for name, default in defaults.items():
            if getattr(self, name) != default:
                raise ValueError(
                    f"`{name}` is only used when `cnn_backend=vgg`. "
                    "Use the `resnet_*` options for a ResNet encoder."
                )

        if self.resnet_dimension not in (18, 34, 50, 101, 152):
            raise ValueError("`resnet_dimension` must be one of: 18, 34, 50, 101, 152")

        if self.resnet_pool is not None:
            for layer_idx, stride in self.resnet_pool.items():
                if layer_idx not in (5, 6, 7):
                    raise ValueError("`resnet_pool` keys must be one of: 5, 6, 7")
                if len(stride) != 2:
                    raise ValueError("`resnet_pool` values must be stride pairs")
                if stride[0] < 1 or stride[1] < 1:
                    raise ValueError("`resnet_pool` strides must be positive")

    def _validate_transformer_args(self):
        defaults = self._rnn_defaults()
        for name, default in defaults.items():
            if getattr(self, name) != default:
                raise ValueError(
                    f"`{name}` is only used when `context_backend=rnn`. "
                    "Use the `transformer_*` options for a transformer context modeler."
                )
        if self.transformer_model_size % self.transformer_num_heads != 0:
            raise ValueError(
                "`transformer_model_size` must be divisible by "
                "`transformer_num_heads`"
            )
    def _validate_rnn_args(self):
        defaults = self._transformer_defaults()
        for name, default in defaults.items():
            if getattr(self, name) != default:
                raise ValueError(
                    f"`{name}` is only used when `context_backend=transformer`. "
                    "Use the `rnn_*` options for an RNN context modeler."
                )

    @staticmethod
    def _rnn_defaults():
        return {
            "rnn_layers": 3,
            "rnn_units": 256,
            "rnn_dropout": 0.5,
            "rnn_type": "LSTM",
        }

    @staticmethod
    def _transformer_defaults():
        return {
            "transformer_encoder_layers": 8,
            "transformer_cross_attention_layers": 4,
            "transformer_model_size": 768,
            "transformer_num_heads": 12,
            "transformer_dropout": 0.1,
        }

    @staticmethod
    def _vgg_defaults():
        return {
            "cnn_num_features": [16, 16, 32, 32],
            "cnn_kernel_size": [3] * 4,
            "cnn_stride": [1] * 4,
            "cnn_dilation": [1] * 4,
            "cnn_activation": ["LeakyReLU"] * 4,
            "cnn_poolsize": [2, 2, 2, 0],
            "cnn_dropout": [0.0] * 4,
            "cnn_batchnorm": [False] * 4,
        }

    @staticmethod
    def parse_parameter(
        layers: List[Union[int, List[int]]], dim: int = 2
    ) -> List[List[int]]:
        parsed = []
        for l in layers:
            if isinstance(l, int):
                parsed.append([l] * dim)
            elif isinstance(l, (tuple, list)):
                if not all(isinstance(v, int) for v in l):
                    raise ValueError(f"An element of {layers} is not an int")
                if len(l) != dim:
                    raise ValueError(
                        f"The given input {layers} does not "
                        f"match the given dimensions {dim}"
                    )
                parsed.append(list(l))
            else:
                raise ValueError(f"{l} ({type(l)}) is neither a tuple nor an int")
        return parsed


@dataclass
class DataArgs:
    """Data arguments

    Args:
        batch_size: Batch size
        color_mode: L (grayscale): 1 channel, RGB: 3 channels, RGBA: 4 channels
    """

    class ColorMode(str, Enum):
        L = "L"
        RGB = "RGB"
        RGBA = "RGBA"

    batch_size: PositiveInt = 8
    color_mode: ColorMode = ColorMode.L


@dataclass
class TrainArgs:
    """Train arguments

    Args:
        delimiters: List of symbols representing the word delimiters.
        checkpoint_k:
            -1: all models are saved.
            0: no models are saved.
            k: the best k models will be saved
        resume: Whether to resume training with a checkpoint. See `checkpoint`.
            If a number, resumes training for this number of epochs.
        early_stopping_patience: Number of validation epochs with no improvement
            after which training will be stopped
        gpu_stats: Whether to include GPU stats in the training progress bar
        augment_training: Whether to use dynamic distortions to augment
            the training data
        hard_negative_mining: Whether to duplicate hard samples after each
            epoch. When enabled, the training dataloader is reloaded every
            epoch so the next epoch uses the updated hard-negative table.
        hard_negative_strategy:
            Strategy used to select hard samples. `ratio` duplicates the top
            `hard_negative_ratio` fraction, while `stddev` duplicates samples
            with loss greater than mean + std over the epoch.
        hard_negative_ratio: Fraction of hardest samples to duplicate (e.g. 0.1, 0.2)
        hard_negative_start_epoch: First epoch index where mining is applied
        output_temperature: Divide model logits by this positive temperature
            during training and validation.
    """

    class HardNegativeStrategy(str, Enum):
        ratio = "ratio"
        stddev = "stddev"

    delimiters: Optional[List[str]] = field(default_factory=lambda: ["<space>"])
    checkpoint_k: GeNeg1Int = 3
    resume: Union[bool, NonNegativeInt] = False
    early_stopping_patience: NonNegativeInt = 20
    gpu_stats: bool = False
    augment_training: bool = False
    hard_negative_mining: bool = False
    hard_negative_strategy: HardNegativeStrategy = HardNegativeStrategy.ratio
    hard_negative_ratio: ClosedUnitInterval = 0.1
    hard_negative_start_epoch: NonNegativeInt = 1
    output_temperature: PositiveFloat = 1.0

    def __post_init__(self):
        if self.hard_negative_ratio <= 0:
            raise ValueError("`hard_negative_ratio` must be > 0.")


@dataclass
class OptimizerArgs:
    """Optimizer arguments

    Args:
        name: Optimization algorithm
        learning_rate: Learning rate
        momentum: Momentum
        weight_l2_penalty: Apply this L2 weight penalty to the loss function
        nesterov: Whether to use Nesterov momentum
    """

    class Name(str, Enum):
        SGD = "SGD"
        RMSProp = "RMSProp"
        Adam = "Adam"
        AdamW = "AdamW"

    name: Name = Name.RMSProp
    learning_rate: PositiveFloat = 0.0005
    momentum: NonNegativeFloat = 0.0
    weight_l2_penalty: NonNegativeFloat = 0.0
    nesterov: bool = False


@dataclass
class SchedulerArgs:
    """Scheduler arguments

    Args:
        active: Whether to use a learning rate scheduler
        name: Learning rate scheduler implementation. Available values are
            `reduce_on_plateau`, `linear`, and `cosine_warm_restarts`.
        monitor: Reduce-on-plateau scheduler only. Metric for the scheduler to
            monitor.
        patience: Reduce-on-plateau scheduler only. Number of epochs with no
            improvement after which learning rate will be reduced.
        factor: Reduce-on-plateau scheduler only. Factor by which the learning
            rate will be reduced.
        min_lr: Reduce-on-plateau scheduler only. Lower bound of the learning
            rate.
        linear_start_factor: Initial multiplicative factor for `LinearLR`
        linear_end_factor: Final multiplicative factor for `LinearLR`
        linear_total_iters: Number of iterations for `LinearLR`
        cosine_t0: Number of epochs for the first restart in cosine warm restarts
        cosine_t_mult: Multiplicative factor for restart period growth
        cosine_eta_min: Minimum learning rate for cosine warm restarts
    """

    class Name(str, Enum):
        reduce_on_plateau = "reduce_on_plateau"
        linear = "linear"
        cosine_warm_restarts = "cosine_warm_restarts"

    active: bool = False
    name: Name = Name.reduce_on_plateau
    monitor: Monitor = Monitor.va_loss
    patience: NonNegativeInt = 5
    factor: float = 0.1
    min_lr: NonNegativeFloat = 0.000001
    linear_start_factor: ClosedUnitInterval = 1.0
    linear_end_factor: ClosedUnitInterval = 0.01
    linear_total_iters: PositiveInt = 150
    cosine_t0: PositiveInt = 10
    cosine_t_mult: PositiveInt = 1
    cosine_eta_min: NonNegativeFloat = 0.0

    def __post_init__(self):
        if self.name == self.Name.reduce_on_plateau and not (0.0 < self.factor < 1.0):
            raise ValueError("For reduce_on_plateau, `factor` must be in (0, 1).")


def __get_trainer_fields() -> List[Tuple[str, Type, Any]]:
    sig = inspect.signature(pl.Trainer.__init__)
    parameters = list(sig.parameters.values())
    parameters = parameters[1:]  # exclude self
    # the following are manually set already in the scripts
    blocklist = (
        "checkpoint_callback",
        "default_root_dir",
        "resume_from_checkpoint",
        "log_gpu_memory",
        "logger",
        "callbacks",
        # TODO: support these 2
        "auto_scale_batch_size",
        "auto_lr_find",
    )
    parameters = [p for p in parameters if p.name not in blocklist]
    return [(p.name, p.annotation, p.default) for p in parameters]


@dataclass
class TrainerArgs(make_dataclass("", __get_trainer_fields())):
    __doc__ = pl.Trainer.__init__.__doc__

    def __post_init__(self):
        if (
            version.parse(torch.__version__) < version.parse("1.7.0")
            and self.precision != 32
        ):
            raise ValueError(
                "AMP requires torch>=1.7.0. Additionally, only "
                "fixed height models are currently supported"
            )


@dataclass
class DecodeArgs:
    """Decode arguments

    Args:
        include_img_ids: Include the associated image ids in the
            decoding/segmentation output
        separator: Use this as the separator between the ids and the
            decoding/segmentation output
        join_string: Join the decoding output using this string
        use_symbols: Convert the decoding output to symbols instead of numbers
        convert_spaces: Whether or not to convert spaces
        input_space: For decoding, if `convert_spaces` is set, space symbol
            to be replaced (`input_space` -> `output_space`).
            For word segmentation, symbol used to convert the sequences of
            characters into sentences of words
        output_space: Output space symbol for decoding.
            (`input_space` -> `output_space`)
        segmentation: Print this kind of segmentation instead of decoding.
        recalibrate_alignment: Use transformer cross-attention to write CTC
            alignment recalibration positions during inference. This mode
            should be run with `data.batch_size=1` because the recalibration
            estimate is sensitive to padding introduced by batched samples.
        cross_attention_recalibration_layer: Decode-time cross-attention layer
            index whose attention weights are returned for CTC alignment
            recalibration. Valid values are in the zero-based range
            [0, `transformer_cross_attention_layers` - 1]. Only used when
            `recalibrate_alignment=true`.
        recalibration_attention_layers: Attention head indices used to build the
            recalibration map.
        recalibration_output_path: File where recalibrated alignment positions
            will be appended.
        output_temperature: Divide model logits by this positive temperature
            before decoding.
    """

    class Segmentation(str, Enum):
        char = "char"
        word = "word"

    class Decoder(str, Enum):
        greedy = "greedy"
        nbest = "nbest"

    include_img_ids: bool = True
    separator: str = " "
    join_string: Optional[str] = " "
    use_symbols: bool = True
    convert_spaces: bool = False
    input_space: str = "<space>"
    output_space: str = " "
    segmentation: Optional[Segmentation] = None
    recalibrate_alignment: bool = False
    cross_attention_recalibration_layer: NonNegativeInt = 1
    recalibration_attention_layers: List[NonNegativeInt] = field(
        default_factory=lambda: [0, 3, 4, 6, 9]
    )
    recalibration_output_path: str = "pos_mapping_ts_det_nogt.txt"
    output_temperature: PositiveFloat = 1.0
    print_line_confidence_scores: bool = False
    print_word_confidence_scores: bool = False
    decoder: Decoder = Decoder.greedy
    nbest: PositiveInt = 50
    use_language_model: bool = False
    language_model_path: str = None
    language_model_weight: float = None
    tokens_path: str = None
    lexicon_path: str = None
    unk_token: str = "<unk>"
    blank_token: str = "<ctc>"


@dataclass
class NetoutArgs:
    """Netout arguments

    Args:
        output_transform: Apply this transformation at the end of the model.
            For instance, use "softmax" to get posterior probabilities as the output
            of the model
        output_temperature: Divide model logits by this positive temperature
            before applying `output_transform` and writing netout files.
        matrix: Path of the Kaldi's archive containing the output matrices
            (one for each sample), where each row represents a timestep and
            each column represents a CTC label
        lattice: Path of the Kaldi's archive containing the output lattices
            (one for each sample), representing the CTC output
        digits: Number of digits to be used for formatting
    """

    class OutputTransform(str, Enum):
        softmax = "softmax"
        log_softmax = "log_softmax"

    output_transform: Optional[OutputTransform] = None
    output_temperature: PositiveFloat = 1.0
    matrix: Optional[str] = None
    lattice: Optional[str] = None
    digits: NonNegativeInt = 10
