# PyLaia-Based Toolkit for Mathematical Expression Recognition

This PyLaia toolkit recognizes typeset mathematical expressions as LaTeX tokens.
It includes a pretrained CTC model for the IBEM dataset, example images, and
tools for training, greedy decoding, and constructing recognition lattices.

The PrIx indexing pipeline uses the exported CTC scores to construct lattices
that represent alternative recognition hypotheses. Its hierarchical segmentation
and indexing tools are available in
[kaldi-lattice-utils-math](https://github.com/d-anitei/kaldi-lattice-utils-math).

## Repository Contents

| Path | Description |
| --- | --- |
| `YAML/model_config.yaml` | Model creation configuration. |
| `YAML/train_config.yaml` | Training configuration. |
| `YAML/decode_config.yaml` | CTC decoding and alignment recalibration configuration. |
| `YAML/netout_config.yaml` | Neural-network output export configuration. |
| `utils/symb_mixt.txt` | Symbol table used by the configured model. |
| `examples/tr.txt`, `examples/va.txt`, `examples/ts.txt` | Example training, validation, and test transcripts. |
| `examples/images/` | PNG crops of mathematical expressions for testing the pipeline. |
| `utils/simplex_out` | Optimized acoustic scale and word insertion penalty for language model decoding. |
| `EXP/m7` | Saved model architecture used to reconstruct the network. |
| `EXP/experiment/epoch=299-lowest_va_cer.ckpt` | Pretrained model weights. |
| `scripts/decode_language_model_pylaia.sh` | N-gram decoding and lattice construction. |
| `laia/` | PyLaia source code and HTR scripts. |

## Installation

From the repository root, fetch the pretrained checkpoint with Git LFS and
install PyLaia in your Python environment. The example workflow was tested
with Python 3.9.

```bash
git lfs install
git lfs pull
pip install -e .
```

## Quick start

Run greedy CTC recognition on the eight sample test images:

```bash
pylaia-htr-decode-ctc --config YAML/decode_config.yaml
```

This prints image IDs and recognized tokens to standard output. The reference
transcripts are in `examples/ts.txt`.

To produce lattices, first [install the language model tools](#language-model-decoding).
Export log-softmax scores and decode them with a 2-gram model:

```bash
pylaia-htr-netout --config YAML/netout_config.yaml
bash scripts/decode_language_model_pylaia.sh 2 EXP/experiment/ConfMats_example_ts.ark
```

The decoder writes best-path transcripts to
`EXP/LM/ts-lm-2.txt` and compressed lattices to
`EXP/LM/ts-lm-2.lat.gz`. The sampled training and validation
transcripts are a small corpus for testing the workflow.

## Dataset

The provided model was trained on the IBEM dataset:

[https://zenodo.org/records/21528113](https://zenodo.org/records/21528113)

The configurations in `YAML/` use the provided examples. Run the commands
from the repository root so the configured paths resolve.

```text
examples/
  images/
  tr.txt
  va.txt
  ts.txt
  ts.lst
utils/
  symb_mixt.txt
  simplex_out
```

The training config reads `tr.txt` and `va.txt`. The CTC decode and Netout configs
both read `ts.lst`, which lists the eight test image IDs.

For data preparation, including normalization of LaTeX math transcripts, see
[latex-math-normalization](https://github.com/d-anitei/latex-math-normalization).

For your own data, replace the example images and image list, or point
`img_dirs` and `img_list` to your files. Use one image ID per line in the list.
For training, also replace the sample transcripts or update `tr_txt_table` and
`va_txt_table`. To build an n-gram model for your data, replace
`examples/tr.txt` and `examples/va.txt` with your tokenized training and
validation transcripts.

## Model Creation

Create the model described by `YAML/model_config.yaml`:

```bash
pylaia-htr-create-model --config YAML/model_config.yaml
```

The provided model architecture uses:

- VGG-style CNN backend
- Transformer context backend
- 8 transformer encoder layers
- 4 cross-attention layers
- transformer model size `768`
- 12 attention heads

The default output is `EXP/m7`. Set `common.train_path` to a separate
experiment directory when creating a model alongside the pretrained one. Use
the same experiment directory in the training config.

## Training

Train using `YAML/train_config.yaml`:

```bash
pylaia-htr-train-ctc --config YAML/train_config.yaml
```

The provided training configuration uses:

- `AdamW`
- learning rate `1.0e-4`
- weight decay `1.0e-4`
- gradient clipping `1.0`
- `accumulate_grad_batches: 4`
- validation monitor `va_cer`
- checkpoint output under `EXP/experiment/`

To resume from a checkpoint, set `common.checkpoint` and `train.resume` in
`YAML/train_config.yaml`.

## Decoding

Run CTC decoding with the configured alignment recalibration:

```bash
pylaia-htr-decode-ctc --config YAML/decode_config.yaml
```

PyLaia pairs `EXP/m7` with the pretrained checkpoint:

```text
EXP/experiment/epoch=299-lowest_va_cer.ckpt
```

The command prints the transcripts and saves alignment recalibration output to:

```text
EXP/experiment/recalibration.txt
```

## Netout

Export network outputs using `YAML/netout_config.yaml`:

```bash
pylaia-htr-netout --config YAML/netout_config.yaml
```

The provided configuration writes log-softmax confidence matrices to:

```text
EXP/experiment/ConfMats_example_ts.ark
```

These matrices represent the CTC posteriorgrams in the log-probability domain.
They encode frame-level log posterior probabilities over the recognition vocabulary,
including the CTC blank symbol, and serve as the input for constructing recognition
lattices in the PrIx indexing pipeline.

Use `netout.output_transform` to choose the exported representation:

- `null`: raw logits
- `softmax`: posterior probabilities
- `log_softmax`: log posterior probabilities

## Language model decoding

The math decoder builds a token n-gram language model from `examples/tr.txt` and
`examples/va.txt`. Each row starts with an image ID followed by space-separated
model tokens, for example `0010040_page005.0010040_page005-3-emb R ^ { p }`.
The decoder uses `utils/symb_mixt.txt` and reads the acoustic scale and insertion
penalty from the last `Optimum command:` line of `utils/simplex_out`.

On Linux, install the build prerequisites first. For Debian or Ubuntu:

```bash
sudo apt-get install build-essential git cmake gfortran wget perl gawk bc \
    autoconf automake libtool patch unzip bzip2 sox subversion \
    libboost-program-options-dev libboost-system-dev libboost-thread-dev \
    libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev
bash scripts/install_lm_tools.sh
```

The installer builds OpenFst, OpenBLAS, Kaldi, and KenLM under `utils/`.
The decoder picks up these builds automatically. Set `LM_BUILD_JOBS` to
change the build parallelism. The build log is at `utils/logs/install.log`.

To use existing builds, set `KALDI_ROOT` and `KENLM_ROOT` to their source
checkouts. Tools already on `PATH` also work when no local build is available.
Order `0` selects a uniform vocabulary baseline and runs without KenLM.

The decoder takes the n-gram order and a log-softmax score archive exported by
Netout. Use a checkpoint whose output classes match the IDs in
`utils/symb_mixt.txt`. To export a different split, update `img_list`,
`img_dirs`, and `netout.matrix` in the Netout config.

The outputs go to `EXP/LM/`. The filenames use the score archive stem and the
n-gram order. A leading `ConfMats_example_` is dropped from the stem. The transcript contains the best path after applying the word
insertion penalty.

## Provided IBEM Results

Recognition results on the IBEM test set for inline, displayed, and all
mathematical expressions:

| Subset | 1-TER | ExStr = 0 | ExStr <= 1 | ExStr <= 2 | ExIm | ExIm-ws |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Inline | 98.7 | 96.9 | 98.3 | 99.1 | 98.8 | 99.0 |
| Displayed | 86.2 | 62.7 | 69.8 | 74.7 | 76.4 | 80.7 |
| All | 90.0 | 90.9 | 93.3 | 94.8 | 95.0 | 95.9 |

## Evaluation Metrics

Evaluation is carried out at both markup and image levels using standard MER
metrics from the MER literature.

- `TER`: Token error rate. It measures the percentage of substitutions,
  insertions, and deletions needed to align the predicted and reference LaTeX
  sequences. In the table above, `1-TER` is reported.
- `ExStr <= epsilon`: Exact string match with tolerance. It computes the
  percentage of mathematical expressions whose predicted LaTeX sequence matches
  the reference while allowing up to `epsilon` token errors, with
  `epsilon in {0, 1, 2}`.
- `ExIm`: Exact image match. It evaluates whether the rendered prediction
  visually matches the rendered reference.
- `ExIm-ws`: Exact image match after whitespace trimming. Empty space columns
  are removed before comparing the rendered prediction and reference.

## Useful Configuration Notes

- `crnn.cnn_backend` can be `vgg` or `resnet`.
- VGG-specific options are under `crnn.cnn_*`.
- ResNet-specific options are under `crnn.resnet_*`.
- `crnn.context_backend` can be `rnn` or `transformer`.
- RNN-specific options are under `crnn.rnn_*`.
- Transformer-specific options are under `crnn.transformer_*`.
- `decode.cross_attention_recalibration_layer` selects which cross-attention
  layer is used for CTC alignment recalibration.
- Runtime logit temperature is configured separately for training, decoding,
  and netout with `output_temperature` in the corresponding config section.

## Relation to Original PyLaia

This toolkit is based on PyLaia, a device-agnostic PyTorch toolkit originally
designed for handwritten text recognition and handwritten document analysis. The
version in this repository extends and configures PyLaia for typeset MER.

Original project:

[https://github.com/jpuigcerver/PyLaia](https://github.com/jpuigcerver/PyLaia)

## Citation

If you use PyLaia, please cite the original toolkit:

```bibtex
@misc{puigcerver2018pylaia,
  author = {Joan Puigcerver and Carlos Mocholí},
  title = {PyLaia},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jpuigcerver/PyLaia}}
}
```
