# PyLaia-Based Toolkit for Mathematical Expression Recognition

This repository contains a modified PyLaia-based toolkit for typeset
mathematical expression recognition (MER) with CTC models. The original PyLaia toolkit
targets handwritten text recognition (HTR); the changes in this repository adapt
the model configuration, transformer context modeling, decoding utilities, and
alignment recalibration workflow to MER.

The provided model is used in the PrIx mathematical expression indexing framework
to generate CTC posteriorgrams from images of mathematical expressions. These
posteriorgrams encode frame-level probability distributions over the recognition
vocabulary, including the CTC blank symbol, and are subsequently used to construct
recognition lattices representing multiple recognition hypotheses for downstream
indexing and exact matching of mathematical expressions.

This repository provides the recognition model and the tools required to generate
these posteriorgrams. Recognition lattice construction, hierarchical segmentation,
and indexing are handled by subsequent stages of the PrIx pipeline and are not
included in this repository.

It includes ready-to-use configuration files for the main workflow:

- model creation
- CTC training
- CTC decoding
- neural-network output export (`netout`)

The current configuration targets the IBEM dataset and includes a trained model
checkpoint in `EXP/experiment/epoch=299-lowest_va_cer.ckpt`.

## Repository Contents

| Path | Description |
| --- | --- |
| `model_config.yaml` | Model creation configuration. |
| `train_config.yaml` | Training configuration. |
| `decode_config.yaml` | CTC decoding and alignment recalibration configuration. |
| `netout_config.yaml` | Neural-network output export configuration. |
| `symb_mixt.txt` | Symbol table used by the configured model. |
| `EXP/` | Experiment folder with the provided model checkpoint. |
| `laia/` | PyLaia source code and HTR scripts. |

The configured model is named `m7` and is loaded from `EXP/`.

## Installation

Install the project:

```bash
pip install -e .
```

The model weights are provided via Git LFS:

```bash
git lfs install
git lfs track "*.ckpt"
```

The following command-line tools are installed:

```bash
pylaia-htr-create-model
pylaia-htr-train-ctc
pylaia-htr-decode-ctc
pylaia-htr-netout
```

## Dataset

The provided model was trained on the IBEM dataset:

[https://zenodo.org/records/21528113](https://zenodo.org/records/21528113)

The example configuration files assume the following dataset files/directories
are available from the working directory:

```text
Training/
Validation/
Test/
tr.txt
va.txt
ts.lst
va.lst
symb_mixt.txt
```

Adjust the `img_dirs`, `tr_txt_table`, `va_txt_table`, `img_list`, and `syms`
entries in the YAML files if your local layout is different.

## Model Creation

Create the model described by `model_config.yaml`:

```bash
pylaia-htr-create-model --config model_config.yaml
```

The current model uses:

- VGG-style CNN backend
- Transformer context backend
- 8 transformer encoder layers
- 4 transformer encoder-like cross-attention layers
- transformer model size `768`
- 12 attention heads

The created model is saved as:

```text
EXP/m7
```

## Training

Train using `train_config.yaml`:

```bash
pylaia-htr-train-ctc --config train_config.yaml
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
`train_config.yaml`.

## Decoding

Decode the test set using `decode_config.yaml`:

```bash
pylaia-htr-decode-ctc --config decode_config.yaml
```

The provided decode configuration loads:

```text
EXP/experiment/epoch=299-lowest_va_cer.ckpt
```

and writes alignment recalibration output to:

```text
EXP/experiment/recalibration.txt
```

## Netout

Export network outputs using `netout_config.yaml`:

```bash
pylaia-htr-netout --config netout_config.yaml
```

The provided configuration writes log-softmax confidence matrices to:

```text
EXP/experiment/ConfMats_va.ark
```

These matrices represent the CTC posteriorgrams in the log-probability domain.
They encode frame-level log posterior probabilities over the recognition vocabulary,
including the CTC blank symbol, and serve as the input for constructing recognition
lattices in the PrIx indexing pipeline.

Use `netout.output_transform` to choose the exported representation:

- `null`: raw logits
- `softmax`: posterior probabilities
- `log_softmax`: log posterior probabilities

The subsequent construction of recognition lattices is performed outside this repository.

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
