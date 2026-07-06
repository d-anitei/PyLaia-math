from pathlib import Path
from typing import Any, Dict, List, Callable, Optional, Union
import warnings

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from laia.utils import SymbolsTable
from laia.decoders import CTCGreedyDecoder

from laia.engine.engine_exception import exception_catcher

import numpy as np
import os, re
import matplotlib.pyplot as plt

import torchvision.transforms as T
from PIL import Image, ImageDraw

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

import math
import torch.nn.functional as F
from torch.nn import MultiheadAttention

import time
import cv2
import imutils

import numpy as np
import cv2

def merge_symbs(seg, probs):
    symbs = []

    for i,s in enumerate(seg):
        frame, symb, pos = s[:3]

        if len(symbs) == 0 or symb != symbs[-1][1]:
            symbs.append(s + [probs[i]])
        else:
            if probs[i] >= symbs[-1][-1]:
                symbs[-1] = s + [probs[i]]

    return symbs

def cross_attn_map(attn_map, layers=None, batch_idx=None):
    attn_heads = attn_map.detach()
    if batch_idx is not None and attn_heads.ndim == 4:
        attn_heads = attn_heads[batch_idx]
    elif attn_heads.ndim == 4:
        if attn_heads.shape[0] != 1:
            raise ValueError("batch_idx is required for batched attention maps")
        attn_heads = attn_heads.squeeze(0)

    if layers is None:
        layers = range(attn_heads.shape[0])
    layers = list(layers)
    if not layers:
        raise ValueError("At least one attention layer must be selected")

    selected_maps = [
        np.array(attn_heads[layer].to("cpu"))
        for layer in layers
    ]
    da_all = sum(selected_maps) / len(selected_maps)
    
    da_all[:,:1] = 0    

    return da_all

def _to_int(value):
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)

def _valid_sample_size(sizes, batch_idx, fallback_shape):
    if sizes is None:
        return fallback_shape
    h, w = sizes[batch_idx, -2:].tolist()
    return _to_int(h), _to_int(w)

def _slice_image(image, height, width):
    return image[..., :height, :width]

def _slice_feature_map(feature_map, height, width):
    return feature_map[..., :height, :width]

def _normalize01(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(x, np.float32)

def _pdf(col):
    col = np.maximum(col.astype(np.float32), 0.0)
    s = float(col.sum())
    return (col / s) if s > 1e-8 else np.ones_like(col, np.float32) / len(col)

def _normalized_axis(length):
    if length <= 1:
        return np.zeros((length,), dtype=np.float32)
    return np.linspace(0.0, 1.0, num=length, dtype=np.float32)

def _peak_centroid_1d(values, radius=3):
    values = np.maximum(values.astype(np.float32), 0.0)
    if values.size == 0:
        return 0.0

    peak = int(np.argmax(values))
    left = max(0, peak - radius)
    right = min(values.shape[0], peak + radius + 1)
    window = values[left:right]
    weight_sum = float(window.sum())
    if weight_sum <= 1e-8:
        return float(peak)

    xs = np.arange(left, right, dtype=np.float32)
    return float((xs * window).sum() / weight_sum)

def _resize_raw_image_to_feature_map(raw_image, target_shape):
    target_h, target_w = target_shape

    if isinstance(raw_image, torch.Tensor):
        raw = raw_image.detach().float().cpu().squeeze(0)
        if raw.ndim == 3:
            raw = raw.mean(dim=0)
        raw = raw.numpy()
    else:
        raw = np.asarray(raw_image, dtype=np.float32)
        if raw.ndim == 3:
            raw = raw.mean(axis=0)

    raw = _normalize01(raw)
    if raw.shape != (target_h, target_w):
        raw = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return raw.astype(np.float32)

def _raw_symbol_contour_prior(raw_image, target_shape):
    raw_small = _resize_raw_image_to_feature_map(raw_image, target_shape)

    # The raw image is already negative, so symbol strokes are bright.
    ink = _normalize01(raw_small)

    edges = cv2.Canny((ink * 255).astype(np.uint8), 40, 120).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(
        edges,
        (3, 3),
        sigmaX=0.8,
        sigmaY=0.8,
        borderType=cv2.BORDER_REPLICATE,
    )

    contour_prior = np.maximum(ink, 0.8 * edges)
    return _normalize01(contour_prior)

def columnwise_elastic_warp_diff_heights(attn_src, cc_tgt, raw_image=None, smooth_sigma_cols=1.5):
    """
    attn_src: (H_src, W) attention map to warp
    cc_tgt  : (H_tgt, W) target (connected components / activations)
    Returns:
      attn_warped: (H_tgt, W)
    """
    kernel = np.ones((2, 2), np.uint8)
    attn_src = cv2.dilate(attn_src, kernel, iterations=1)
    cc_tgt_soft = cv2.GaussianBlur(
        cc_tgt.astype(np.float32),
        (5, 3),
        sigmaX=1.1,
        sigmaY=0.25,
        borderType=cv2.BORDER_REPLICATE,
    )
    cc_tgt_soft = _normalize01(cc_tgt_soft)
    cc_tgt_soft = np.power(cc_tgt_soft, 1.5)
    cc_tgt_soft = cv2.morphologyEx(
        cc_tgt_soft,
        cv2.MORPH_CLOSE,
        np.ones((1, 3), np.uint8),
    )

    A = _normalize01(attn_src)
    C = _normalize01(cc_tgt_soft)
    if raw_image is not None:
        raw_prior = _raw_symbol_contour_prior(raw_image, C.shape)
        C = _normalize01(np.maximum(0.65 * C, raw_prior))
        C = np.power(C, 1.3)

    # cv2.imwrite("feature_maps/A.jpg",A*255)
    # cv2.imwrite("feature_maps/C.jpg",C*255)

    Hs, W = A.shape
    Ht, _ = C.shape

    # Inverse map for cv.remap: for each (y_tgt, w) give y_src to sample
    map_x = np.tile(np.arange(W, dtype=np.float32)[None, :], (Ht, 1))  # dx = 0
    map_y = np.zeros((Ht, W), dtype=np.float32)

    ys_norm = _normalized_axis(Hs)
    yt_norm = _normalized_axis(Ht)

    for w in range(W):
        p_src = _pdf(A[:, w])   # PDF along src height
        p_tgt = _pdf(C[:, w])   # PDF along tgt height

        P = np.cumsum(p_src)
        Q = np.cumsum(p_tgt)
        phi_norm = np.interp(P, Q, yt_norm, left=0.0, right=1.0)  # y_src_norm -> y_tgt_norm

        # Invert forward map in normalized space, then convert back to source rows for remap.
        ys_inv_norm = np.interp(yt_norm, phi_norm, ys_norm, left=0.0, right=1.0)
        ys_inv = ys_inv_norm * float(max(Hs - 1, 0))
        map_y[:, w] = ys_inv

    # Optional: smooth inverse map across columns (stabilizes warping)
    if smooth_sigma_cols and smooth_sigma_cols > 0:
        ksize = int(max(3, round(6 * smooth_sigma_cols)) | 1)
        kx = cv2.getGaussianKernel(ksize, smooth_sigma_cols).reshape(-1).astype(np.float32)
        map_y = cv2.sepFilter2D(map_y, cv2.CV_32F, kx, np.array([1.0], np.float32))

    # Warp attention with the (possibly smoothed) inverse map
    attn_warped = cv2.remap(A, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)

    # Recompute a forward map consistent with the (smoothed) inverse:
    phi_field = np.zeros((Hs, W), dtype=np.float32)
    for w in range(W):
        psi = map_y[:, w]  # length Ht, maps target rows -> source rows
        psi_norm = psi / float(max(Hs - 1, 1))
        # Invert psi by interpolation to get forward: ys -> yt
        phi_w_norm = np.interp(ys_norm, psi_norm, yt_norm, left=0.0, right=1.0)
        phi_w = phi_w_norm * float(max(Ht - 1, 0))
        phi_field[:, w] = phi_w

    return attn_warped, phi_field, C


class EvaluatorModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        symbols: Optional[Union[dict, SymbolsTable]] = None,
        batch_input_fn: Optional[Callable] = None,
        batch_id_fn: Optional[Callable] = None,
        recalibrate_alignment: bool = False,
        cross_attention_recalibration_layer: int = 1,
        recalibration_attention_layers: Optional[List[int]] = None,
        recalibration_output_path: str = "pos_mapping_ts_det_nogt.txt",
        output_temperature: float = 1.0,
    ):
        super().__init__()
        if output_temperature <= 0:
            raise ValueError("`output_temperature` must be positive")
        self.model = model
        self.symbols = symbols
        self.batch_input_fn = batch_input_fn
        self.batch_id_fn = batch_id_fn
        self.output_temperature = output_temperature
        self.decoder = CTCGreedyDecoder()
        self.recalibrate_alignment = recalibrate_alignment
        self.cross_attention_recalibration_layer = (
            cross_attention_recalibration_layer
        )
        self.recalibration_attention_layers = (
            [0, 3, 4, 6, 9]
            if recalibration_attention_layers is None
            else recalibration_attention_layers
        )
        self.recalibration_output_path = recalibration_output_path
        self._recalibration_output_initialized = False
        if self.recalibrate_alignment:
            if self.batch_id_fn is None:
                raise ValueError(
                    "Alignment recalibration requires a batch_id_fn to label "
                    "output entries."
                )
            self._validate_alignment_recalibration_model()

    def _scale_model_output(self, output):
        if self.output_temperature == 1.0:
            return output
        if isinstance(output, PackedSequence):
            return PackedSequence(
                output.data / self.output_temperature,
                output.batch_sizes,
                output.sorted_indices,
                output.unsorted_indices,
            )
        return output / self.output_temperature

    def _prepare_recalibration_output_once(self):
        if (
            not self.recalibrate_alignment
            or self._recalibration_output_initialized
        ):
            return

        try:
            is_global_zero = getattr(self.trainer, "is_global_zero", True)
        except RuntimeError:
            is_global_zero = True

        output_path = Path(self.recalibration_output_path)
        if is_global_zero and output_path.exists():
            warnings.warn(
                f"Removing existing alignment recalibration output file: {output_path}",
                RuntimeWarning,
            )
            output_path.unlink()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        self._recalibration_output_initialized = True

    def on_test_start(self):
        self._prepare_recalibration_output_once()

    def on_test_epoch_start(self):
        self._prepare_recalibration_output_once()

    def _validate_alignment_recalibration_model(self):
        if getattr(self.model, "context_backend", None) != "transformer":
            raise ValueError(
                "Alignment recalibration is only available for transformer "
                "context models."
            )
        decoder_layers = getattr(self.model, "decoder_layers", None)
        if decoder_layers is None or len(decoder_layers) < 1:
            raise ValueError(
                "Alignment recalibration requires at least one transformer "
                "cross-attention layer."
            )
        if self.cross_attention_recalibration_layer >= len(decoder_layers):
            raise ValueError(
                "`cross_attention_recalibration_layer` must be in the "
                "zero-based range [0, number_of_cross_attention_layers - 1]"
            )
        self.model.cross_attention_recalibration_layer = (
            self.cross_attention_recalibration_layer
        )

    @staticmethod
    def _normalize_batch_filenames(filenames):
        if isinstance(filenames, str):
            return [filenames]
        return list(filenames)

    def _write_alignment_recalibration(
        self,
        fd,
        filename,
        raw_img,
        conv_features,
        raw_sizes,
        conv_sizes,
        attn_map,
        decoding,
        batch_idx,
    ):
        transform = T.ToPILImage()

        seg = [
            [i, self.symbols[s], s]
            for i, s in enumerate(decoding["indices"][batch_idx].tolist())
        ]
        probs = [i for i in decoding["probs"][batch_idx].tolist()]

        raw_h, raw_w = _valid_sample_size(
            raw_sizes,
            batch_idx,
            raw_img[batch_idx].shape[-2:],
        )
        conv_h, conv_w = _valid_sample_size(
            conv_sizes,
            batch_idx,
            conv_features[batch_idx].shape[-2:],
        )
        sample_raw_img = _slice_image(raw_img[batch_idx], raw_h, raw_w)
        sample_conv_features = _slice_feature_map(
            conv_features[batch_idx],
            conv_h,
            conv_w,
        )

        x_img = sample_conv_features.mean(dim=0)
        x_img[:,:1] = 0
        x_img[:,-1:] = 0
        img = transform(x_img)
        img_cc = np.array(img)

        cross_map = cross_attn_map(
            attn_map,
            layers=self.recalibration_attention_layers,
            batch_idx=batch_idx,
        )[:, :conv_w]

        attn_warped, px_map, c_soft = columnwise_elastic_warp_diff_heights(
            cross_map,
            img_cc,
            raw_image=sample_raw_img,
        )

        symbs = merge_symbs(seg, probs)

        fd.write(filename + "\n")

        for s in symbs:
            frame, symb, symb_id = s[:3]

            row = np.maximum(cross_map[frame].astype(np.float32), 0.0)
            y_candidates = np.clip(
                np.rint(px_map[frame]).astype(np.int32),
                0,
                c_soft.shape[0] - 1,
            )
            grounded_row = row * c_soft[y_candidates, np.arange(row.shape[0])]

            if grounded_row.max() <= 1e-8:
                grounded_row = row

            w_align = _peak_centroid_1d(grounded_row, radius=3)
            w_idx = int(np.clip(round(w_align), 0, row.shape[0] - 1))
            h_align = int(round(px_map[frame, w_idx]))

            print(frame, int(w_align), h_align, symb, file=fd)

        fd.write("\n")

    def test_step(self, batch: Any, *args, **kwargs) -> torch.Tensor:
        batch_x = self.batch_input_fn(batch)
        
        with exception_catcher(
            self.batch_id_fn(batch) if self.batch_id_fn else batch,
            self.current_epoch,
            self.global_step,
        ):
            model_output = self.model(batch_x)
            output = model_output[0] if isinstance(model_output, tuple) else model_output
            output = self._scale_model_output(output)
            if not self.recalibrate_alignment:
                return output
            self._prepare_recalibration_output_once()

            if not isinstance(model_output, tuple) or len(model_output) < 3:
                raise ValueError(
                    "Alignment recalibration requires model output in the form "
                    "(output, attention_map, feature_maps)."
                )
            _, attn_map, x_features = model_output
            if attn_map is None or x_features is None:
                raise ValueError(
                    "Alignment recalibration requires model attention maps and "
                    "feature maps."
                )

            fnames = self._normalize_batch_filenames(self.batch_id_fn(batch))
            decoding = self.decoder(output)
            if len(fnames) != len(decoding["indices"]):
                raise ValueError(
                    "The number of batch ids does not match the decoded batch size"
                )
            raw_img, conv_features = x_features[:2]
            raw_sizes = x_features[2] if len(x_features) > 2 else None
            conv_sizes = x_features[3] if len(x_features) > 3 else None

            with open(self.recalibration_output_path, "a") as fd:
                for batch_idx, filename in enumerate(fnames):
                    self._write_alignment_recalibration(
                        fd,
                        filename,
                        raw_img,
                        conv_features,
                        raw_sizes,
                        conv_sizes,
                        attn_map,
                        decoding,
                        batch_idx,
                    )

        return output

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # remove version number
        items.pop("v_num", None)
        return items
