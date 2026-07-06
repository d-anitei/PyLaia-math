import math

import numpy as np
import pytest
import torch
from PIL import Image

from laia.data.transforms.vision import (
    Convert,
    DigitalBornMathAugment,
    Invert,
    RandomPadCropJitter,
    RandomResolutionDegradation,
    ToImageTensor,
)


def test_invert():
    t = Invert()
    x = Image.new("L", (30, 40), color=0)
    y = t(x)
    assert y.size == x.size
    assert y.mode == x.mode
    y = np.asarray(y)
    assert y.max() == 255
    assert y.min() == 255


def test_convert_greyscale():
    t = Convert(mode="L")
    x = Image.new("RGB", (30, 40), color=(127, 127, 127))
    y = t(x)
    assert y.size == x.size
    assert y.mode == "L"
    y = np.asarray(y)
    assert y.max(), 127
    assert y.min(), 127


def test_convert_rgba():
    t = Convert(mode="RGBA")
    x = Image.new("L", (30, 40), color=127)
    y = t(x)
    assert y.size == x.size
    assert y.mode == "RGBA"
    y = np.asarray(y)
    # Alpha channel
    assert y[:, :, -1].max(), 255
    assert y[:, :, -1].min(), 255
    # Color channels
    assert y[:, :, :3].max(), 127
    assert y[:, :, :3].min(), 127


@pytest.mark.parametrize("w", [5, 6, 11, 12])
@pytest.mark.parametrize("h", [5, 6, 11, 12])
@pytest.mark.parametrize("fw", [2, 3, 10, 11])
def test_resize_transform_fixed_width(w, h, fw):
    img = Image.new("RGB", size=(w, h))
    out = ToImageTensor.resize_transform(img, fw=fw)
    nw, nh = out.size
    assert nw == fw
    assert nh in (op(h * fw / w) for op in (math.floor, math.ceil))


@pytest.mark.parametrize("w", [5, 6, 11, 12])
@pytest.mark.parametrize("h", [5, 6, 11, 12])
@pytest.mark.parametrize("fh", [2, 3, 10, 11])
def test_resize_transform_fixed_height(w, h, fh):
    img = Image.new("L", size=(w, h))
    out = ToImageTensor.resize_transform(img, fh=fh)
    nw, nh = out.size
    assert nw in (op(w * fh / h) for op in (math.floor, math.ceil))
    assert nh == fh


@pytest.mark.parametrize("w", [5, 6, 11, 12])
@pytest.mark.parametrize("h", [5, 6, 11, 12])
@pytest.mark.parametrize("fh", [2, 3, 10, 11])
@pytest.mark.parametrize("fw", [2, 3, 10, 11])
def test_resize_transform(w, h, fw, fh):
    img = Image.new("L", size=(w, h))
    out = ToImageTensor.resize_transform(img, fw=fw, fh=fh)
    assert out.size == (fw, fh)


@pytest.mark.parametrize("w", [5, 6, 11, 12])
@pytest.mark.parametrize("h", [5, 6, 11, 12])
@pytest.mark.parametrize("mw", [2, 3, 10, 11])
def test_pad_transform_minimum_width(w, h, mw):
    img = Image.new("L", size=(w, h))
    out = ToImageTensor.pad_transform(img, mw=mw)
    assert out.size == (max(w, mw), h)


@pytest.mark.parametrize("w", [5, 6, 11, 12])
@pytest.mark.parametrize("h", [5, 6, 11, 12])
@pytest.mark.parametrize("mh", [2, 3, 10, 11])
def test_pad_transform_minimum_height(w, h, mh):
    img = Image.new("L", size=(w, h))
    out = ToImageTensor.pad_transform(img, mh=mh)
    assert out.size == (w, max(h, mh))


@pytest.mark.parametrize("w", [5, 6, 11, 12])
@pytest.mark.parametrize("h", [5, 6, 11, 12])
@pytest.mark.parametrize("mw", [2, 3, 10, 11])
@pytest.mark.parametrize("mh", [2, 3, 10, 11])
def test_pad_transform(w, h, mw, mh):
    img = Image.new("L", size=(w, h))
    out = ToImageTensor.pad_transform(img, mw=mw, mh=mh)
    assert out.size == (max(w, mw), max(h, mh))


def test_to_image_tensor():
    img = Image.new("L", size=(456, 123))  # color = 0, 255 after inverting

    class Foo:
        def __call__(self, x):
            return x

        def __repr__(self):
            return "Foo()"

    transform = ToImageTensor(
        invert=True,
        random_transform=Foo(),
        fixed_height=100,
        min_height=123,
        min_width=456,
        pad_color=255,
    )
    img = transform(img)
    assert isinstance(img, torch.Tensor)
    assert list(img.size()) == [1, 123, 456]
    assert img.eq(1).all()
    assert str(transform) == (
        "ToImageTensor(\n"
        "  vision.Convert(mode=L),\n"
        "  vision.Invert(),\n"
        "  Foo(),\n"
        "  vision.resize_transform(),\n"
        "  vision.pad_transform(),\n"
        "  ToTensor()\n"
        ")"
    )


def test_random_resolution_degradation_preserves_size_and_mode():
    transform = RandomResolutionDegradation()
    img = Image.new("L", size=(128, 64), color=255)
    out = transform(img)
    assert isinstance(out, Image.Image)
    assert out.mode == img.mode
    assert out.size == img.size


def test_random_pad_crop_jitter_preserves_size_and_mode():
    transform = RandomPadCropJitter(fillcolor=255)
    img = Image.new("L", size=(128, 64), color=255)
    out = transform(img)
    assert isinstance(out, Image.Image)
    assert out.mode == img.mode
    assert out.size == img.size


def test_digital_born_math_augment_preserves_size_and_mode():
    transform = DigitalBornMathAugment(fillcolor=255)
    img = Image.new("L", size=(128, 64), color=255)
    out = transform(img)
    assert isinstance(out, Image.Image)
    assert out.mode == img.mode
    assert out.size == img.size


def test_to_image_tensor_with_digital_born_math_augment_repr():
    transform = ToImageTensor(
        invert=True,
        random_transform=DigitalBornMathAugment(fillcolor=255),
    )
    expected = (
        "ToImageTensor(\n"
        "  vision.Convert(mode=L),\n"
        "  vision.Invert(),\n"
        "  vision.DigitalBornMathAugment(\n"
        "  vision.RandomResolutionDegradation(min_scale=0.55, max_scale=0.9),\n"
        "  vision.RandomMorphology(Dilate|Erode, p=0.25),\n"
        "  vision.RandomBetaAffine(max_offset_ratio=0.05, alpha=3, beta=3, fillcolor=255),\n"
        "  vision.RandomPadCropJitter(max_pad_ratio=0.08, max_crop_ratio=0.03)\n"
        "),\n"
        "  ToTensor()\n"
        ")"
    )
    assert str(transform) == expected
