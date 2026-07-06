from typing import Callable, Optional

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps

from laia.data.transforms.vision.random_beta_affine import RandomBetaAffine
from laia.data.transforms.vision.random_beta_morphology import Dilate, Erode


class Invert:
    """Invert the colors of a PIL image with the given probability."""

    def __call__(self, img: Image) -> Image:
        return ImageOps.invert(img)

    def __repr__(self) -> str:
        return f"vision.{self.__class__.__name__}()"


class Convert:
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode: str) -> None:
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode

    def __call__(self, img: Image) -> Image:
        return img.convert(self.mode)

    def __repr__(self) -> str:
        return f"vision.{self.__class__.__name__}(mode={self.mode})"


class ToImageTensor:
    def __init__(
        self,
        invert: bool = True,
        mode: str = "L",
        random_transform: Optional[Callable] = None,
        fixed_height: Optional[int] = None,
        fixed_width: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        pad_color: int = 0,
    ) -> None:
        assert mode in ("L", "RGB", "RGBA")
        assert fixed_height is None or fixed_height > 0
        assert fixed_width is None or fixed_width > 0
        assert min_height is None or min_height > 0
        assert min_width is None or min_width > 0
        self.convert_transform = Convert(mode)
        self.invert_transform = Invert() if invert else None
        self.random_transform = random_transform if random_transform else None
        self.resize_transform = (
            self.resize_transform if fixed_width or fixed_height else None
        )
        self.fixed_width = fixed_width
        self.fixed_height = fixed_height
        self.pad_transform = self.pad_transform if min_width or min_height else None
        self.min_width = min_width
        self.min_height = min_height
        self.pad_color = pad_color
        self.tensor_transform = ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # W x H
        assert isinstance(img, Image.Image)
        img = self.convert_transform(img)
        if self.invert_transform:
            img = self.invert_transform(img)
        if self.random_transform:
            img = self.random_transform(img)
        if self.resize_transform:
            img = self.resize_transform(img, fw=self.fixed_width, fh=self.fixed_height)
        if self.pad_transform:
            img = self.pad_transform(
                img, mw=self.min_width, mh=self.min_height, pad_color=self.pad_color
            )
        img = self.tensor_transform(img)
        # C x H x W
        return img

    @staticmethod
    def resize_transform(
        img: Image.Image,
        fw: Optional[int] = None,
        fh: Optional[int] = None,
        resample: int = Image.ANTIALIAS,
    ) -> Image.Image:
        if fw and fh:
            # resize to a fixed size
            return img.resize((fw, fh), resample=resample)
        w, h = img.size
        if fw is None:
            fw = w
        if fh is None:
            fh = h
        if fw > w:
            # upscale to a fixed width
            return img.resize((fw, h * fw // w), resample=resample)
        if fh > h:
            # upscale to a fixed height
            return img.resize((w * fh // h, fh), resample=resample)
        # downscale
        img.thumbnail((fw, fh), resample=resample)
        return img

    @staticmethod
    def pad_transform(
        img: Image.Image,
        mw: Optional[int] = None,
        mh: Optional[int] = None,
        pad_color: int = 0,
    ) -> Image.Image:
        w, h = img.size
        nw = w if mw is None or w > mw else mw
        nh = h if mh is None or h > mh else mh
        # maybe an extra pixel to the bottom
        a, b = divmod(nw - w, 2)
        left, right = a, a + b
        # maybe an extra pixel to the right
        a, b = divmod(nh - h, 2)
        top, bottom = a, a + b
        return torchvision.transforms.functional.pad(
            img, (left, top, right, bottom), fill=pad_color
        )

    def __repr__(self) -> str:
        invert_txt = f"{self.invert_transform},\n  " if self.invert_transform else ""
        random_txt = f"{self.random_transform},\n  " if self.random_transform else ""
        return (
            f"{self.__class__.__name__}(\n  "
            f"{self.convert_transform},\n  "
            f"{invert_txt}"
            f"{random_txt}"
            + ("vision.resize_transform(),\n  " if self.resize_transform else "")
            + ("vision.pad_transform(),\n  " if self.pad_transform else "")
            + f"{self.tensor_transform}\n)"
        )


class RandomResolutionDegradation:
    """Downscale and restore an image using random interpolation modes."""

    def __init__(
        self,
        min_scale: float = 0.55,
        max_scale: float = 0.9,
        resamples: Optional[tuple] = None,
    ) -> None:
        assert 0 < min_scale <= max_scale <= 1
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.resamples = resamples or (
            Image.NEAREST,
            Image.BILINEAR,
            Image.BICUBIC,
            Image.LANCZOS,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < 2 or h < 2:
            return img
        scale = np.random.uniform(self.min_scale, self.max_scale)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        down_resample = self.resamples[np.random.randint(len(self.resamples))]
        up_resample = self.resamples[np.random.randint(len(self.resamples))]
        img = img.resize((nw, nh), resample=down_resample)
        return img.resize((w, h), resample=up_resample)

    def __repr__(self) -> str:
        return (
            f"vision.{self.__class__.__name__}("
            f"min_scale={self.min_scale}, max_scale={self.max_scale})"
        )


class RandomPadCropJitter:
    """Apply mild padding/cropping jitter and restore the original size."""

    def __init__(
        self,
        max_pad_ratio: float = 0.08,
        max_crop_ratio: float = 0.03,
        fillcolor: int = 0,
        resample: int = Image.BILINEAR,
    ) -> None:
        assert max_pad_ratio >= 0
        assert max_crop_ratio >= 0
        self.max_pad_ratio = max_pad_ratio
        self.max_crop_ratio = max_crop_ratio
        self.fillcolor = fillcolor
        self.resample = resample

    @staticmethod
    def _sample_delta(size: int, max_crop_ratio: float, max_pad_ratio: float) -> int:
        max_crop = int(round(size * max_crop_ratio))
        max_pad = int(round(size * max_pad_ratio))
        return int(np.random.randint(-max_crop, max_pad + 1))

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < 2 or h < 2:
            return img

        left_delta = self._sample_delta(w, self.max_crop_ratio, self.max_pad_ratio)
        right_delta = self._sample_delta(w, self.max_crop_ratio, self.max_pad_ratio)
        top_delta = self._sample_delta(h, self.max_crop_ratio, self.max_pad_ratio)
        bottom_delta = self._sample_delta(h, self.max_crop_ratio, self.max_pad_ratio)

        crop_left = max(0, -left_delta)
        crop_right = max(0, -right_delta)
        crop_top = max(0, -top_delta)
        crop_bottom = max(0, -bottom_delta)

        right = max(crop_left + 1, w - crop_right)
        bottom = max(crop_top + 1, h - crop_bottom)
        img = img.crop((crop_left, crop_top, right, bottom))

        pad_left = max(0, left_delta)
        pad_right = max(0, right_delta)
        pad_top = max(0, top_delta)
        pad_bottom = max(0, bottom_delta)
        if pad_left or pad_right or pad_top or pad_bottom:
            img = ImageOps.expand(
                img,
                border=(pad_left, pad_top, pad_right, pad_bottom),
                fill=self.fillcolor,
            )

        return img.resize((w, h), resample=self.resample)

    def __repr__(self) -> str:
        return (
            f"vision.{self.__class__.__name__}("
            f"max_pad_ratio={self.max_pad_ratio}, "
            f"max_crop_ratio={self.max_crop_ratio})"
        )


class DigitalBornMathAugment:
    """Training-time augmentation for digitally born mathematical expressions."""

    def __init__(
        self,
        resolution_prob: float = 0.9,
        morphology_prob: float = 0.25,
        affine_prob: float = 0.8,
        pad_crop_prob: float = 0.5,
        fillcolor: int = 0,
    ) -> None:
        assert 0 <= resolution_prob <= 1
        assert 0 <= morphology_prob <= 1
        assert 0 <= affine_prob <= 1
        assert 0 <= pad_crop_prob <= 1
        self.resolution_prob = resolution_prob
        self.morphology_prob = morphology_prob
        self.affine_prob = affine_prob
        self.pad_crop_prob = pad_crop_prob
        self.resolution_transform = RandomResolutionDegradation()
        self.morphology_transforms = (
            Dilate(filter_size_min=3, filter_size_max=3),
            Erode(filter_size_min=3, filter_size_max=3),
        )
        self.affine_transform = RandomBetaAffine(
            max_offset_ratio=0.05,
            alpha=3,
            beta=3,
            fillcolor=fillcolor,
        )
        self.pad_crop_transform = RandomPadCropJitter(fillcolor=fillcolor)

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.random() < self.resolution_prob:
            img = self.resolution_transform(img)
        if np.random.random() < self.morphology_prob:
            transform = self.morphology_transforms[
                np.random.randint(len(self.morphology_transforms))
            ]
            img = transform(img)
        if np.random.random() < self.affine_prob:
            img = self.affine_transform(img)
        if np.random.random() < self.pad_crop_prob:
            img = self.pad_crop_transform(img)
        return img

    def __repr__(self) -> str:
        return (
            f"vision.{self.__class__.__name__}(\n"
            f"  {self.resolution_transform},\n"
            f"  vision.RandomMorphology(Dilate|Erode, p={self.morphology_prob}),\n"
            f"  {self.affine_transform},\n"
            f"  {self.pad_crop_transform}\n"
            f")"
        )


ToTensor = torchvision.transforms.transforms.ToTensor
