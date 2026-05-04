"""
ML4Health 2026 -- WBC Semantic Segmentation  (Section B of notebook.py)
===========================================================================
Dataset : WBC Image Dataset (segmentation_WBC, Dataset 1)
Classes : background (0), cytoplasm (1), nucleus (2)

Mask format
-----------
The masks are 8-bit grayscale PNGs with exactly three pixel values:
    0   -> background (incl. red blood cells)
    128 -> cytoplasm
    255 -> nucleus
We remap them to integer class indices 0, 1, 2.

Instructions
------------
- Complete every function marked with TODO. Do NOT change function signatures.
  The exercise number in each section comment matches the numbering in
  ``notebook.py``.
- Multiple-choice / numeric questions are answered with widgets in
  ``notebook.py``; their values are auto-saved to ``submission.json``.
- Run ``pytest tests/ -v`` locally to check both your implementations and
  the answers in ``submission.json`` before pushing.
- Push to ``main`` only for final submission; save work-in-progress on a branch.
- Record issues and reflections in ``experiences.md``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Reproducibility  (do not modify)
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Canonical class order. Index in this list == integer class label.
SEG_CLASS_NAMES: list[str] = ["background", "cytoplasm", "nucleus"]

# Pixel value in the raw PNG mask for each class index.
MASK_VALUES: tuple[int, int, int] = (0, 128, 255)

# ImageNet statistics -- the pretrained encoder expects inputs normalised this way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _decode_mask(mask_array: np.ndarray) -> np.ndarray:
    """Map raw PNG values {0, 128, 255} to class indices {0, 1, 2}."""
    out = np.zeros_like(mask_array, dtype=np.int64)
    out[mask_array == 128] = 1
    out[mask_array == 255] = 2
    return out


# ===========================================================================
# Exercise 4 -- Dice score
# ===========================================================================

def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
) -> tuple[float, list[float]]:
    """Compute the multi-class Dice coefficient.

    The Dice score for class ``c`` is

        Dice_c = (2 * TP_c) / (2 * TP_c + FP_c + FN_c)

    where TP/FP/FN are pixel counts computed in a one-vs-rest fashion.

    Parameters
    ----------
    pred         : LongTensor of predicted class indices (any shape).
    target       : LongTensor of ground-truth class indices, same shape as `pred`.
    num_classes  : number of classes.

    Returns
    -------
    (mean_dice, [dice_class_0, dice_class_1, ...])  -- each rounded to 4 decimals.
    A class with no ground-truth and no predictions returns Dice = 1.0 by convention.

    Hints
    -----
    1. Flatten both ``pred`` and ``target`` so the rest of the function is
       agnostic to the original shape.
    2. For each class index ``c``, build boolean masks ``p = (pred == c)`` and
       ``t = (target == c)``, then count TP, FP, and FN as integer pixel sums.
    3. Watch for the all-zero case: if the denominator is zero return 1.0.
    """
    # TODO: implement the multi-class Dice score.
    raise NotImplementedError("Exercise 4: implement dice_score")


# ===========================================================================
# Exercise 6 -- Pixel-level class imbalance (weights)
# ===========================================================================

def compute_pixel_class_weights(class_pixel_counts: dict[str, int]) -> dict[str, float]:
    """Compute inverse-frequency class weights from *pixel* counts.

    Same formula as `compute_class_weights` in `classification.py`: each weight is the
    mean count divided by that class's count, so balanced classes give weight 1.0
    and the rarest class gets the highest weight.

    Parameters
    ----------
    class_pixel_counts : dict mapping class name -> total pixel count over the
                         training split.

    Returns
    -------
    dict mapping class name -> weight (rounded to 4 decimal places).
    """
    # TODO: implement inverse-frequency weights from pixel counts.
    raise NotImplementedError("Exercise 6: implement compute_pixel_class_weights")


# ===========================================================================
# Exercise 8 -- Classical baseline: Otsu's method
# ===========================================================================

def otsu_threshold(image: np.ndarray) -> int:
    """Otsu's automatic thresholding (Otsu, 1979).

    Given an 8-bit grayscale image, find the threshold ``t`` in ``[0, 255]``
    that maximises the inter-class variance:

        sigma_b^2(t) = w_0(t) * w_1(t) * (mu_0(t) - mu_1(t))^2

    where:
        w_0(t) = fraction of pixels with intensity <  t  (class "below")
        w_1(t) = fraction of pixels with intensity >= t  (class "above")
        mu_0(t), mu_1(t) = mean intensities of the two classes

    Equivalently, picking ``t`` that minimises within-class variance.

    Parameters
    ----------
    image : np.ndarray of dtype uint8 (any 2D shape).

    Returns
    -------
    int in [0, 255] -- the optimal threshold. Pixels strictly less than the
    threshold belong to the "below" class.

    Hints
    -----
    1. Build a 256-bin intensity histogram with ``np.histogram``.
    2. For each candidate threshold ``t`` in 1..255, compute the class weights
       and means from the histogram, then the inter-class variance.
    3. Skip thresholds where one of the two classes is empty (variance is
       undefined). Track the threshold with the highest inter-class variance.
    """
    # TODO: implement Otsu's threshold selection.
    raise NotImplementedError("Exercise 8: implement otsu_threshold")


# ===========================================================================
# Exercise 9 -- Custom segmentation Dataset and DataLoaders
# ===========================================================================

class WBCSegDataset(Dataset):
    """Paired image + mask dataset for WBC semantic segmentation.

    Each item is a `(image_tensor[3,H,W], mask_tensor[H,W])` pair where the mask
    contains integer class indices in {0, 1, 2}.

    Parameters
    ----------
    image_dir  : folder containing both .bmp images and .png masks named NNN.{bmp,png}
    image_ids  : iterable of integer image IDs to expose via this dataset.
    image_size : side length to resize both image and mask to (square).
                 Bilinear for the image, **nearest** for the mask (nearest is
                 critical -- bilinear would create non-{0,128,255} values).
    augment    : if True, apply paired horizontal flip with probability 0.5.

    Notes
    -----
    The image is normalised with ImageNet mean/std because the pretrained encoder
    used by ``build_pretrained_unet`` (below) expects that distribution.
    """

    def __init__(
        self,
        image_dir: Path,
        image_ids: list[int],
        image_size: int = 128,
        augment: bool = False,
    ) -> None:
        self.image_dir  = Path(image_dir)
        self.image_ids  = list(image_ids)
        self.image_size = image_size
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: load image+mask, resize each with the correct interpolation
        # (bilinear for the image, nearest for the mask), normalise the image
        # with ImageNet stats, decode the mask with `_decode_mask`, and apply
        # paired horizontal flip (probability 0.5) when self.augment is True.
        raise NotImplementedError("Exercise 9: implement WBCSegDataset.__getitem__")


# ===========================================================================
# Exercise 9 (continued) -- DataLoaders
# ===========================================================================

def build_seg_dataloaders(
    image_dir: Path,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    image_size: int = 128,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train / val / test DataLoaders for the WBC segmentation task.

    Splitting is random at the *image* level (each image already contains all
    three classes, so per-class stratification is not meaningful here).

    Augmentation: paired horizontal flip on the training set only.

    Returns
    -------
    (train_loader, val_loader, test_loader, class_names)

    Hints
    -----
    1. Discover image IDs by globbing ``*.bmp`` in ``image_dir`` and parsing
       the integer stems.
    2. Shuffle with a seeded ``np.random.default_rng(seed)``, then split off
       ``n * test_fraction`` for test, ``n * val_fraction`` for val, and the
       rest for train.
    3. Wrap each split in a ``WBCSegDataset`` (only the training split should
       use ``augment=True``) and a ``DataLoader`` (only train should shuffle).
    """
    # TODO: implement the random image-level split and build the loaders.
    raise NotImplementedError("Exercise 9: implement build_seg_dataloaders")


# ===========================================================================
# Exercise 12 -- SmallUNet (from scratch)
# ===========================================================================

class _DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2 -- the basic building block used everywhere."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """A compact UNet with random initialisation.

    Architecture (channel widths quadruple from `base`)
    ---------------------------------------------------
    Encoder  : DoubleConv(3 -> base) -> pool -> DoubleConv(base -> 2b) -> pool ->
               DoubleConv(2b -> 4b) -> pool -> DoubleConv(4b -> 8b)            (bottleneck)
    Decoder  : upsample x2 -> concat skip -> DoubleConv(8b+4b -> 4b)
               upsample x2 -> concat skip -> DoubleConv(4b+2b -> 2b)
               upsample x2 -> concat skip -> DoubleConv(2b+ b ->  b)
    Head     : Conv1x1(b -> num_classes)
    """

    def __init__(self, num_classes: int = 3, base: int = 32) -> None:
        super().__init__()
        # TODO: define encoder blocks (enc1..enc3 + bottleneck), decoder
        # upsamples + DoubleConvs, the 1x1 head, and a MaxPool2d(2). Use the
        # _DoubleConv helper for the conv blocks.
        raise NotImplementedError("Exercise 12: implement SmallUNet.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: encode (with pooling) -> bottleneck -> decode (upsample +
        # concat skip + DoubleConv) -> 1x1 head. Return the per-class logits.
        raise NotImplementedError("Exercise 12: implement SmallUNet.forward")


# ===========================================================================
# Pretrained-encoder UNet  (provided -- no changes needed)
# ===========================================================================

def build_pretrained_unet(num_classes: int = 3) -> nn.Module:
    """Return a UNet whose ResNet18 encoder is pretrained on ImageNet."""
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )


# ===========================================================================
# Exercise 13 -- Training and evaluation loop
# ===========================================================================

def run_epoch_seg(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Run one full pass over a segmentation DataLoader.

    Pass an optimizer for training mode; pass None for evaluation mode.

    Returns
    -------
    (avg_loss, pixel_accuracy, all_targets, all_preds)
        - avg_loss       : float, mean per-pixel CE loss
        - pixel_accuracy : float, fraction of correctly classified pixels
        - all_targets    : LongTensor [N_total_pixels] of ground-truth class ids
        - all_preds      : LongTensor [N_total_pixels] of predicted class ids

    `dice_score(all_preds, all_targets, num_classes=3)` can be called on the
    returned tensors to obtain per-class Dice.

    Hints
    -----
    1. Logits come back as ``[B, C, H, W]``; predictions are ``logits.argmax(dim=1)``.
    2. The criterion (CrossEntropyLoss) expects long-tensor masks of shape
       ``[B, H, W]``.
    3. Accumulate the per-batch loss weighted by pixel count, count correct
       pixels, and append flattened tensors to lists for later concatenation.
    """
    # TODO: implement one full training/evaluation pass over the segmentation loader.
    raise NotImplementedError("Exercise 13: implement run_epoch_seg")
