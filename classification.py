"""
ML4Health 2026 -- WBC Classification  (Section A of notebook.py)
====================================================================
Dataset : WBC Image Dataset (segmentation_WBC, Dataset 1)
Classes : neutrophil | lymphocyte | monocyte | eosinophil
          (basophil is dropped -- only one example is available)

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
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Reproducibility  (do not modify)
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Canonical class order used throughout the assignment.
# Index in this list == integer label fed to the model.
WBC_CLASS_NAMES: list[str] = ["neutrophil", "lymphocyte", "monocyte", "eosinophil"]


# ===========================================================================
# Exercise 6 -- Evaluation metrics
# ===========================================================================

def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1-score for a single class (one-vs-rest).

    Parameters
    ----------
    tp : true positives
    fp : false positives
    fn : false negatives

    Returns
    -------
    (precision, recall, f1)  -- each rounded to 4 decimal places.

    Edge cases
    ----------
    If a denominator is zero (e.g. no predicted positives), return 0.0 for that
    metric instead of raising.
    """
    # TODO: implement precision, recall, and F1.
    raise NotImplementedError("Exercise 6: implement compute_metrics")


# ===========================================================================
# Exercise 17 -- Matthews Correlation Coefficient (MCC)
# ===========================================================================

def compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute the Matthews Correlation Coefficient for a single class.

    The MCC uses *all four* cells of the confusion matrix and takes values in
    [-1, +1]: +1 = perfect, 0 = random, -1 = inverted. F1 ignores the True
    Negative count entirely; MCC does not.

        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Parameters
    ----------
    tp, tn, fp, fn : confusion-matrix cell counts.

    Returns
    -------
    float -- MCC rounded to 4 decimal places.

    Edge cases
    ----------
    If any factor under the square root is zero, the formula is undefined; by
    the same convention we use elsewhere in this assignment, return 0.0.
    """
    # TODO: implement MCC. Hint: math.sqrt for the denominator.
    raise NotImplementedError("Exercise 17: implement compute_mcc")


# ===========================================================================
# Exercise 12 -- Class imbalance (weights)
# ===========================================================================

def compute_class_weights(class_counts: dict[str, int]) -> dict[str, float]:
    """Compute inverse-frequency class weights.

    Each class's weight should equal the *mean* count across classes divided by
    that class's count, so balanced classes get weight 1.0 and the rarest class
    gets the highest weight.

    Parameters
    ----------
    class_counts : dict mapping class name -> number of training samples.

    Returns
    -------
    dict mapping class name -> weight (rounded to 4 decimal places).
    """
    # TODO: implement inverse-frequency class weights.
    raise NotImplementedError("Exercise 12: implement compute_class_weights")


# ===========================================================================
# Exercise 3 -- Custom WBC Dataset and DataLoaders
# ===========================================================================

class WBCDataset(Dataset):
    """A torch Dataset for the WBC image collection.

    The raw dataset stores all images in a single folder; the labels are stored
    in a separate CSV file. This class wraps a *pre-filtered* DataFrame of rows
    (one per sample) and loads the corresponding image on demand.

    Parameters
    ----------
    image_dir : folder containing the .bmp images (e.g. 'segmentation_WBC/Dataset 1')
    records   : pandas DataFrame with two columns:
                  - 'image_id' : int, 1-indexed image ID
                  - 'label'    : int, class label already remapped to 0..C-1
                Each row is one sample exposed by this Dataset.
    transform : torchvision transform applied to the loaded PIL image.

    Notes
    -----
    Image filenames follow the pattern '{id:03d}.bmp' (e.g. '001.bmp').
    Convert each image to RGB before applying the transform.
    """

    def __init__(self, image_dir: Path, records: pd.DataFrame, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # TODO: load the image at row `idx`, apply the transform, and return
        # (image_tensor, label_int). Filenames are formatted as f"{id:03d}.bmp".
        # Convert each image to RGB before applying the transform.
        raise NotImplementedError("Exercise 3: implement WBCDataset.__getitem__")


# ===========================================================================
# Exercise 3 (continued) -- DataLoaders
# ===========================================================================

def build_dataloaders(
    image_dir: Path,
    labels_csv: Path,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train, validation, and test DataLoaders from the WBC dataset.

    The CSV lists every image and its integer class label (1..5). The basophil
    class (label 5) is dropped because there is only a single example. The
    remaining four classes are remapped to 0..3 in this order:
        0 = neutrophil, 1 = lymphocyte, 2 = monocyte, 3 = eosinophil.
    (See WBC_CLASS_NAMES at the top of this file.)

    A *stratified* random split is produced: each class is split independently
    so the train / val / test ratios are preserved per class.

    Augmentation: RandomHorizontalFlip and RandomRotation(15) are applied to
    the training set only; validation and test sets use deterministic transforms.

    Parameters
    ----------
    image_dir     : folder containing the .bmp images
    labels_csv    : CSV file with one row per image (image ID, label)
    val_fraction  : fraction of all images reserved for validation
    test_fraction : fraction of all images reserved for the held-out test set
    batch_size    : mini-batch size for all loaders
    seed          : random seed for the stratified split

    Returns
    -------
    (train_loader, val_loader, test_loader, class_names)

    where class_names is WBC_CLASS_NAMES (so class_names[i] is the human-readable
    name for label i).

    Hints
    -----
    1. Read the CSV, name its columns ``image_id`` and ``label``, drop label==5,
       then remap labels 1..4 -> 0..3.
    2. Define two ``transforms.Compose`` pipelines: a training one with resize +
       RandomHorizontalFlip + RandomRotation(15) + ToTensor, and an eval one
       without augmentation. Both should resize to (224, 224).
    3. Stratified split: for each class, shuffle rows with a seeded RNG, then
       slice off ``n * test_fraction`` for test, the next ``n * val_fraction``
       for val, and the rest for train.
    4. Wrap each split in a ``WBCDataset`` and a ``DataLoader``. Only the train
       loader should ``shuffle=True``.
    """
    # TODO: implement the stratified split and build the three DataLoaders.
    raise NotImplementedError("Exercise 3: implement build_dataloaders")


# ===========================================================================
# Exercise 4 -- Simple CNN baseline
# ===========================================================================

class SimpleCNN(nn.Module):
    """Minimal 2-block CNN classifier.

    Architecture to implement
    -------------------------
    block1 : Conv2d(3->16, kernel=3, padding=1) -> ReLU -> MaxPool2d(2)
    block2 : Conv2d(16->32, kernel=3, padding=1) -> ReLU -> MaxPool2d(2)
    head   : AdaptiveAvgPool2d(1) -> Flatten -> Linear(32, num_classes)

    Use nn.Sequential for each block and the head.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        # TODO: define self.block1, self.block2, and self.head per the docstring.
        raise NotImplementedError("Exercise 4: implement SimpleCNN.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: pass x through block1 -> block2 -> head and return the logits.
        raise NotImplementedError("Exercise 4: implement SimpleCNN.forward")


# ===========================================================================
# Pretrained ResNet18  (provided -- no changes needed)
# ===========================================================================

def build_resnet18(num_classes: int = 4) -> nn.Module:
    """Return a ResNet18 pretrained on ImageNet with a new classification head."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ===========================================================================
# Exercise 5 -- Training and evaluation loop
# ===========================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run one full pass over a DataLoader (training or evaluation).

    Pass an optimizer for training mode; pass None for evaluation mode.

    Parameters
    ----------
    model     : the neural network
    loader    : DataLoader for the split to process
    criterion : loss function (e.g. CrossEntropyLoss)
    optimizer : optimizer for training; None for validation / test
    device    : torch device (cpu or cuda)

    Returns
    -------
    (avg_loss, avg_accuracy, all_targets, all_preds)
        - avg_loss     : sample-weighted mean of the per-batch loss
        - avg_accuracy : fraction of correctly classified samples
        - all_targets  : np.ndarray of all ground-truth labels (concatenated)
        - all_preds    : np.ndarray of all predicted labels (concatenated)

    Hints
    -----
    1. Decide between training and eval mode based on ``optimizer``: pass an
       optimizer for training, or ``None`` for evaluation.
    2. In training mode: zero gradients, forward, compute loss, backward, step.
       In eval mode: just forward + compute loss.
    3. Predict by taking ``argmax`` over the class dimension.
    4. Accumulate the per-batch loss weighted by batch size, then divide by the
       total number of samples at the end.
    """
    # TODO: implement one full training/evaluation pass over the loader.
    raise NotImplementedError("Exercise 5: implement run_epoch")
