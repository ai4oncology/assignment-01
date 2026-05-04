"""
ML4Health 2026 -- WBC Multiple Instance Learning  (Section C of notebook.py)
================================================================================
Dataset : WBC Image Dataset (segmentation_WBC, Dataset 1)

Setup
-----
The WBC dataset stores one cell per image. To play out a MIL scenario we
*synthesise* slide-like "bags" of K cell images and ask one bag-level question:

    Does this slide contain at least one eosinophil (the rare class)?

This is the classical MIL assumption (`bag positive iff any-instance positive`)
and mirrors a real haematology workflow where a clinician screens a slide for
abnormal cells without first labelling each cell individually.

Instructions
------------
- Complete every function marked with TODO. Do NOT change function signatures.
  The three pooling models below correspond to *Part 4 — Three pooling
  strategies* in ``notebook.py``.
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
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Reproducibility (do not modify)
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Class index of eosinophil after the 0..3 remap used in `classification.py`.
EOSINOPHIL_LABEL: int = 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ===========================================================================
# Bag construction (provided)
# ===========================================================================

def make_bags(
    image_ids: list[int],
    labels:    list[int],
    k:         int = 5,
    n_bags:    int = 400,
    seed:      int = 42,
    eosinophil_label: int = EOSINOPHIL_LABEL,
) -> list[tuple[list[int], int]]:
    """Synthesise `n_bags` slide-like bags from the cell pool.

    Each bag has exactly `k` cells. Half the bags are negative (no eosinophil)
    and half are positive (at least one eosinophil). Sampling is *with*
    replacement at the bag level (the same cell can appear in multiple bags).

    Returns
    -------
    A shuffled list of (cell_id_list, bag_label) where bag_label is 0 or 1.
    """
    rng = np.random.default_rng(seed)
    eos_pool   = [int(i) for i, l in zip(image_ids, labels) if l == eosinophil_label]
    other_pool = [int(i) for i, l in zip(image_ids, labels) if l != eosinophil_label]
    if not eos_pool:
        raise ValueError("no eosinophil instances found")
    if len(other_pool) < k:
        raise ValueError("not enough non-eosinophil instances to fill a bag")

    bags: list[tuple[list[int], int]] = []
    n_neg = n_bags // 2
    n_pos = n_bags - n_neg

    for _ in range(n_neg):
        ids = rng.choice(other_pool, size=k, replace=False).tolist()
        bags.append(([int(x) for x in ids], 0))
    for _ in range(n_pos):
        eos = int(rng.choice(eos_pool))
        rest = rng.choice(other_pool, size=k - 1, replace=False).tolist()
        bag_ids = [eos] + [int(x) for x in rest]
        rng.shuffle(bag_ids)
        bags.append((bag_ids, 1))

    rng.shuffle(bags)
    return bags


# ===========================================================================
# Frozen ResNet18 feature extractor (provided)
# ===========================================================================

def extract_features(
    image_dir: Path,
    image_ids: list[int],
    device:    torch.device,
    image_size: int = 224,
) -> dict[int, torch.Tensor]:
    """Run frozen ImageNet-pretrained ResNet18 on every image and cache the 512-d feature."""
    weights  = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    image_dir = Path(image_dir)
    feats: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for img_id in image_ids:
            img = Image.open(image_dir / f"{int(img_id):03d}.bmp").convert("RGB")
            x   = tfm(img).unsqueeze(0).to(device)
            f   = backbone(x).squeeze(0).cpu()
            feats[int(img_id)] = f
    return feats


# ===========================================================================
# Bag Dataset (provided)
# ===========================================================================

class BagDataset(Dataset):
    """Wraps a list of bags + a feature cache.

    Each item returned is a (features[K, D], bag_label) pair.

    To recover the cell IDs that make up a bag for visualisation, read
    `dataset.bags[i]` directly -- __getitem__ only returns model-ready tensors.
    """

    def __init__(self, bags: list[tuple[list[int], int]], features: dict[int, torch.Tensor]) -> None:
        self.bags     = bags
        self.features = features

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, label = self.bags[idx]
        feats = torch.stack([self.features[int(i)] for i in ids])    # [K, D]
        return feats, torch.tensor(label, dtype=torch.long)


# ===========================================================================
# Part 4 -- Pooling-based MIL classifiers (mean / max / attention)
# ===========================================================================

class MeanPoolMIL(nn.Module):
    """Pool instance features with a simple arithmetic mean, then classify the bag.

    Forward
    -------
    x : Tensor of shape [B, K, D] -- B bags, K instances per bag, D feature dim.

    Returns
    -------
    (logits[B, 2], attention_weights[B, K] or None)
        attention_weights is None for mean pooling (every instance contributes equally).
    """

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        # TODO: define a Linear classifier from feature_dim to 2 classes.
        raise NotImplementedError("MIL exercise: implement MeanPoolMIL.__init__")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: average across the K-dimension to get [B, D], then classify.
        # Return (logits[B, 2], None) -- mean pooling has no attention weights.
        raise NotImplementedError("MIL exercise: implement MeanPoolMIL.forward")


class MaxPoolMIL(nn.Module):
    """Pool by element-wise max across instances, then classify."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        # TODO: define a Linear classifier from feature_dim to 2 classes.
        raise NotImplementedError("MIL exercise: implement MaxPoolMIL.__init__")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: take element-wise max across the K-dimension to get [B, D],
        # then classify. Return (logits[B, 2], None).
        raise NotImplementedError("MIL exercise: implement MaxPoolMIL.forward")


class AttentionPoolMIL(nn.Module):
    """Gated Attention-MIL pooling (Ilse, Tomczak, Welling, 2018).

    Two parallel branches score each instance: a tanh branch carries content,
    a sigmoid branch gates it; their elementwise product is projected to a
    scalar and softmaxed over the bag to give attention weights. The bag
    representation is the weighted sum of the instance features. Attention
    weights are returned for interpretability.

    Architecture
    ------------
        V    : Linear(D -> H), activation Tanh        (content branch)
        U    : Linear(D -> H), activation Sigmoid     (gating branch)
        w    : Linear(H -> 1)                         (score projection)
        e_k  = w( tanh(V h_k) * sigmoid(U h_k) )      (elementwise product)
        a_k  = softmax_k(e_k);   z = sum_k a_k h_k
        head : Linear(D -> 2)
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        # TODO: define the gated-attention branches self.V (Linear D->H) and
        # self.U (Linear D->H), the score projection self.w (Linear H->1), and
        # the bag classifier self.classifier (Linear D->2) per the architecture
        # above. Activations (Tanh / Sigmoid) are applied in forward().
        raise NotImplementedError("MIL exercise: implement AttentionPoolMIL.__init__")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: compute the gated score per instance
        #   e_k = w( tanh(V h_k) * sigmoid(U h_k) ),
        # softmax over the K dimension to get attention weights, take the
        # weighted sum across instances, then classify. Return
        # (logits[B, 2], attention_weights[B, K]).
        raise NotImplementedError("MIL exercise: implement AttentionPoolMIL.forward")


# ===========================================================================
# Train / eval loop
# ===========================================================================

def run_mil_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, list[int], list[float]]:
    """Run one full pass over a bag-level DataLoader (train or eval).

    Returns
    -------
    (avg_loss, bag_accuracy, all_targets, all_pos_probs)
        - all_targets  : list of bag labels (0/1)
        - all_pos_probs: P(positive bag) from softmax(logits)[:, 1]; suitable for AUROC.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_targets:  list[int]   = []
    all_pos_probs: list[float] = []

    for feats, labels in loader:
        feats  = feats.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        logits, _ = model(feats)
        loss      = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        probs = logits.softmax(dim=1)[:, 1]
        bs    = labels.size(0)
        total_loss    += loss.item() * bs
        total_correct += (preds == labels).sum().item()
        total_samples += bs
        all_targets.extend(labels.detach().cpu().tolist())
        all_pos_probs.extend(probs.detach().cpu().tolist())

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc, all_targets, all_pos_probs
