import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json as _json
    from pathlib import Path as _Path

    _submission_path = _Path("submission.json")

    def submission_default(key, default=None):
        """Return the saved value for `key`, re-reading submission.json each call."""
        if not _submission_path.exists():
            return default
        try:
            data = _json.loads(_submission_path.read_text())
        except _json.JSONDecodeError:
            return default
        return data.get(key, default)

    def submission_radio_default(key, options, default=None):
        saved = submission_default(key, default)
        if saved is None:
            return default
        if saved in options:
            return saved
        for opt_key, opt_value in options.items():
            if opt_value == saved:
                return opt_key
        return default

    return submission_default, submission_radio_default


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment-01: White Blood Cell Image Analysis (Classification, Segmentation, MIL)
    **ML4Health 2026**

    ---

    This notebook includes following Sections:

    1. **Section A — Classification** (`classification.py`): predict the WBC type of a single cell
    2. **Section B — Segmentation** (`segmentation.py`): per-pixel labels (background / cytoplasm / nucleus)
    3. **Section C — Multiple Instance Learning** (`mil.py`): screen synthetic slide bags of cells for the rare eosinophil

    All sections share the same dataset and the same train/val/test split.
    Run cells top-to-bottom to follow the intended narrative arc.

    ### How to work
    - Coding tasks live in **`classification.py`**, **`segmentation.py`**, **`mil.py`**.
      Each function has a `TODO` comment describing the inputs, outputs and expected behavior.
    - Multiple-choice and short-numeric questions are answered with the radio /
      number widgets in this notebook. A hidden cell at the bottom of the
      notebook auto-saves every widget value to **`submission.json`** whenever
      anything changes — so as long as you've run all the cells at least once,
      the file stays in sync. The autograder reads from `submission.json`.
    - Run `pytest tests/ -v` locally to check both your code implementations and
      your widget answers before pushing.
    - Push to `main` only for **final submission**; use a branch to save work in progress.
    - Record difficulties or open questions in `experiences.md`.
    """)
    return


@app.cell
def _():
    # Shared environment setup -- run this cell first.
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from PIL import Image
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        classification_report, confusion_matrix, ConfusionMatrixDisplay,
        roc_auc_score
    )

    from classification import (
        set_seed,
        compute_metrics,
        compute_mcc,
        compute_class_weights,
        build_dataloaders,
        SimpleCNN,
        build_resnet18,
        run_epoch,
        WBC_CLASS_NAMES,
    )
    from segmentation import (
        dice_score,
        compute_pixel_class_weights,
        otsu_threshold,
        build_seg_dataloaders,
        SmallUNet,
        build_pretrained_unet,
        run_epoch_seg,
        SEG_CLASS_NAMES,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    from mil import (
        make_bags,
        extract_features,
        BagDataset,
        MeanPoolMIL,
        MaxPoolMIL,
        AttentionPoolMIL,
        run_mil_epoch,
    )

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT  = Path('segmentation_WBC')
    IMAGES_DIR = DATA_ROOT / 'Dataset 1'
    LABELS_CSV = DATA_ROOT / 'Class Labels of Dataset 1.csv'
    SEG_CMAP = ListedColormap(['#000000', '#888888', '#FF4040'])
    print('Device       :', device)
    print('Images dir   :', IMAGES_DIR)
    print('Labels CSV   :', LABELS_CSV)
    return (
        AttentionPoolMIL,
        BagDataset,
        ConfusionMatrixDisplay,
        DataLoader,
        IMAGENET_MEAN,
        IMAGENET_STD,
        IMAGES_DIR,
        Image,
        LABELS_CSV,
        MaxPoolMIL,
        MeanPoolMIL,
        SEG_CLASS_NAMES,
        SEG_CMAP,
        SimpleCNN,
        SmallUNet,
        WBC_CLASS_NAMES,
        build_dataloaders,
        build_pretrained_unet,
        build_resnet18,
        build_seg_dataloaders,
        classification_report,
        compute_class_weights,
        compute_mcc,
        compute_metrics,
        compute_pixel_class_weights,
        confusion_matrix,
        device,
        dice_score,
        extract_features,
        make_bags,
        nn,
        np,
        otsu_threshold,
        pd,
        plt,
        roc_auc_score,
        run_epoch,
        run_epoch_seg,
        run_mil_epoch,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section A — Cell-type Classification

    All functions and classes you need to implement live in **classification.py**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — Understanding the Problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""For this assignment we work with **Dataset 1** (300 images, 120×120 pixels).
    The dataset contains microscope images of stained white blood cells, each labelled with one
    of five WBC types:
    $$\{\text{neutrophil},\ \text{lymphocyte},\ \text{monocyte},\ \text{eosinophil},\ \text{basophil}\}$$

    The **basophil** class has only a single example, so we drop it and work
    with the remaining four classes (neutrophil, lymphocyte, monocyte,
    eosinophil) with labels remapped to 0..3. You will see this confirmed in
    Part 2 when we load the data.
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 1 — Task framing
    Which formulation correctly describes the supervised learning problem in Section A?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Input x = a single cell-crop image; output y = an integer label in {0, 1, 2, 3}; task = multi-class classification.": "a",
            "Input x = a single cell-crop image; output y = a 4-dimensional one-hot vector; task = multi-label classification.": "b",
            "Input x = a single cell-crop image; output y = an integer label in {0, 1, 2, 3, 4} covering all five WBC types in the original dataset; task = multi-class classification.": "c",
            "Input x = a single cell-crop image; output y = an integer in {0, 1, 2, 3}; task = ordinal regression (WBC types form a natural ordering).": "d",
        }
    q_cls_taskframing = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_TASKFRAMING", _options),
    )
    q_cls_taskframing
    return (q_cls_taskframing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — Exploring the Data
    """)
    return


@app.cell
def _(LABELS_CSV, pd):
    cls_raw_df = pd.read_csv(LABELS_CSV)
    cls_raw_df.columns = ['image_id', 'label']
    print('Total rows in CSV :', len(cls_raw_df))
    print('\nRaw label counts (1=neutrophil, 2=lymphocyte, 3=monocyte, 4=eosinophil, 5=basophil):')
    print(cls_raw_df['label'].value_counts().sort_index().to_string())

    cls_df = cls_raw_df[cls_raw_df['label'] != 5].copy()
    cls_df['label'] = cls_df['label'] - 1
    print(f'\nAfter dropping basophil: {len(cls_df)} images, {cls_df["label"].nunique()} classes')
    return (cls_df,)


@app.cell
def _(IMAGES_DIR, Image, WBC_CLASS_NAMES, cls_df, plt):
    n_classes = len(WBC_CLASS_NAMES)
    _fig, _axes = plt.subplots(n_classes, 4, figsize=(10, 3 * n_classes))
    for row, cls_name in enumerate(WBC_CLASS_NAMES):
        samples = cls_df[cls_df['label'] == row].head(4)
        for col, (_, r) in enumerate(samples.iterrows()):
            img = Image.open(IMAGES_DIR / f"{int(r['image_id']):03d}.bmp")
            _axes[row, col].imshow(img)
            _axes[row, col].axis('off')
            if col == 0:
                _axes[row, col].set_title(cls_name, fontsize=11, fontweight='bold')
    plt.suptitle('Sample images per class', fontsize=13)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2 — Slide-level metadata and data splitting

    The dataset contains single-cell images. Imagine we also had metadata indicating
    which original slide each cell came from. What is the best way to split the data for classification?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
        "Randomly split all single-cell images into train/val/test, ignoring slide ID.": "a",
        "Split at the slide level, so all cells from the same slide appear only in one split.": "b",
        "Stratified split keeping each slide's cells proportionally distributed across train/val/test.": "c",
        "Slide-level test, but random cell-level for train/val.": "d",
        }
    q2_1 = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_SPLIT", _options),
    )
    q2_1
    return (q2_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 — Building the Pipeline

    ### Exercise 3 — Implement `WBCDataset` and `build_dataloaders` in `classification.py`

    Open `classification.py` and complete the two TODOs:

    1. `WBCDataset.__getitem__` — load the image at the given row index, apply
       the transform, and return `(image_tensor, label_int)`.
    2. `build_dataloaders` — read the CSV, drop the basophil class, do a
       stratified train/val/test split, and wrap each split in a `DataLoader`.
       Augmentation (random flip, random rotation) is applied only to the
       training loader.

    The labels live in a CSV (no per-class subfolders), so `ImageFolder` does not work.
    Run the cell below to verify your implementation.
    """)
    return


@app.cell
def _(IMAGES_DIR, LABELS_CSV, build_dataloaders):
    cls_train_loader, cls_val_loader, cls_test_loader, cls_class_names = build_dataloaders(
        image_dir=IMAGES_DIR,
        labels_csv=LABELS_CSV,
        val_fraction=0.2,
        test_fraction=0.2,
        batch_size=32,
        seed=42,
    )

    print(f'Class names   : {cls_class_names}')
    print(f'Train batches : {len(cls_train_loader)}  (size {len(cls_train_loader.dataset)})')
    print(f'Val   batches : {len(cls_val_loader)}  (size {len(cls_val_loader.dataset)})')
    print(f'Test  batches : {len(cls_test_loader)}  (size {len(cls_test_loader.dataset)})')

    _images, _labels = next(iter(cls_train_loader))
    print(f'Batch shape   : {_images.shape}  labels shape: {_labels.shape}')
    return cls_class_names, cls_train_loader, cls_val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4 — Implement `SimpleCNN` in `classification.py`

    Build the minimal 2-block CNN described in the docstring.
    """)
    return


@app.cell
def _(SimpleCNN, device, torch):
    cnn = SimpleCNN(num_classes=4).to(device)
    _dummy = torch.zeros(2, 3, 224, 224).to(device)
    with torch.no_grad():
        _out = cnn(_dummy)
    print('Output shape:', _out.shape)  # expect torch.Size([2, 4])
    return (cnn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5 — Implement `run_epoch` in `classification.py`

    > **Note:** SimpleCNN is intentionally minimal.
    """)
    return


@app.cell
def _(cls_train_loader, cls_val_loader, cnn, device, nn, run_epoch, torch):
    cls_criterion = nn.CrossEntropyLoss()
    _opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
    cls_history_cnn = []
    for _epoch in range(30):
        _tr_loss, _tr_acc, _, _ = run_epoch(cnn, cls_train_loader, cls_criterion, _opt, device)
        _va_loss, _va_acc, _, _ = run_epoch(cnn, cls_val_loader,   cls_criterion, None, device)
        cls_history_cnn.append(dict(epoch=_epoch + 1,
                                    train_loss=_tr_loss, train_acc=_tr_acc,
                                    val_loss=_va_loss,   val_acc=_va_acc))
        print(f'Epoch {_epoch + 1:02d} | train {_tr_loss:.3f}/{_tr_acc:.3f} | val {_va_loss:.3f}/{_va_acc:.3f}')
    return cls_criterion, cls_history_cnn


@app.cell
def _(cls_history_cnn, plt):
    _epochs = [h['epoch'] for h in cls_history_cnn]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(_epochs, [h['train_loss'] for h in cls_history_cnn], label='train')
    _axes[0].plot(_epochs, [h['val_loss']   for h in cls_history_cnn], label='val')
    _axes[0].set(title='Loss — SimpleCNN', xlabel='Epoch', ylabel='Loss');     _axes[0].legend()
    _axes[1].plot(_epochs, [h['train_acc']  for h in cls_history_cnn], label='train')
    _axes[1].plot(_epochs, [h['val_acc']    for h in cls_history_cnn], label='val')
    _axes[1].set(title='Accuracy — SimpleCNN', xlabel='Epoch', ylabel='Acc'); _axes[1].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    classification_report,
    cls_class_names,
    cls_criterion,
    cls_val_loader,
    cnn,
    confusion_matrix,
    device,
    plt,
    run_epoch,
):
    _, _, _va_tgt, _va_pred = run_epoch(cnn, cls_val_loader, cls_criterion, None, device)
    print('SimpleCNN — Validation report')
    print(classification_report(_va_tgt, _va_pred, target_names=cls_class_names, zero_division=0))
    _cm = confusion_matrix(_va_tgt, _va_pred)
    _fig, _ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(_cm, display_labels=cls_class_names).plot(ax=_ax, xticks_rotation=45, colorbar=False)
    _ax.set_title('Validation Confusion Matrix — SimpleCNN (unweighted)')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4 — Evaluation Metrics

    ### Exercise 6 — Implement `compute_metrics` in `classification.py`

    For class **eosinophil** in a one-vs-rest setting, suppose
    $TP = 42,\ FP = 8,\ FN = 18$. Implement `compute_metrics(tp, fp, fn)` so it returns
    `(precision, recall, f1)` rounded to 4 decimals, then run the cell below.
    """)
    return


@app.cell
def _(compute_metrics):
    cls_precision, cls_recall, cls_f1 = compute_metrics(tp=42, fp=8, fn=18)
    print(f'Precision : {cls_precision:.4f}')
    print(f'Recall    : {cls_recall:.4f}')
    print(f'F1-score  : {cls_f1:.4f}')

    # Percentage of true eosinophil cases the model misses == 1 - recall.
    cls_missed_pct = (1 - cls_recall) * 100
    print(f'Missed eosinophil cases: {cls_missed_pct:.1f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 7 — Precision and recall formulas

    Which formula is correct?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Precision = TP / (TP + FN), Recall = TP / (TP + FP)": "a",
            "Precision = TP / (TP + FP), Recall = TP / (TP + FN)": "b",
            "Precision = TN / (TN + FP), Recall = TN / (TN + FN)": "c",
            "Precision = FP / (TP + FP), Recall = FN / (TP + FN)": "d",
        }
    q_cls_formulas = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_FORMULAS", _options),
    )
    q_cls_formulas
    return (q_cls_formulas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 8 — F1-score quick calculation

    For one class, suppose $TP = 10$, $FP = 5$, $FN = 5$. What is the F1-score closest to?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "0.25": "a",
            "0.50": "b",
            "0.67": "c",
            "1.00": "d",
        }
    q_cls_f1_calc = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_F1_CALC", _options),
    )
    q_cls_f1_calc
    return (q_cls_f1_calc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 9 — Precision vs recall in clinical use

    Consider two clinical settings:

    - **A.** A first-pass screening test where missing a positive is much worse than a
      false alarm (a follow-up test can rule out false alarms).
    - **B.** A confirmatory test that triggers an invasive biopsy: every positive
      prediction leads to surgery.

    Which pairing of priorities is correct?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "A → precision, B → recall": "a",
            "A → recall, B → precision": "b",
            "A → accuracy, B → recall": "c",
            "A → recall, B → accuracy": "d",
        }
    q_cls_pr_scenario = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_PR_SCENARIO", _options),
    )
    q_cls_pr_scenario
    return (q_cls_pr_scenario,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 10 — Always-majority baseline

    The confusion matrix above for SimpleCNN does not look promising. Imagine you have a model which always predicts **neutrophil** (the majority class).
    For the rare class **eosinophil**, what are the precision and recall of such a model?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Precision = 0.0 and recall = 0.0.": "a",
            "Precision = 1.0 and recall = 0.0.": "b",
            "Precision = 0.0 and recall = 1.0.": "c",
            "Precision = 1.0 and recall = 1.0.": "d",
        }
    q_majority = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_MAJORITY", _options),
    )
    q_majority
    return (q_majority,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 11 — Why accuracy can mislead

    Why can accuracy be misleading on an imbalanced multi-class dataset like this one?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Accuracy can be high even if the model ignores rare classes.": "a",
            "Accuracy cannot be computed for multi-class classification.": "b",
            "Accuracy is always lower than macro-F1.": "c",
            "Accuracy only measures false positives.": "d",
        }
    q_cls_acc_misleading = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_ACC_MISLEADING", _options),
    )
    q_cls_acc_misleading
    return (q_cls_acc_misleading,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Diagnosing the SimpleCNN baseline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""Two things stand out from the curve and the confusion matrix:

    1. Training accuracy plateaus very early — at almost exactly the proportion of neutrophils in the training set.
    2. The confusion matrix has **all predictions in a single column**: the model has learned the trivial *"always predict neutrophil"* strategy. Every minority class gets recall = 0.

    This is a textbook failure mode of training a low-capacity model on imbalanced data with
    *unweighted* cross-entropy. We now retrain **the same architecture** with one change only —
    the class weights from Exercise 12 are passed to `CrossEntropyLoss`.
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5 — Class Imbalance & Weighted Retrain

    ### Exercise 12 — Implement `compute_class_weights` in `classification.py`

    Implement inverse-frequency weights (mean count over count). Then run below.
    """)
    return


@app.cell
def _(WBC_CLASS_NAMES, cls_df, compute_class_weights):
    cls_actual_counts = {
        WBC_CLASS_NAMES[label]: int(count)
        for label, count in cls_df['label'].value_counts().sort_index().items()
    }
    print('Class counts:', cls_actual_counts)

    cls_weights = compute_class_weights(cls_actual_counts)
    print('\nClass weights:')
    for cls, w in cls_weights.items():
        print(f'  {cls:>12} : {w:.4f}')

    cls_rarest = max(cls_weights, key=cls_weights.get)
    print(f'\nHighest-weighted (rarest) class: {cls_rarest}')
    return (cls_weights,)


@app.cell
def _(
    SimpleCNN,
    WBC_CLASS_NAMES,
    cls_train_loader,
    cls_val_loader,
    cls_weights,
    device,
    nn,
    run_epoch,
    torch,
):
    cnn_w = SimpleCNN(num_classes=4).to(device)

    _wt = torch.tensor(
        [cls_weights[name] for name in WBC_CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )
    print('Weight tensor (label order 0..3):', _wt.tolist())

    cls_criterion_w = nn.CrossEntropyLoss(weight=_wt)
    _opt = torch.optim.Adam(cnn_w.parameters(), lr=0.001)
    cls_history_cnn_w = []
    for _epoch in range(30):
        _tr_loss, _tr_acc, _, _ = run_epoch(cnn_w, cls_train_loader, cls_criterion_w, _opt, device)
        _va_loss, _va_acc, _, _ = run_epoch(cnn_w, cls_val_loader,   cls_criterion_w, None, device)
        cls_history_cnn_w.append(dict(epoch=_epoch + 1,
                                      train_loss=_tr_loss, train_acc=_tr_acc,
                                      val_loss=_va_loss,   val_acc=_va_acc))
        print(f'Epoch {_epoch + 1:02d} | train {_tr_loss:.3f}/{_tr_acc:.3f} | val {_va_loss:.3f}/{_va_acc:.3f}')
    return cls_criterion_w, cls_history_cnn_w, cnn_w


@app.cell
def _(cls_history_cnn, cls_history_cnn_w, plt):
    _epochs = [h['epoch'] for h in cls_history_cnn]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(_epochs, [h['val_loss'] for h in cls_history_cnn],   label='unweighted')
    _axes[0].plot(_epochs, [h['val_loss'] for h in cls_history_cnn_w], label='weighted')
    _axes[0].set(title='Val loss — SimpleCNN', xlabel='Epoch', ylabel='Loss'); _axes[0].legend()
    _axes[1].plot(_epochs, [h['val_acc'] for h in cls_history_cnn],   label='unweighted')
    _axes[1].plot(_epochs, [h['val_acc'] for h in cls_history_cnn_w], label='weighted')
    _axes[1].set(title='Val accuracy — SimpleCNN', xlabel='Epoch', ylabel='Accuracy'); _axes[1].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    classification_report,
    cls_class_names,
    cls_criterion_w,
    cls_val_loader,
    cnn_w,
    confusion_matrix,
    device,
    plt,
    run_epoch,
):
    _, _, _va_tgt, _va_pred = run_epoch(cnn_w, cls_val_loader, cls_criterion_w, None, device)
    print('SimpleCNN (class-weighted) — Validation report')
    print(classification_report(_va_tgt, _va_pred, target_names=cls_class_names, zero_division=0))
    _cm = confusion_matrix(_va_tgt, _va_pred)
    _fig, _ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(_cm, display_labels=cls_class_names).plot(ax=_ax, xticks_rotation=45, colorbar=False)
    _ax.set_title('Validation Confusion Matrix — SimpleCNN (weighted)')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unweighted vs class-weighted SimpleCNN
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""Comparing the two SimpleCNN runs we see:

    - **Unweighted**: high accuracy on paper, but only because it always predicts the majority class. Macro-F1 is poor and rare-class recall is zero.
    - **Weighted**: accuracy may *drop*, but the model now actually attempts to predict every class. Macro-F1 (which averages across classes) goes up — even though the architecture is too small to do the job well.

        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 13 — Macro-F1 vs micro-F1

    Which statement about macro-F1 vs micro-F1 is correct?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Micro-F1 gives each class equal weight, while macro-F1 gives each example equal weight.": "a",
            "Macro-F1 averages F1 across classes equally, while micro-F1 aggregates decisions across all examples.": "b",
            "Macro-F1 is always the same as accuracy.": "c",
            "Micro-F1 ignores majority classes.": "d",
        }
    q_cls_macro_micro = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_MACRO_MICRO", _options),
    )
    q_cls_macro_micro
    return (q_cls_macro_micro,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 6 — Transfer Learning: ResNet18
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""**Why a pretrained backbone?** Medical-imaging datasets are almost always small. Datasets in the hundreds-to-low-thousands range are typical, and ours has only ~180 training images after the splits.

    Training a deep network from scratch on that little data is the failure mode you saw with SimpleCNN: the network has too many parameters and not enough examples to learn useful features, so it collapses onto the easiest signal available — usually the class prior.

     **Transfer learning** sidesteps this. The early layers of a network trained on a *large* upstream dataset (here, ImageNet's ~1.3M natural images) learn generic visual features — edges, textures, colour blobs, simple shape primitives — that turn out to be useful far beyond ImageNet's original task. We reuse those features and only retrain the final classification head (or fine-tune the whole network at a small learning rate) on our few hundred cell crops. ImageNet features are not perfect for stained microscopy — the domain gap is real — but they are still a far better starting point than random initialisation when training data is scarce.

        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `build_resnet18` is provided: it loads a ResNet18 with ImageNet-pretrained weights and swaps in a 4-class head for our task.
    """)
    return


@app.cell
def _(
    build_resnet18,
    cls_criterion,
    cls_train_loader,
    cls_val_loader,
    device,
    run_epoch,
    torch,
):
    cls_resnet = build_resnet18(num_classes=4).to(device)
    _opt = torch.optim.Adam(cls_resnet.parameters(), lr=1e-5)
    cls_history_rn = []
    for _epoch in range(30):
        _tr_loss, _tr_acc, _, _ = run_epoch(cls_resnet, cls_train_loader, cls_criterion, _opt, device)
        _va_loss, _va_acc, _, _ = run_epoch(cls_resnet, cls_val_loader,   cls_criterion, None, device)
        cls_history_rn.append(dict(epoch=_epoch + 1,
                                   train_loss=_tr_loss, train_acc=_tr_acc,
                                   val_loss=_va_loss,   val_acc=_va_acc))
        print(f'Epoch {_epoch + 1:02d} | train {_tr_loss:.3f}/{_tr_acc:.3f} | val {_va_loss:.3f}/{_va_acc:.3f}')
    return (cls_resnet,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    classification_report,
    cls_class_names,
    cls_criterion,
    cls_resnet,
    cls_val_loader,
    confusion_matrix,
    device,
    plt,
    run_epoch,
):
    _, _, _va_tgt, _va_pred = run_epoch(cls_resnet, cls_val_loader, cls_criterion, None, device)
    print('ResNet18 — Validation report')
    print(classification_report(_va_tgt, _va_pred, target_names=cls_class_names, zero_division=0))
    _cm = confusion_matrix(_va_tgt, _va_pred)
    _fig, _ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(_cm, display_labels=cls_class_names).plot(ax=_ax, xticks_rotation=45, colorbar=False)
    _ax.set_title('Validation Confusion Matrix — ResNet18')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 7 — Model Comparison

    ### Exercise 14 — Model Comparison

    Below is the canonical table of validation metrics produced by training the
    three models in this section with `seed=42` for 30 epochs each. *Use this
    table to answer the three short questions that follow.* Your own training
    run will produce numbers that are very close to these — small differences
    are expected (CUDA non-determinism, augmentation order) and don't change
    the qualitative picture. The questions below test how you *interpret* the
    table, not whether your run reproduces it bit-for-bit.

    | Model                          | Val Accuracy | Val Macro-F1 | Worst-class Recall |
    |--------------------------------|:------------:|:------------:|:------------------:|
    | SimpleCNN — unweighted         | 0.58         | 0.18         | 0.00               |
    | SimpleCNN — class-weighted     | 0.40         | 0.29         | 0.00               |
    | ResNet18 (full fine-tune)      | 0.95         | 0.88         | 0.75               |

    **Quick read.** The unweighted SimpleCNN looks accurate but its macro-F1
    is poor and at least one class has zero recall — classic majority-class
    collapse. Adding class weights drops the headline accuracy *and* leaves at
    least one class still unrecognised — weights cured the collapse but didn't
    add capacity. ResNet18 with ImageNet pretraining is the only model that
    actually distinguishes all four cell types.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(a)** If missing a rare WBC type carries the highest clinical cost, which validation metric should you optimize when choosing between models?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Validation accuracy.": "a",
            "Validation macro-F1.": "b",
            "Worst-class recall (the recall on the rarest / most-missed class).": "c",
            "Training loss.": "d",
        }
    q6a = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_E14A", _options),
    )
    q6a
    return (q6a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(b)** What does the gap between the two SimpleCNN rows (unweighted vs class-weighted) tell you about the role of class weights vs the role of model capacity?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Class weights and model capacity are interchangeable — either alone is enough.": "a",
            "Class weights stop the model from collapsing onto the majority class without changing capacity; further absolute gains require more capacity (e.g. a pretrained backbone).": "b",
            "Class weights hurt rare-class recall and should be avoided.": "c",
            "Once class weights are applied, model capacity becomes irrelevant.": "d",
        }
    q6b = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_E14B", _options),
    )
    q6b
    return (q6b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(c)** What additional check would you run before finalising the choice between these three models?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Train each model for more epochs and pick the one with the lowest training loss.": "a",
            "Try several optimizers and pick whichever wins on the validation set.": "b",
            "Evaluate the chosen model on the held-out test set, compute bootstrap confidence intervals, and inspect the per-class confusion matrix.": "c",
            "Skip further checks — the validation numbers are enough.": "d",
        }
    q6c = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_E14C", _options),
    )
    q6c
    return (q6c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 15 — Patient-level CV: direction of bias

    Suppose this dataset had patient IDs and each patient contributed multiple
    cell-crop images. You evaluate with k-fold cross-validation.

    **(a)** Random k-fold CV vs patient-level k-fold CV — what is the direction
    of bias on test AUROC?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Random CV gives an upward-biased estimate (correlated train/test sibling images).": "a",
            "Random CV gives a downward-biased estimate.": "b",
            "No bias — both are equivalent.": "c",
            "The sign depends on the random seed.": "d",
        }
    q_cls_cvbias_dir = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_CVBIAS_DIR", _options),
    )
    q_cls_cvbias_dir
    return (q_cls_cvbias_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        The WBC Dataset 1 used here ships **without patient IDs**, so the splits in `build_dataloaders` are random across
        cells. Exercise 15 is therefore conceptual — reason about the bias as if patient IDs were available.
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(b)** Which assumption do you need for the bias direction in (a) to hold?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Cells from the same patient are positively correlated through latent patient factors (e.g. staining batch, scanner, anatomy).": "a",
            "Labels are uniformly distributed.": "b",
            "All patients contribute the same number of cells.": "c",
            "The model is convex.": "d",
        }
    q_cls_cvbias_assum = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_CVBIAS_ASSUM", _options),
    )
    q_cls_cvbias_assum
    return (q_cls_cvbias_assum,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 16 — Marketing claim on a rare-cell screen

    The basophil class was dropped at the start of this assignment because
    Dataset 1 contains only one basophil image. Clinically, basophils are the
    rarest white blood cell — about **0.5%** of WBCs in a healthy differential —
    yet basophilia is a meaningful diagnostic signal (e.g. it is part of the
    WHO criteria for chronic myeloid leukaemia accelerated phase).

    A startup markets an AI **pre-screen** for basophils with **95% sensitivity**
    and **95% specificity**. They claim a lab tech only needs to manually verify
    the cells flagged as positive — saving a lot of work.

    Take prevalence = 0.5% (5 basophils per 1000 cells).

    **(a)** Out of 1000 cells, approximately how many will the technician need to
    manually verify (= the number of cells flagged positive by the AI)?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_cls_baso_count = mo.ui.number(
        start=0, stop=1000, step=1,
        label="Cells flagged for manual verification:",
        value=submission_default("Q_CLS_BASO_COUNT"),
    )
    q_cls_baso_count
    return (q_cls_baso_count,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(b)** Of those flagged positives, what fraction are *true* basophils
    (the positive predictive value, PPV)? Round to two decimals.
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_cls_baso_ppv = mo.ui.number(
        start=0.0, stop=1.0, step=0.01,
        label="PPV:",
        value=submission_default("Q_CLS_BASO_PPV"),
    )
    q_cls_baso_ppv
    return (q_cls_baso_ppv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(c)** Should the lab adopt this pre-screen?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Yes — 95% sensitivity is excellent and saves about 94% of manual lab work.": "a",
            "No — at this prevalence ~91% of \"positives\" are false alarms (PPV ≈ 9%) and missing 5% of true basophils may be clinically unacceptable when basophil counts are themselves a diagnostic signal.": "b",
            "Yes, but only if the lab also uses a separate model for monocytes.": "c",
            "No, because 95% sensitivity is too low to be useful.": "d",
        }
    q_cls_baso_adopt = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_BASO_ADOPT", _options),
    )
    q_cls_baso_adopt
    return (q_cls_baso_adopt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 8 — Matthews Correlation Coefficient

    F1 summarises performance using only TP, FP, and FN — the True Negative cell
    of the confusion matrix never enters the formula. **MCC** uses all four
    cells and so distinguishes models that F1 cannot.

    $$\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

    Range $[-1, +1]$: $+1$ perfect, $0$ random, $-1$ inverted.

    ### Exercise 17 — Implement `compute_mcc` in `classification.py`

    Open `classification.py` and complete `compute_mcc(tp, tn, fp, fn)`. Use the
    same zero-denominator convention as `compute_metrics` (return `0.0`).
    """)
    return


@app.cell
def _(compute_mcc):
    # Sanity check: extend the Exercise 6 numbers (TP=42, FP=8, FN=18) with
    # TN=132 -- the True Negatives that F1 ignored.
    cls_mcc = compute_mcc(tp=42, tn=132, fp=8, fn=18)
    print(f"MCC for (TP=42, TN=132, FP=8, FN=18) : {cls_mcc:.4f}")
    print("(expect ~0.6803 -- decisively non-random, but visibly below 1.0)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 18 — Why also report MCC alongside F1?

    On a small clinical test set you find two classifiers with **identical macro-F1**
    but visibly different confusion matrices (one is much better at clearing
    healthy patients, the other much better at flagging sick ones). Why does
    the lecture recommend reporting MCC in addition to F1?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "MCC is mathematically guaranteed to be larger than F1, so it is the more optimistic single-number summary.": "a",
            "F1 ignores the True Negative cell of the confusion matrix; MCC uses all four cells, so two models with the same F1 can still be distinguished by their MCC.": "b",
            "MCC works only on balanced datasets, so reporting it alongside F1 confirms that the test set is balanced.": "c",
            "MCC is computed without the sigmoid or softmax, so it bypasses calibration issues that affect F1.": "d",
        }
    q_cls_mcc_vs_f1 = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_CLS_MCC_VS_F1", _options),
    )
    q_cls_mcc_vs_f1
    return (q_cls_mcc_vs_f1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 19 — Make F1 lie on a referred cytology queue

    A senior haematologist's review queue only receives WBC slides already
    flagged as suspicious by junior technicians. In this filtered cohort,
    **90 %** of slides contain at least one abnormal cell — *not* because the
    disease is that prevalent in the population, but because the cohort itself
    is biased by the referral pipeline.

    You evaluate a slide-level "abnormal vs. normal" classifier on
    **N = 1000** queued slides (so 900 are truly abnormal, 100 are truly
    normal). Construct a confusion matrix `(TP, FN, FP, TN)` such that *all*
    of the following hold simultaneously:

    1. **F1 ≥ 0.90** (the dashboard shows a triumphant green metric)
    2. **MCC ≤ 0.30** (but the model is essentially guessing)
    3. All four cells are non-negative integers
    4. `TP + FN + FP + TN = 1000`
    5. `TP + FN = 900` (true abnormals are fixed by the cohort)
    6. `FP + TN = 100`  (true normals are fixed)

    Many tuples are valid; the autograder verifies the inequalities, not your
    exact numbers. The lesson is constructive: **prevalence is not a property
    of the disease, it depends on the workflow step you measure at**, and F1
    silently inflates whenever the positive class dominates the cohort.
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q19_tp = mo.ui.number(start=0, stop=1000, step=1, label="TP (abnormal predicted abnormal)", value=submission_default("Q19_TP"))
    q19_fn = mo.ui.number(start=0, stop=1000, step=1, label="FN (abnormal predicted normal)", value=submission_default("Q19_FN"))
    q19_fp = mo.ui.number(start=0, stop=1000, step=1, label="FP (normal predicted abnormal)", value=submission_default("Q19_FP"))
    q19_tn = mo.ui.number(start=0, stop=1000, step=1, label="TN (normal predicted normal)", value=submission_default("Q19_TN"))
    mo.vstack([q19_tp, q19_fn, q19_fp, q19_tn])
    return q19_fn, q19_fp, q19_tn, q19_tp


@app.cell
def _(compute_mcc, compute_metrics, q19_fn, q19_fp, q19_tn, q19_tp):
    # Live feedback so students can iterate.
    _tp = int(q19_tp.value or 0)
    _fn = int(q19_fn.value or 0)
    _fp = int(q19_fp.value or 0)
    _tn = int(q19_tn.value or 0)
    _N = _tp + _fn + _fp + _tn

    if _N == 0:
        print("Fill in TP, FN, FP, TN above to see the metrics.")
    else:
        _, _, _f1 = compute_metrics(_tp, _fp, _fn)
        _mcc      = compute_mcc(_tp, _tn, _fp, _fn)
        _ok = lambda b: "OK " if b else "X  "
        print(f"N = {_N}     TP+FN = {_tp + _fn}     FP+TN = {_fp + _tn}")
        print(f"F1  = {_f1:.4f}   {_ok(_f1 >= 0.90)} (target  >= 0.90)")
        print(f"MCC = {_mcc:.4f}   {_ok(_mcc <= 0.30)} (target  <= 0.30)")
        print(f"totals: {_ok(_N == 1000)} N = 1000   "
              f"{_ok(_tp + _fn == 900)} TP+FN = 900   "
              f"{_ok(_fp + _tn == 100)} FP+TN = 100")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section B — Segmentation

    Per-pixel classification with three classes:
    `0 = background (incl. RBCs)`, `1 = cytoplasm`, `2 = nucleus`.

    Mask format: 8-bit grayscale PNG with values `{0, 128, 255}`.

    All functions and classes you need to implement live in **`segmentation.py`**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — Understanding the Problem

    ### Exercise 1 — Task framing

    In Section B we move from one label per image (cell type) to per-pixel labels.
    The mask uses three classes: `0 = background (incl. RBCs)`, `1 = cytoplasm`,
    `2 = nucleus`.

    **Which formulation correctly describes the supervised learning problem in Section B?**
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Input x = an RGB cell-crop image; output y = a per-pixel class label map with values in {0, 1, 2}; task = semantic segmentation.": "a",
            "Input x = an RGB cell-crop image; output y = a single class label per image; task = whole-image multi-class classification.": "b",
            "Input x = an RGB cell-crop image; output y = a real-valued mask with continuous values in [0, 1]; task = pixel-wise regression.": "c",
            "Input x = an RGB cell-crop image; output y = bounding-box coordinates around each cell; task = object detection.": "d",
        }
    q_seg_taskframing = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_TASKFRAMING", _options),
    )
    q_seg_taskframing
    return (q_seg_taskframing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — Exploring the Data
    """)
    return


@app.cell
def _(IMAGES_DIR, Image, SEG_CLASS_NAMES, np):
    seg_bmp_paths = sorted(IMAGES_DIR.glob('*.bmp'))
    print(f'Found {len(seg_bmp_paths)} image/mask pairs.')

    seg_pixel_counts = {name: 0 for name in SEG_CLASS_NAMES}
    _raw_to_class = {0: 'background', 128: 'cytoplasm', 255: 'nucleus'}
    for _bmp in seg_bmp_paths:
        _m = np.array(Image.open(_bmp.with_suffix('.png')))
        _vals, _ct = np.unique(_m, return_counts=True)
        for _v, _c in zip(_vals.tolist(), _ct.tolist()):
            seg_pixel_counts[_raw_to_class[_v]] += _c

    _total = sum(seg_pixel_counts.values())
    print('\nPixel-level class distribution (whole dataset):')
    for _name, _c in seg_pixel_counts.items():
        print(f'  {_name:>11} : {_c:>10}  ({100 * _c / _total:5.2f}%)')
    return seg_bmp_paths, seg_pixel_counts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2 — Why pixel accuracy can mislead

    Given the pixel-level class distribution above, why can **pixel accuracy** be a misleading
    metric on this dataset?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Most pixels belong to the background class, so a model can score highly while failing on important structures (cytoplasm and nucleus).": "a",
            "Pixel accuracy cannot be computed for segmentation tasks.": "b",
            "Pixel accuracy is always lower than Dice.": "c",
            "Pixel accuracy ignores the background class.": "d",
        }
    q_seg_pixacc = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_PIXACC", _options),
    )
    q_seg_pixacc
    return (q_seg_pixacc,)


@app.cell
def _(IMAGES_DIR, Image, SEG_CMAP, np, plt):
    _sample_ids = [1, 50, 150, 250]
    _fig, _axes = plt.subplots(len(_sample_ids), 3, figsize=(9, 3 * len(_sample_ids)))
    _raw_to_class = {0: 0, 128: 1, 255: 2}
    for _row, _sid in enumerate(_sample_ids):
        _img = np.array(Image.open(IMAGES_DIR / f'{_sid:03d}.bmp'))
        _mask_raw = np.array(Image.open(IMAGES_DIR / f'{_sid:03d}.png'))
        _mask = np.vectorize(_raw_to_class.get)(_mask_raw)

        _axes[_row, 0].imshow(_img);                                  _axes[_row, 0].set_title(f'{_sid:03d}.bmp');     _axes[_row, 0].axis('off')
        _axes[_row, 1].imshow(_mask, cmap=SEG_CMAP, vmin=0, vmax=2);  _axes[_row, 1].set_title('mask (class ids)');    _axes[_row, 1].axis('off')
        _axes[_row, 2].imshow(_img)
        _axes[_row, 2].imshow(_mask, cmap=SEG_CMAP, vmin=0, vmax=2, alpha=0.5)
        _axes[_row, 2].set_title('overlay');  _axes[_row, 2].axis('off')
    plt.suptitle('Image | Mask | Overlay   (black = bg, gray = cytoplasm, red = nucleus)', fontsize=11)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3 — Augmentation, evaluation, and loss choices

    Three short MCs probing pitfalls that are specific to segmentation.

    **(a)** You add four augmentations to your training pipeline: random horizontal
    flip, random 15° rotation, brightness jitter, and Gaussian noise. Which of these
    must be applied to **both** the image and its mask?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "All four, applied identically.": "a",
            "Only flip and rotation; brightness and noise apply to the image only.": "b",
            "Only brightness and noise; geometric augmentations break segmentation alignment.": "c",
            "None — augmentation isn't safe in segmentation.": "d",
        }
    q_seg_aug_asym = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_AUG_ASYM", _options),
    )
    q_seg_aug_asym
    return (q_seg_aug_asym,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(b)** Two common ways to compute *mean Dice* over the validation set are:

    - **(i)** Compute Dice on each image, then average across images.
    - **(ii)** Sum TP, FP, FN across all images, then compute a single Dice on the totals.

    Suppose part of your validation set contains images with no foreground at all
    (pure background). Which method is more **sensitive** to those edge cases?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Method (i) — a 'no-foreground' image gives Dice = 1 by convention, inflating the per-image average.": "a",
            "Method (ii) — summing TP/FP/FN across images amplifies the effect of empty images.": "b",
            "Both methods are equivalent.": "c",
            "Neither — Dice cannot be computed at all on no-foreground images.": "d",
        }
    q_seg_dice_agg = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_DICE_AGG", _options),
    )
    q_seg_dice_agg
    return (q_seg_dice_agg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(c)** Cross-entropy loss converges quickly but the nucleus Dice plateaus low.
    A colleague suggests training directly with **Dice loss** (using Dice as the loss
    function, not just as a metric). What is the main reason this can help?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Dice loss is faster to compute than cross-entropy.": "a",
            "Dice loss directly optimizes the metric you care about, sidestepping the per-pixel imbalance that cross-entropy struggles with on rare classes.": "b",
            "Dice loss requires fewer epochs to train.": "c",
            "Dice loss is the only loss compatible with multi-class segmentation.": "d",
        }
    q_seg_diceloss = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_DICELOSS", _options),
    )
    q_seg_diceloss
    return (q_seg_diceloss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 — Multi-class Dice

    ### Exercise 4 — Implement `dice_score` in `segmentation.py`

    For a single class $c$:

    $$\text{Dice}_c = \frac{2 \cdot TP_c}{2 \cdot TP_c + FP_c + FN_c}$$

    Return `(mean_dice, [dice_class_0, dice_class_1, ...])` rounded to 4 decimals.
    """)
    return


@app.cell
def _(dice_score, torch):
    _target_a = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
    print('Perfect prediction        ->', dice_score(_target_a.clone(), _target_a, num_classes=3))
    print('Always predict background ->', dice_score(torch.zeros_like(_target_a), _target_a, num_classes=3))

    _target_b = torch.tensor([[[0, 0, 1], [1, 1, 2], [2, 2, 2]]])
    _pred_b   = torch.tensor([[[0, 0, 1], [1, 1, 1], [2, 2, 2]]])
    print('Single-pixel error        ->', dice_score(_pred_b, _target_b, num_classes=3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5 — Why per-class Dice?

    Why is per-class Dice especially informative on this dataset?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Each class gets its own score, so poor performance on small but important structures (e.g. nuclei) is not hidden by good background performance.": "a",
            "It is always identical to pixel accuracy.": "b",
            "It removes the need for visual inspection.": "c",
            "It only evaluates minority classes.": "d",
        }
    q_seg_perclass = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_PERCLASS", _options),
    )
    q_seg_perclass
    return (q_seg_perclass,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4 — Pixel-level Class Imbalance

    ### Exercise 6 — Implement `compute_pixel_class_weights` in `segmentation.py`

    Same inverse-frequency formula as Section A, just applied to *pixel* counts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        **Background — U-Net loss simplification (no exercise).**
        Ronneberger 2015 also adds a *boundary distance* term,
        $w_0\,\exp(-(d_1{+}d_2)^2/2\sigma^2)$, that spikes between touching cells and forces
        the network to predict background there. We use only the inverse-frequency class
        weight $w_c(x)$ — enough to fix background dominance, but it does not include the
        boundary trick that lets a binary mask separate adjacent cells. Adding it is a
        natural extension if you want to experiment.
        """),
        kind="info",
    )
    return


@app.cell
def _(compute_pixel_class_weights, seg_pixel_counts):
    seg_weights = compute_pixel_class_weights(seg_pixel_counts)
    print('Pixel-level class weights:')
    for _cls, _w in seg_weights.items():
        print(f'  {_cls:>11} : {_w:.4f}')
    return (seg_weights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 7 — Pixel-level class imbalance

    Why can pixel-level class imbalance be a problem in semantic segmentation?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Background pixels often dominate, so the loss may under-emphasize smaller classes such as cytoplasm or nucleus.": "a",
            "It prevents the use of convolutional networks.": "b",
            "It makes Dice impossible to compute.": "c",
            "It means segmentation should be treated as classification instead.": "d",
        }
    q_seg_imbalance = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_IMBALANCE", _options),
    )
    q_seg_imbalance
    return (q_seg_imbalance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5 — Classical Baseline: Otsu's Method

    Before training a UNet, ask: **how well does a one-page unsupervised algorithm do
    on this segmentation task?**

    *Otsu's method* (Otsu 1979) finds the grayscale threshold $t$ that maximises the
    inter-class variance

    $$\sigma_b^2(t) \;=\; w_0(t)\, w_1(t)\, (\mu_0(t) - \mu_1(t))^2$$

    where $w_0,w_1$ are the pixel fractions below/above $t$ and $\mu_0,\mu_1$ are the
    corresponding mean intensities. Output: a single threshold, then a **binary** mask.

    The catch: Otsu cannot tell *nucleus* from *cytoplasm* — it only separates two
    intensity populations. The fairest comparison is therefore against a **binarised**
    version of the ground truth (`cytoplasm ∪ nucleus = 1`, `bg = 0`).

    ### Exercise 8 — Implement `otsu_threshold` in `segmentation.py`

    Sweep all 256 candidate thresholds and return the integer $t \in [0, 255]$ that
    maximises the inter-class variance.
    """)
    return


@app.cell
def _(IMAGES_DIR, Image, np, otsu_threshold, plt):
    # Sanity check: pick one image, plot histogram with Otsu threshold + the binary masks.
    _sid       = 50
    _img_gray  = np.array(Image.open(IMAGES_DIR / f'{_sid:03d}.bmp').convert('L'))
    _mask_raw  = np.array(Image.open(IMAGES_DIR / f'{_sid:03d}.png'))
    _t         = otsu_threshold(_img_gray)
    _otsu_mask = (_img_gray < _t).astype(np.int64)   # cell tissue is darker than the yellow background
    _gt_binary = (_mask_raw > 0).astype(np.int64)    # cyto + nucleus = 1, bg = 0
    print(f'Otsu threshold for image {_sid:03d}: t = {_t}')

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.6))
    _axes[0].hist(_img_gray.ravel(), bins=64, range=(0, 256), color='steelblue')
    _axes[0].axvline(_t, color='crimson', linestyle='--', label=f't = {_t}')
    _axes[0].set(title='Grayscale histogram', xlabel='intensity'); _axes[0].legend()
    _axes[1].imshow(_img_gray, cmap='gray');                  _axes[1].axis('off'); _axes[1].set_title('Image (gray)')
    _axes[2].imshow(_otsu_mask, cmap='gray');                 _axes[2].axis('off'); _axes[2].set_title('Otsu mask (cell = 1)')
    _axes[3].imshow(_gt_binary, cmap='gray');                 _axes[3].axis('off'); _axes[3].set_title('GT binary (cyto + nuc = 1)')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(Image, dice_score, np, otsu_threshold, seg_bmp_paths, torch):
    # Apply Otsu to every image, compute dataset-level binary Dice
    # (concatenated over all pixels, same convention as run_epoch_seg uses for UNet).
    _all_pred = []
    _all_gt   = []
    for _bmp in seg_bmp_paths:
        _gray = np.array(Image.open(_bmp).convert('L'))
        _raw  = np.array(Image.open(_bmp.with_suffix('.png')))
        _t    = otsu_threshold(_gray)
        _all_pred.append((_gray < _t).astype(np.int64).ravel())
        _all_gt.append((_raw > 0).astype(np.int64).ravel())

    otsu_pred_all = torch.from_numpy(np.concatenate(_all_pred))
    otsu_gt_all   = torch.from_numpy(np.concatenate(_all_gt))
    otsu_mean, otsu_per = dice_score(otsu_pred_all, otsu_gt_all, num_classes=2)
    print(f'Otsu (whole dataset, binary cell vs. background)')
    print(f'  mean Dice : {otsu_mean:.4f}')
    print(f'  background: {otsu_per[0]:.4f}')
    print(f'  cell      : {otsu_per[1]:.4f}')
    return (otsu_mean,)


@app.cell
def _(IMAGES_DIR, Image, np, otsu_threshold, plt):
    # Qualitative comparison on a few images.
    _sample_ids = [1, 50, 150, 250]
    _fig, _axes = plt.subplots(len(_sample_ids), 4, figsize=(12, 3 * len(_sample_ids)))
    for _row, _sid in enumerate(_sample_ids):
        _img      = np.array(Image.open(IMAGES_DIR / f'{_sid:03d}.bmp'))
        _gray     = np.array(Image.open(IMAGES_DIR / f'{_sid:03d}.bmp').convert('L'))
        _mask_raw = np.array(Image.open(IMAGES_DIR / f'{_sid:03d}.png'))
        _t        = otsu_threshold(_gray)
        _otsu     = (_gray < _t).astype(np.int64)
        _gt_bin   = (_mask_raw > 0).astype(np.int64)

        _axes[_row, 0].imshow(_img);                 _axes[_row, 0].axis('off'); _axes[_row, 0].set_title(f'{_sid:03d}.bmp')
        _axes[_row, 1].imshow(_gray, cmap='gray');   _axes[_row, 1].axis('off'); _axes[_row, 1].set_title(f'gray (t = {_t})')
        _axes[_row, 2].imshow(_otsu, cmap='gray');   _axes[_row, 2].axis('off'); _axes[_row, 2].set_title('Otsu mask')
        _axes[_row, 3].imshow(_gt_bin, cmap='gray'); _axes[_row, 3].axis('off'); _axes[_row, 3].set_title('GT binary')
    plt.suptitle('Otsu predictions vs binarised ground truth', fontsize=12)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Take-away.** Otsu reaches a respectable cell-vs-background Dice with **zero
    parameters and no training**. But by construction it produces only a binary mask
    — it cannot disambiguate nucleus from cytoplasm. The next sections train a UNet
    that solves the harder *per-class* problem; in Exercise 15 you will compare Otsu's
    binary Dice to UNet's binary Dice (UNet's `cytoplasm ∪ nucleus` collapsed to one
    class) to isolate the supervised gain over a classical baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 6 — Building the Pipeline

    ### Exercise 9 — Implement `WBCSegDataset` and `build_seg_dataloaders` in `segmentation.py`

    Two important details:
    1. **Bilinear** for images, **nearest** for masks (otherwise non-{0,128,255} values appear).
    2. **Paired** augmentation: an image flip must come with a mask flip.
    """)
    return


@app.cell
def _(IMAGES_DIR, build_seg_dataloaders):
    seg_train_loader, seg_val_loader, seg_test_loader, seg_class_names = build_seg_dataloaders(
        image_dir=IMAGES_DIR,
        val_fraction=0.2,
        test_fraction=0.2,
        image_size=128,
        batch_size=8,
        seed=42,
    )

    print(f'Class names   : {seg_class_names}')
    print(f'Train batches : {len(seg_train_loader)}  (size {len(seg_train_loader.dataset)})')
    print(f'Val   batches : {len(seg_val_loader)}  (size {len(seg_val_loader.dataset)})')
    print(f'Test  batches : {len(seg_test_loader)}  (size {len(seg_test_loader.dataset)})')

    _images, _masks = next(iter(seg_train_loader))
    print(f'Image batch shape : {_images.shape}  range=({_images.min():.3f}, {_images.max():.3f})')
    print(f'Mask  batch shape : {_masks.shape}   unique values={_masks.unique().tolist()}')
    return seg_class_names, seg_train_loader, seg_val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 10 — Mask corruption after resizing

    A segmentation mask originally contains only values {0, 128, 255}. After resizing,
    it contains values such as {0, 17, 64, 128, 201, 255}. What most likely happened?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "The mask was normalized correctly.": "a",
            "Bilinear interpolation was applied to the mask, creating invalid intermediate values.": "b",
            "The model discovered new valid segmentation classes.": "c",
            "The RGB image was accidentally converted to grayscale.": "d",
        }
    q_seg_resize = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_RESIZE", _options),
    )
    q_seg_resize
    return (q_seg_resize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 11 — Paired augmentation

    Why must geometric augmentations (flips, rotations) be applied jointly to the image and its mask?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "The image can be flipped, but the mask should stay unchanged.": "a",
            "The mask can be flipped, but the image should stay unchanged.": "b",
            "The image and mask must be transformed together so corresponding pixels remain aligned.": "c",
            "Paired augmentation is only important in image classification.": "d",
        }
    q_seg_paired = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_PAIRED", _options),
    )
    q_seg_paired
    return (q_seg_paired,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 12 — Implement `SmallUNet` in `segmentation.py`
    """)
    return


@app.cell
def _(SmallUNet, device, torch):
    seg_unet = SmallUNet(num_classes=3).to(device)
    print(f'SmallUNet : {sum(p.numel() for p in seg_unet.parameters()):,} parameters')

    _dummy = torch.zeros(2, 3, 128, 128, device=device)
    with torch.no_grad():
        _out = seg_unet(_dummy)
    print('Output shape:', _out.shape)  # expect torch.Size([2, 3, 128, 128])
    return (seg_unet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 13 — Implement `run_epoch_seg` in `segmentation.py`

    Targets are `[B, H, W]` long-tensors and accuracy is *pixel* accuracy.
    Return flattened `all_targets` and `all_preds` so they can feed `dice_score`.
    """)
    return


@app.cell
def _(
    SEG_CLASS_NAMES,
    device,
    dice_score,
    nn,
    run_epoch_seg,
    seg_train_loader,
    seg_unet,
    seg_val_loader,
    seg_weights,
    torch,
):
    _wt = torch.tensor(
        [seg_weights[c] for c in SEG_CLASS_NAMES], dtype=torch.float32, device=device,
    )
    print('Loss weight tensor:', _wt.tolist())

    seg_criterion = nn.CrossEntropyLoss(weight=_wt)
    _opt = torch.optim.Adam(seg_unet.parameters(), lr=1e-3)

    seg_history_unet = []
    for _epoch in range(30):
        _tr_loss, _tr_acc, _, _ = run_epoch_seg(seg_unet, seg_train_loader, seg_criterion, _opt, device)
        _va_loss, _va_acc, _va_t, _va_p = run_epoch_seg(seg_unet, seg_val_loader, seg_criterion, None, device)
        _, _va_per = dice_score(_va_p, _va_t, num_classes=3)
        _va_mean = sum(_va_per) / 3
        seg_history_unet.append(dict(
            epoch=_epoch + 1, train_loss=_tr_loss, train_acc=_tr_acc,
            val_loss=_va_loss, val_acc=_va_acc, val_dice=_va_mean, val_dice_per=_va_per,
        ))
        print(f'ep{_epoch + 1:02d} | tr {_tr_loss:.3f}/{_tr_acc:.3f} '
              f'| val {_va_loss:.3f}/{_va_acc:.3f} mean-Dice {_va_mean:.3f} per {_va_per}')
    return seg_criterion, seg_history_unet


@app.cell
def _(plt, seg_history_unet):
    _epochs = [h['epoch'] for h in seg_history_unet]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(_epochs, [h['train_loss'] for h in seg_history_unet], label='train')
    _axes[0].plot(_epochs, [h['val_loss']   for h in seg_history_unet], label='val')
    _axes[0].set(title='Loss — SmallUNet', xlabel='Epoch', ylabel='Loss'); _axes[0].legend()
    _axes[1].plot(_epochs, [h['val_dice']   for h in seg_history_unet], label='mean')
    for _ci, _name in enumerate(['background', 'cytoplasm', 'nucleus']):
        _axes[1].plot(_epochs, [h['val_dice_per'][_ci] for h in seg_history_unet], label=_name)
    _axes[1].set(title='Val Dice — SmallUNet', xlabel='Epoch', ylabel='Dice'); _axes[1].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEG_CMAP,
    device,
    dice_score,
    plt,
    run_epoch_seg,
    seg_class_names,
    seg_criterion,
    seg_unet,
    seg_val_loader,
    torch,
):
    _, _va_acc, _va_t, _va_p = run_epoch_seg(seg_unet, seg_val_loader, seg_criterion, None, device)
    _mean_d, seg_per_d_scratch = dice_score(_va_p, _va_t, num_classes=3)
    print(f'SmallUNet  pixel acc {_va_acc:.3f}  mean Dice {_mean_d:.3f}')
    for _c, _name in enumerate(seg_class_names):
        print(f'  {_name:>11} Dice : {seg_per_d_scratch[_c]:.3f}')

    _images, _masks = next(iter(seg_val_loader))
    seg_unet.eval()
    with torch.no_grad():
        _preds = seg_unet(_images.to(device)).argmax(dim=1).cpu()
    _mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    _std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    _n = min(4, _images.shape[0])
    _fig, _axes = plt.subplots(_n, 3, figsize=(9, 3 * _n))
    for _i in range(_n):
        _disp = (_images[_i] * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()
        _axes[_i, 0].imshow(_disp);                                          _axes[_i, 0].axis('off'); _axes[_i, 0].set_title('image')
        _axes[_i, 1].imshow(_masks[_i], cmap=SEG_CMAP, vmin=0, vmax=2);      _axes[_i, 1].axis('off'); _axes[_i, 1].set_title('ground truth')
        _axes[_i, 2].imshow(_preds[_i], cmap=SEG_CMAP, vmin=0, vmax=2);      _axes[_i, 2].axis('off'); _axes[_i, 2].set_title('SmallUNet pred')
    plt.suptitle('SmallUNet predictions on validation set', fontsize=11)
    plt.tight_layout()
    plt.show()
    return (seg_per_d_scratch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 7 — Pretrained-encoder UNet

    `build_pretrained_unet()` returns a UNet whose ResNet18 encoder is pretrained on ImageNet
    (via `segmentation_models_pytorch`). The decoder is randomly initialised. Compare per-class
    Dice and qualitative predictions.
    """)
    return


@app.cell
def _(
    SEG_CLASS_NAMES,
    build_pretrained_unet,
    device,
    dice_score,
    nn,
    run_epoch_seg,
    seg_train_loader,
    seg_val_loader,
    seg_weights,
    torch,
):
    seg_pre_unet = build_pretrained_unet(num_classes=3).to(device)
    print(f'Pretrained UNet : {sum(p.numel() for p in seg_pre_unet.parameters()):,} parameters')

    _wt = torch.tensor([seg_weights[c] for c in SEG_CLASS_NAMES], dtype=torch.float32, device=device)
    seg_criterion_pre = nn.CrossEntropyLoss(weight=_wt)
    _opt = torch.optim.Adam(seg_pre_unet.parameters(), lr=1e-4)

    seg_history_pre = []
    for _epoch in range(30):
        _tr_loss, _tr_acc, _, _ = run_epoch_seg(seg_pre_unet, seg_train_loader, seg_criterion_pre, _opt, device)
        _va_loss, _va_acc, _va_t, _va_p = run_epoch_seg(seg_pre_unet, seg_val_loader, seg_criterion_pre, None, device)
        _, _va_per = dice_score(_va_p, _va_t, num_classes=3)
        _va_mean = sum(_va_per) / 3
        seg_history_pre.append(dict(
            epoch=_epoch + 1, train_loss=_tr_loss, train_acc=_tr_acc,
            val_loss=_va_loss, val_acc=_va_acc, val_dice=_va_mean, val_dice_per=_va_per,
        ))
        print(f'ep{_epoch + 1:02d} | tr {_tr_loss:.3f}/{_tr_acc:.3f} '
              f'| val {_va_loss:.3f}/{_va_acc:.3f} mean-Dice {_va_mean:.3f} per {_va_per}')
    return seg_criterion_pre, seg_history_pre, seg_pre_unet


@app.cell
def _(plt, seg_history_pre, seg_history_unet):
    _epochs = [h['epoch'] for h in seg_history_unet]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(_epochs, [h['val_dice'] for h in seg_history_unet], label='SmallUNet (scratch)')
    _axes[0].plot(_epochs, [h['val_dice'] for h in seg_history_pre],  label='UNet (pretrained encoder)')
    _axes[0].set(title='Val mean Dice', xlabel='Epoch', ylabel='Dice'); _axes[0].legend()
    _classes = ['background', 'cytoplasm', 'nucleus']
    for _ci, _name in enumerate(_classes):
        _axes[1].plot(_epochs, [h['val_dice_per'][_ci] for h in seg_history_unet], '--', label=f'{_name} (scratch)')
        _axes[1].plot(_epochs, [h['val_dice_per'][_ci] for h in seg_history_pre],  '-',  label=f'{_name} (pretrained)')
    _axes[1].set(title='Val per-class Dice', xlabel='Epoch', ylabel='Dice'); _axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEG_CMAP,
    device,
    dice_score,
    plt,
    run_epoch_seg,
    seg_class_names,
    seg_criterion_pre,
    seg_pre_unet,
    seg_val_loader,
    torch,
):
    _, _va_acc, _va_t, _va_p = run_epoch_seg(seg_pre_unet, seg_val_loader, seg_criterion_pre, None, device)
    _mean_d, seg_per_d_pre = dice_score(_va_p, _va_t, num_classes=3)
    print(f'Pretrained UNet  pixel acc {_va_acc:.3f}  mean Dice {_mean_d:.3f}')
    for _c, _name in enumerate(seg_class_names):
        print(f'  {_name:>11} Dice : {seg_per_d_pre[_c]:.3f}')

    _images, _masks = next(iter(seg_val_loader))
    seg_pre_unet.eval()
    with torch.no_grad():
        _preds = seg_pre_unet(_images.to(device)).argmax(dim=1).cpu()
    _mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    _std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    _n = min(4, _images.shape[0])
    _fig, _axes = plt.subplots(_n, 3, figsize=(9, 3 * _n))
    for _i in range(_n):
        _disp = (_images[_i] * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()
        _axes[_i, 0].imshow(_disp);                                     _axes[_i, 0].axis('off'); _axes[_i, 0].set_title('image')
        _axes[_i, 1].imshow(_masks[_i], cmap=SEG_CMAP, vmin=0, vmax=2); _axes[_i, 1].axis('off'); _axes[_i, 1].set_title('ground truth')
        _axes[_i, 2].imshow(_preds[_i], cmap=SEG_CMAP, vmin=0, vmax=2); _axes[_i, 2].axis('off'); _axes[_i, 2].set_title('Pretrained pred')
    plt.suptitle('Pretrained-encoder UNet predictions', fontsize=11)
    plt.tight_layout()
    plt.show()
    return (seg_per_d_pre,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 14 — Picking the right metric for the downstream task

    If the downstream goal is to count cells by detecting **nuclei**, which validation metric is most relevant?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Background Dice.": "a",
            "Nucleus Dice.": "b",
            "Training pixel accuracy.": "c",
            "Image-level classification accuracy.": "d",
        }
    q_seg_nucleus = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_NUCLEUS", _options),
    )
    q_seg_nucleus
    return (q_seg_nucleus,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 15 — Segmentation Comparison

    Read the per-class Dice scores printed below for the three models — Otsu
    (classical baseline, binary only), SmallUNet (from scratch), and the
    UNet with a pretrained ResNet18 encoder. The two short MCs that follow
    test the conclusions you should draw from those numbers.

    **(a)** Which model gives the best **nucleus** Dice on this run, and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Pretrained-encoder UNet — ImageNet features transfer well to small high-contrast structures like nuclei.": "a",
            "SmallUNet from scratch — random initialisation avoids overfitting on small data.": "b",
            "Otsu — automatic thresholding handles nucleus intensity directly.": "c",
            "All three are roughly equal on nucleus Dice.": "d",
        }
    q_seg_e15a = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_E15A", _options),
    )
    q_seg_e15a
    return (q_seg_e15a,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(b)** What additional check is most important before deploying the chosen
    segmentation model?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Train for more epochs to drive validation Dice higher.": "a",
            "Inspect qualitative predictions on edge cases (overlapping cells, staining variation, out-of-focus images) and report per-class Dice on the held-out test set.": "b",
            "Switch from Dice to pixel accuracy.": "c",
            "Skip — visualization is unreliable for segmentation, only Dice matters.": "d",
        }
    q_seg_e15b = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_SEG_E15B", _options),
    )
    q_seg_e15b
    return (q_seg_e15b,)


@app.cell
def _(seg_per_d_pre, seg_per_d_scratch):
    print('SmallUNet (scratch)        per-class Dice [bg, cyto, nuc]:', seg_per_d_scratch)
    print('UNet (pretrained encoder)  per-class Dice [bg, cyto, nuc]:', seg_per_d_pre)
    print(f'Mean Dice -- scratch: {sum(seg_per_d_scratch)/3:.3f}   pretrained: {sum(seg_per_d_pre)/3:.3f}')
    return


@app.cell
def _(
    device,
    dice_score,
    otsu_mean,
    run_epoch_seg,
    seg_criterion,
    seg_criterion_pre,
    seg_pre_unet,
    seg_unet,
    seg_val_loader,
):
    # Binary-collapse comparison vs. Otsu: collapse {cyto, nuc} -> 1 in both UNet
    # predictions and the ground truth, then compute Dice on the same val split
    # Otsu was evaluated against (the whole dataset).
    def _binary_dice(model):
        _, _, vt, vp = run_epoch_seg(model, seg_val_loader, seg_criterion, None, device)
        m, _ = dice_score((vp > 0).long(), (vt > 0).long(), num_classes=2)
        return m

    scratch_bin = _binary_dice(seg_unet)
    # Pretrained UNet uses its own (weighted) criterion; same val_loader.
    _, _, vt, vp = run_epoch_seg(seg_pre_unet, seg_val_loader, seg_criterion_pre, None, device)
    pre_bin, _ = dice_score((vp > 0).long(), (vt > 0).long(), num_classes=2)

    print(f'Binary cell-vs-bg Dice (val split):')
    print(f'  Otsu (whole dataset)        : {otsu_mean:.4f}')
    print(f'  SmallUNet (scratch, val)    : {scratch_bin:.4f}')
    print(f'  Pretrained UNet (val)       : {pre_bin:.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section C — Multiple Instance Learning on synthetic slides

    Each cell image becomes one *instance*. We synthesise slide-like *bags* of K=5 cell images
    and ask one bag-level question:

    > Does this slide contain at least one **eosinophil** (the rare class)?

    This is the classical MIL assumption (`bag positive iff any instance positive`) and mirrors
    a real haematology screening workflow. All functions and classes you need to implement live in **`mil.py`**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 — Understanding MIL

    ### Exercise 1 — Task framing

    Section C steps from one cell at a time to a *bag* of cells per training
    example. Each bag carries a single bag-level label (does it contain at least
    one eosinophil?), and the model never sees per-instance labels at training
    time.

    **Which formulation correctly describes the supervised learning problem in Section C?**
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Input x = a bag of K cell-crop instances; output y = a single binary bag-level label (1 if the bag contains ≥1 eosinophil, else 0); task = bag-level binary classification with weak supervision (no instance labels seen during training).": "a",
            "Input x = a single cell-crop image; output y = the WBC cell type; task = multi-class classification.": "b",
            "Input x = a whole-slide image; output y = a pixel-wise mask marking eosinophils; task = semantic segmentation.": "c",
            "Input x = a bag of K instances; output y = a length-K vector of instance-level labels; task = fully supervised instance classification.": "d",
        }
    q_mil_taskframing = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_TASKFRAMING", _options),
    )
    q_mil_taskframing
    return (q_mil_taskframing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2 — Hidden information

    The data-generating process knows which cells are eosinophils.
    What is hidden from the MIL model during training?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "The bag label.": "a",
            "The number of instances per bag.": "b",
            "Which individual instance caused the positive bag label.": "c",
            "The image features for each cell.": "d",
        }
    q_mil_hidden = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_HIDDEN", _options),
    )
    q_mil_hidden
    return (q_mil_hidden,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3 — Permutation invariance

    Why should a MIL pooling function be permutation-invariant?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Changing the order of cells in a bag should not change the bag-level prediction.": "a",
            "Cells are sorted by clinical importance.": "b",
            "The first cell in the bag should always receive the highest weight.": "c",
            "Segmentation masks are unordered.": "d",
        }
    q_mil_perm = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_PERM", _options),
    )
    q_mil_perm
    return (q_mil_perm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2 — Constructing slide-like bags

    K=5 cells per bag, half positive (≥1 eosinophil) / half negative. Sampling is with
    replacement at the bag level — only 22 eosinophil images exist, so each appears in many
    positive bags.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
        **Background — witness rate (no exercise).**
        Each positive bag here has exactly 1 eosinophil among K=5 cells — a witness rate of
        **20%**. The lecture warns that real medical bags often have witness rates
        **<1%** ("needle-in-a-haystack"); this synthetic regime is intentionally easier so the
        pooling lesson is visible in a few hundred bags.
        """),
        kind="info",
    )
    return


@app.cell
def _(LABELS_CSV, make_bags, np, pd):
    mil_df = pd.read_csv(LABELS_CSV)
    mil_df.columns = ['image_id', 'label']
    mil_df = mil_df[mil_df['label'] != 5].copy()
    mil_df['label'] = mil_df['label'] - 1

    mil_image_ids = mil_df['image_id'].astype(int).tolist()
    _labels       = mil_df['label'].astype(int).tolist()

    mil_all_bags = make_bags(mil_image_ids, _labels, k=5, n_bags=400, seed=42)

    _pos = sum(1 for _, y in mil_all_bags if y == 1)
    print(f'Built {len(mil_all_bags)} bags  (pos={_pos}, neg={len(mil_all_bags) - _pos}, K=5)')

    _n      = len(mil_all_bags)
    _n_test = int(round(_n * 0.20))
    _n_val  = int(round(_n * 0.20))
    mil_test_bags  = mil_all_bags[:_n_test]
    mil_val_bags   = mil_all_bags[_n_test:_n_test + _n_val]
    mil_train_bags = mil_all_bags[_n_test + _n_val:]
    print(f'Splits (train/val/test): {len(mil_train_bags)} / {len(mil_val_bags)} / {len(mil_test_bags)}')
    print('Pos fraction per split:',
          f'train={np.mean([y for _, y in mil_train_bags]):.2f}',
          f'val={np.mean([y for _, y in mil_val_bags]):.2f}',
          f'test={np.mean([y for _, y in mil_test_bags]):.2f}')
    return mil_df, mil_image_ids, mil_test_bags, mil_train_bags, mil_val_bags


@app.cell
def _(IMAGES_DIR, Image, WBC_CLASS_NAMES, mil_df, mil_train_bags, plt):
    _pos_idx = next(i for i, (_, y) in enumerate(mil_train_bags) if y == 1)
    _neg_idx = next(i for i, (_, y) in enumerate(mil_train_bags) if y == 0)
    _label_lookup = dict(zip(mil_df['image_id'].astype(int).tolist(),
                             mil_df['label'].astype(int).tolist()))

    _fig, _axes = plt.subplots(2, 5, figsize=(12, 5.5))
    for _row, (_bag_idx, _header) in enumerate([(_pos_idx, 'positive bag'), (_neg_idx, 'negative bag')]):
        _ids, _y = mil_train_bags[_bag_idx]
        for _col, _cid in enumerate(_ids):
            _img = Image.open(IMAGES_DIR / f'{int(_cid):03d}.bmp')
            _axes[_row, _col].imshow(_img)
            _axes[_row, _col].axis('off')
            _cls = WBC_CLASS_NAMES[_label_lookup[int(_cid)]]
            _axes[_row, _col].set_title(f'id {_cid}\n{_cls}', fontsize=9)
        _axes[_row, 0].set_ylabel(f'{_header} (y={_y})', fontsize=11, fontweight='bold')
    plt.suptitle('Two example bags — a positive bag contains ≥1 eosinophil', fontsize=12)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3 — Frozen feature cache

    Every pooling model below sees the *same* 512-d frozen ResNet18 features per cell. This
    isolates the lesson: only the pooling logic differs.
    """)
    return


@app.cell
def _(IMAGES_DIR, device, extract_features, mil_image_ids):
    mil_features = extract_features(IMAGES_DIR, mil_image_ids, device=device, image_size=224)
    print(f'Cached {len(mil_features)} feature vectors of dim {next(iter(mil_features.values())).shape[0]}')
    return (mil_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4 — Three pooling strategies

    Implement `MeanPoolMIL`, `MaxPoolMIL`, and `AttentionPoolMIL` in
    `mil.py`, then run the cell below to train all three from the same features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 4 — Pooling and the any-positive assumption

    For the rule *"a bag is positive if it contains at least one rare-class cell"*,
    which pooling strategy is most naturally aligned with the task? Predict before training.
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Mean pooling — all cells contribute equally to the bag label.": "a",
            "Max pooling — one strongly positive instance can make the whole bag positive.": "b",
            "Random pooling — the rare-cell location is unknown.": "c",
            "Pixel accuracy — MIL is a pixel-level task.": "d",
        }
    q_mil_anypos = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_ANYPOS", _options),
    )
    q_mil_anypos
    return (q_mil_anypos,)


@app.cell
def _(
    AttentionPoolMIL,
    BagDataset,
    DataLoader,
    MaxPoolMIL,
    MeanPoolMIL,
    device,
    mil_features,
    mil_test_bags,
    mil_train_bags,
    mil_val_bags,
    nn,
    roc_auc_score,
    run_mil_epoch,
    torch,
):
    _train_ds = BagDataset(mil_train_bags, mil_features)
    _val_ds   = BagDataset(mil_val_bags,   mil_features)
    _test_ds  = BagDataset(mil_test_bags,  mil_features)
    mil_train_loader = DataLoader(_train_ds, batch_size=32, shuffle=True)
    mil_val_loader   = DataLoader(_val_ds,   batch_size=32, shuffle=False)
    mil_test_loader  = DataLoader(_test_ds,  batch_size=32, shuffle=False)

    _feat_dim   = next(iter(mil_features.values())).shape[0]
    mil_results   = {}
    mil_histories = {}
    mil_models    = {}
    for _name, _ctor in [
        ('mean',      lambda: MeanPoolMIL(_feat_dim)),
        ('max',       lambda: MaxPoolMIL(_feat_dim)),
        ('attention', lambda: AttentionPoolMIL(_feat_dim)),
    ]:
        torch.manual_seed(42)
        _model = _ctor().to(device)
        _opt   = torch.optim.Adam(_model.parameters(), lr=1e-3)
        _crit  = nn.CrossEntropyLoss()
        _hist  = []
        for _ep in range(30):
            _tl, _ta, _, _ = run_mil_epoch(_model, mil_train_loader, _crit, _opt, device)
            _vl, _va, _vt, _vp = run_mil_epoch(_model, mil_val_loader, _crit, None, device)
            _vauc = roc_auc_score(_vt, _vp) if len(set(_vt)) == 2 else float('nan')
            _hist.append(dict(epoch=_ep + 1, train_loss=_tl, train_acc=_ta,
                              val_loss=_vl, val_acc=_va, val_auc=_vauc))
        _, _te_acc, _te_t, _te_p = run_mil_epoch(_model, mil_test_loader, _crit, None, device)
        _te_auc = roc_auc_score(_te_t, _te_p) if len(set(_te_t)) == 2 else float('nan')
        mil_results[_name]   = dict(test_acc=_te_acc, test_auc=_te_auc)
        mil_histories[_name] = _hist
        mil_models[_name]    = _model
        print(f'{_name:>9}  -> test acc {_te_acc:.3f}   test AUROC {_te_auc:.3f}')
    return mil_histories, mil_models


@app.cell
def _(mil_histories, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _name, _hist in mil_histories.items():
        _epochs = [h['epoch'] for h in _hist]
        _axes[0].plot(_epochs, [h['val_loss'] for h in _hist], label=_name)
        _axes[1].plot(_epochs, [h['val_auc']  for h in _hist], label=_name)
    _axes[0].set(title='Validation loss',  xlabel='Epoch', ylabel='Loss');  _axes[0].legend()
    _axes[1].set(title='Validation AUROC', xlabel='Epoch', ylabel='AUROC'); _axes[1].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5 — Mean pooling limitation

    Why can mean pooling struggle when only one of several instances is responsible for a positive bag label?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "The signal from the one rare cell may be diluted by the other non-rare cells.": "a",
            "Mean pooling cannot process fixed-size bags.": "b",
            "Mean pooling requires segmentation masks.": "c",
            "Mean pooling is not permutation-invariant.": "d",
        }
    q_mil_dilution = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_DILUTION", _options),
    )
    q_mil_dilution
    return (q_mil_dilution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5 — Inside the attention pooler

    Only `AttentionPoolMIL` returns instance-level scores. Below: a few *positive* validation
    bags with attention overlaid — bright bars indicate the cell the model thinks is the
    eosinophil. A well-trained attention head puts almost all its mass on the single eosinophil
    in each positive bag.
    """)
    return


@app.cell
def _(
    IMAGES_DIR,
    Image,
    WBC_CLASS_NAMES,
    device,
    mil_df,
    mil_features,
    mil_models,
    mil_val_bags,
    plt,
    torch,
):
    _label_lookup = dict(zip(mil_df['image_id'].astype(int).tolist(),
                             mil_df['label'].astype(int).tolist()))
    _attn_model = mil_models['attention'].eval()

    _pos_indices = [i for i, (_, y) in enumerate(mil_val_bags) if y == 1][:4]
    _n_rows = len(_pos_indices)
    _fig, _axes = plt.subplots(_n_rows, 5, figsize=(12, 2.6 * _n_rows))
    if _n_rows == 1:
        _axes = _axes[None, :]

    with torch.no_grad():
        for _row, _bag_idx in enumerate(_pos_indices):
            _ids, _y = mil_val_bags[_bag_idx]
            _feats = torch.stack([mil_features[int(i)] for i in _ids]).unsqueeze(0).to(device)
            _, _weights = _attn_model(_feats)
            _w = _weights.squeeze(0).cpu().numpy()
            for _col, _cid in enumerate(_ids):
                _img = Image.open(IMAGES_DIR / f'{int(_cid):03d}.bmp')
                _ax = _axes[_row, _col]
                _ax.imshow(_img)
                _ax.set_xticks([]); _ax.set_yticks([])
                _ax.set_box_aspect(1)
                _cls = WBC_CLASS_NAMES[_label_lookup[int(_cid)]]
                _is_eos = _cls == 'eosinophil'
                _ax.set_title(
                    f'cell type: {_cls}\nattention = {_w[_col]:.2f}',
                    fontsize=10,
                    color=('crimson' if _is_eos else 'black'),
                    fontweight=('bold' if _is_eos else 'normal'),
                )
                if _is_eos:
                    for _spine in _ax.spines.values():
                        _spine.set_edgecolor('crimson')
                        _spine.set_linewidth(2.5)
            _axes[_row, 0].set_ylabel(
                f'positive bag #{_row + 1}', fontsize=11, fontweight='bold',
            )

    plt.suptitle(
        "Attention weights per cell on positive bags\n"
        "(crimson border = ground-truth eosinophil; attention values are a softmax over the 5 cells in each bag and sum to 1.0)",
        fontsize=10,
        y=1.0,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 6 — Attention weight interpretation

    In an attention-based MIL model, what does a high attention weight on one cell usually suggest?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "The model considered that instance important for the bag-level prediction.": "a",
            "The instance is guaranteed to be the true rare-class cell.": "b",
            "The instance must have the best segmentation mask.": "c",
            "The bag must be negative.": "d",
        }
    q_mil_attn = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_ATTN", _options),
    )
    q_mil_attn
    return (q_mil_attn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 7 — Comparing pooling strategies

    Two MCs about the three poolers — one about model capacity, one about
    what attention should do at inference time.

    **(a)** Compare the three pooling models by *trainable* parameter count
    (excluding the frozen ResNet18 backbone). Which has the most parameters?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Mean pooling — averaging features adds learnable weights.": "a",
            "Max pooling — element-wise max is parameter-heavy.": "b",
            "Attention pooling — the gated attention head (parallel tanh & sigmoid branches → score → softmax) adds parameters that mean and max do not have.": "c",
            "All three have identical parameter counts.": "d",
        }
    q_mil_pool_params = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_POOL_PARAMS", _options),
    )
    q_mil_pool_params
    return (q_mil_pool_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(b)** On a *positive* bag containing exactly one eosinophil among K = 5
    cells, what should a *well-trained* attention-MIL pooler do with its
    attention weights?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Spread attention uniformly across all K cells (≈ 1/K each).": "a",
            "Concentrate most of its attention mass on the eosinophil, leaving the other K−1 instances with low attention weights.": "b",
            "Concentrate on whichever cell is largest in the image, regardless of class.": "c",
            "Concentrate on the cell with the highest pixel brightness.": "d",
        }
    q_mil_attn_behavior = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_ATTN_BEHAVIOR", _options),
    )
    q_mil_attn_behavior
    return (q_mil_attn_behavior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 8 — AUROC as a ranking metric

    AUROC equals the probability that a random *positive* example is scored
    higher than a random *negative* example. Now suppose model B's score is a
    monotone transformation of model A's: $\text{score}_B = \log(1 + \text{score}_A)$.
    Both models therefore produce the same *ranking* on every test bag.

    **(a)** Which model has the higher validation AUROC?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Model A.": "a",
            "Model B.": "b",
            "Both have the same AUROC.": "c",
            "Cannot tell without seeing the raw scores.": "d",
        }
    q_mil_auroc_rank = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_AUROC_RANK", _options),
    )
    q_mil_auroc_rank
    return (q_mil_auroc_rank,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(b)** If both models reach AUROC = 0.95 on validation, can you conclude
    that their predicted probabilities are well calibrated (i.e. a score of 0.7
    really means a 70% chance of a positive bag)?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Yes — AUROC ≥ 0.95 implies good calibration.": "a",
            "No — AUROC measures only ranking; calibration is a separate property and must be measured separately.": "b",
            "Only after applying softmax.": "c",
            "Yes, but only on the test set.": "d",
        }
    q_mil_auroc_calib = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_AUROC_CALIB", _options),
    )
    q_mil_auroc_calib
    return (q_mil_auroc_calib,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 9 — PR vs ROC under class imbalance

    Imagine extending the MIL setup to a *real* slide-screening problem: each
    whole slide contains thousands of cells and the rare-cell prevalence is far
    lower than in our synthetic bags. Two models report:

    | Model | AUROC | AUPRC |
    |-------|-------|-------|
    | A     | 0.92  | 0.04  |
    | B     | 0.88  | 0.31  |

    **(a)** Which model is more useful for the screening task?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Model A — higher AUROC.": "a",
            "Model B — much higher AUPRC at very low prevalence.": "b",
            "Both equally useful.": "c",
            "Neither — AUROC and AUPRC are uncorrelated.": "d",
        }
    q_mil_proc_ship = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_PROC_SHIP", _options),
    )
    q_mil_proc_ship
    return (q_mil_proc_ship,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(b)** With AUPRC ≈ 0.04, precision is roughly 0.04 at moderate recall.
    Approximately how many false alarms does the model raise per true positive?

    $$\text{FP per TP} = \frac{1}{\text{PPV}} - 1$$
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_default):
    q_mil_proc_fpratio = mo.ui.number(
        start=0, stop=200, step=1,
        label="False alarms per true positive:",
        value=submission_default("Q_MIL_PROC_FPRATIO"),
    )
    q_mil_proc_fpratio
    return (q_mil_proc_fpratio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **(c)** Why didn't AUROC catch this screening problem?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "AUROC integrates TPR vs FPR; a tiny absolute change in FPR hides a huge change in raw FP count when prevalence is very low. AUROC is prevalence-blind.": "a",
            "AUROC is broken under class imbalance.": "b",
            "AUROC counts false positives directly.": "c",
            "The test set was too small.": "d",
        }
    q_mil_proc_why = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_PROC_WHY", _options),
    )
    q_mil_proc_why
    return (q_mil_proc_why,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 10 — Synthetic bags vs real slides

    Why is our synthetic MIL setup easier than a real digital-pathology MIL task?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Synthetic bags are cleaner and smaller than real slides, with fewer artifacts and fewer irrelevant cells.": "a",
            "Synthetic bags are always harder than real whole-slide images.": "b",
            "Synthetic bags automatically solve patient-level data leakage.": "c",
            "Synthetic bags remove the need for train/validation/test splits.": "d",
        }
    q_mil_synth = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_SYNTH", _options),
    )
    q_mil_synth
    return (q_mil_synth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 6 — Beyond ABMIL: CLAM

    The lecture introduced **CLAM** (Clustering-constrained Attention Multiple-instance
    Learning, Lu 2021) as a follow-up to ABMIL: same gated-attention pooler as
    ABMIL, plus a *per-class* auxiliary classifier on the cell embeddings that
    is trained with a hinge loss alongside the bag cross-entropy. We will not
    implement CLAM, but the question below checks that you understand the
    trickiest piece — how the auxiliary head gets supervised.

    ### Exercise 11 — Where do CLAM's per-cell pseudo-labels come from?

    CLAM's per-cell auxiliary classifier needs `+1` / `-1` labels at the cell
    level, but the dataset only has bag-level labels. How does CLAM produce
    these pseudo-labels during training?
    """)
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "Each cell gets the bag's label copied onto it (as in instance-space MIL); the per-cell classifier sees the noisy labels directly.": "a",
            "For a bag of class c, the top-B attention-ranked cells get +1 and the bottom-B get -1; bags that do not have class c contribute every cell as -1 for the class-c head.": "b",
            "A pretrained pathology foundation model is used to label every cell before training, and CLAM treats those labels as ground truth.": "c",
            "CLAM samples cell labels uniformly at random from {-1, +1} so the auxiliary loss acts as a regulariser.": "d",
        }
    q_mil_clam_pseudo = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_CLAM_PSEUDO", _options),
    )
    q_mil_clam_pseudo
    return (q_mil_clam_pseudo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 7 — Pooling fingerprints
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""Each pooling strategy leaves a recognisable signature when you visualise the per-cell weights it assigns to a bag. ."""),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 12 — Match the panel to the pooling method

    The figure below shows three heatmaps over the *same* positive bag (5 WBCs, the eosinophil sits in
    position 3). Each row was produced by a different pooler — the redness of the overlay on a cell is proportional to the weight that pooler gave it when forming the bag prediction.
    """)
    return


@app.cell(hide_code=True)
def _(IMAGES_DIR, Image, np, pd, plt):
    # Deterministic toy bag: 4 non-eosinophils + 1 eosinophil placed at
    # position 3. We pick the first available cells of each label so that the
    # exercise is identical for every student.
    _df = pd.read_csv(
        IMAGES_DIR.parent / "Class Labels of Dataset 1.csv",
        names=["image_id", "label"], header=0,
    )
    _eos = _df[_df["label"] == 4]["image_id"].astype(int).head(1).tolist()
    _other = _df[(_df["label"] != 4) & (_df["label"] != 5)]["image_id"].astype(int).head(4).tolist()
    _bag = [_other[0], _other[1], _eos[0], _other[2], _other[3]]   # witness at idx 2

    # Hard-coded pooling weights for the same bag. These are the canonical
    # signatures: uniform / binary / soft-peaked.
    _w_mean = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    _w_max  = np.array([0.00, 0.00, 1.00, 0.00, 0.00])
    _w_attn = np.array([0.04, 0.06, 0.85, 0.03, 0.02])

    # Panel order is fixed so the MCQ key is stable. Panel A = attention,
    # Panel B = mean, Panel C = max  ->  correct mapping is choice (c) below.
    _panels = [("Panel A", _w_attn), ("Panel B", _w_mean), ("Panel C", _w_max)]

    _fig, _axes = plt.subplots(3, 5, figsize=(12, 6.6))
    for _row, (_name, _w) in enumerate(_panels):
        for _col, _cid in enumerate(_bag):
            _img = np.array(Image.open(IMAGES_DIR / f"{int(_cid):03d}.bmp"))
            _ax = _axes[_row, _col]
            _ax.imshow(_img)
            # Red overlay; alpha encodes the pooling weight on this cell.
            _overlay = np.zeros((_img.shape[0], _img.shape[1], 4))
            _overlay[..., 0] = 1.0           # red channel
            _overlay[..., 3] = float(_w[_col]) * 0.65  # alpha proportional to weight
            _ax.imshow(_overlay)
            _ax.set_xticks([]); _ax.set_yticks([])
            _ax.set_box_aspect(1)
            _ax.set_title(f"cell {_col + 1}\nweight = {_w[_col]:.2f}", fontsize=9)
        _axes[_row, 0].set_ylabel(_name, fontsize=11, fontweight="bold")

    plt.suptitle(
        "Three pooling fingerprints on the same positive bag\n"
        "(witness = the eosinophil at cell 3; redness = pooling weight)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, submission_radio_default):
    _options = {
            "A = mean,      B = max,       C = attention": "a",
            "A = mean,      B = attention, C = max":       "b",
            "A = attention, B = mean,      C = max":       "c",
            "A = max,       B = attention, C = mean":      "d",
        }
    q_mil_pool_fingerprints = mo.ui.radio(
        options=_options,
        label="Your answer:",
        value=submission_radio_default("Q_MIL_POOL_FINGERPRINTS", _options),
    )
    q_mil_pool_fingerprints
    return (q_mil_pool_fingerprints,)


@app.cell(hide_code=True)
def _(
    mo,
    q19_fn,
    q19_fp,
    q19_tn,
    q19_tp,
    q2_1,
    q6a,
    q6b,
    q6c,
    q_cls_acc_misleading,
    q_cls_baso_adopt,
    q_cls_baso_count,
    q_cls_baso_ppv,
    q_cls_cvbias_assum,
    q_cls_cvbias_dir,
    q_cls_f1_calc,
    q_cls_formulas,
    q_cls_macro_micro,
    q_cls_mcc_vs_f1,
    q_cls_pr_scenario,
    q_cls_taskframing,
    q_majority,
    q_mil_anypos,
    q_mil_attn,
    q_mil_attn_behavior,
    q_mil_auroc_calib,
    q_mil_auroc_rank,
    q_mil_clam_pseudo,
    q_mil_dilution,
    q_mil_hidden,
    q_mil_perm,
    q_mil_pool_fingerprints,
    q_mil_pool_params,
    q_mil_proc_fpratio,
    q_mil_proc_ship,
    q_mil_proc_why,
    q_mil_synth,
    q_mil_taskframing,
    q_seg_aug_asym,
    q_seg_dice_agg,
    q_seg_diceloss,
    q_seg_e15a,
    q_seg_e15b,
    q_seg_imbalance,
    q_seg_nucleus,
    q_seg_paired,
    q_seg_perclass,
    q_seg_pixacc,
    q_seg_resize,
    q_seg_taskframing,
):
    import json as _json
    from pathlib import Path as _Path

    submission = {
        # --- Section A ---
        "Q_CLS_TASKFRAMING":     q_cls_taskframing.value,
        "Q_CLS_SPLIT":           q2_1.value,
        "Q_CLS_FORMULAS":        q_cls_formulas.value,
        "Q_CLS_F1_CALC":         q_cls_f1_calc.value,
        "Q_CLS_PR_SCENARIO":     q_cls_pr_scenario.value,
        "Q_CLS_MAJORITY":        q_majority.value,
        "Q_CLS_ACC_MISLEADING":  q_cls_acc_misleading.value,
        "Q_CLS_MACRO_MICRO":     q_cls_macro_micro.value,
        "Q_CLS_E14A":            q6a.value,
        "Q_CLS_E14B":            q6b.value,
        "Q_CLS_E14C":            q6c.value,
        "Q_CLS_CVBIAS_DIR":      q_cls_cvbias_dir.value,
        "Q_CLS_CVBIAS_ASSUM":    q_cls_cvbias_assum.value,
        "Q_CLS_BASO_COUNT":      q_cls_baso_count.value,
        "Q_CLS_BASO_PPV":        q_cls_baso_ppv.value,
        "Q_CLS_BASO_ADOPT":      q_cls_baso_adopt.value,
        "Q_CLS_MCC_VS_F1":       q_cls_mcc_vs_f1.value,
        "Q19_TP":                q19_tp.value,
        "Q19_FN":                q19_fn.value,
        "Q19_FP":                q19_fp.value,
        "Q19_TN":                q19_tn.value,
        # --- Section B ---
        "Q_SEG_TASKFRAMING":     q_seg_taskframing.value,
        "Q_SEG_PIXACC":          q_seg_pixacc.value,
        "Q_SEG_AUG_ASYM":        q_seg_aug_asym.value,
        "Q_SEG_DICE_AGG":        q_seg_dice_agg.value,
        "Q_SEG_DICELOSS":        q_seg_diceloss.value,
        "Q_SEG_PERCLASS":        q_seg_perclass.value,
        "Q_SEG_IMBALANCE":       q_seg_imbalance.value,
        "Q_SEG_RESIZE":          q_seg_resize.value,
        "Q_SEG_PAIRED":          q_seg_paired.value,
        "Q_SEG_NUCLEUS":         q_seg_nucleus.value,
        "Q_SEG_E15A":            q_seg_e15a.value,
        "Q_SEG_E15B":            q_seg_e15b.value,
        # --- Section C ---
        "Q_MIL_TASKFRAMING":     q_mil_taskframing.value,
        "Q_MIL_HIDDEN":          q_mil_hidden.value,
        "Q_MIL_PERM":            q_mil_perm.value,
        "Q_MIL_ANYPOS":          q_mil_anypos.value,
        "Q_MIL_DILUTION":        q_mil_dilution.value,
        "Q_MIL_ATTN":            q_mil_attn.value,
        "Q_MIL_POOL_PARAMS":     q_mil_pool_params.value,
        "Q_MIL_ATTN_BEHAVIOR":   q_mil_attn_behavior.value,
        "Q_MIL_AUROC_RANK":      q_mil_auroc_rank.value,
        "Q_MIL_AUROC_CALIB":     q_mil_auroc_calib.value,
        "Q_MIL_PROC_SHIP":       q_mil_proc_ship.value,
        "Q_MIL_PROC_FPRATIO":    q_mil_proc_fpratio.value,
        "Q_MIL_PROC_WHY":        q_mil_proc_why.value,
        "Q_MIL_SYNTH":           q_mil_synth.value,
        "Q_MIL_CLAM_PSEUDO":     q_mil_clam_pseudo.value,
        "Q_MIL_POOL_FINGERPRINTS": q_mil_pool_fingerprints.value,
    }
    _Path("submission.json").write_text(_json.dumps(submission, indent=2))
    n_unanswered = sum(1 for v in submission.values() if v is None)
    mo.md(
        f"**Submission auto-saved to `submission.json`** "
        f"({len(submission)} questions, {n_unanswered} unanswered)."
    )
    return


if __name__ == "__main__":
    app.run()
