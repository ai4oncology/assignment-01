import json
import pathlib

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Notebook -- submission completeness
# ---------------------------------------------------------------------------

class TestNotebookAnswers:
    """Sanity-check that the student has actually answered every widget.

    `submission.json` is auto-written by the export cell at the end of
    `notebook.py`. A widget the student never touched stays as `None`,
    which would silently lose marks on the per-question tests below.
    This test surfaces all unanswered questions in one shot.
    """

    def test_submission_complete(self):
        path = pathlib.Path("submission.json")
        assert path.exists(), (
            "submission.json not found -- open notebook.py in marimo, "
            "fill in the widgets, and the export cell will write the file."
        )
        data = json.loads(path.read_text())
        unanswered = sorted(k for k, v in data.items() if v is None)
        assert not unanswered, (
            f"{len(unanswered)} question(s) unanswered in submission.json: "
            f"{unanswered}. Open the notebook and click through every widget."
        )


# ---------------------------------------------------------------------------
# Classification -- compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_known_values(self):
        from classification import compute_metrics
        p, r, f1 = compute_metrics(42, 8, 18)
        assert abs(p - 0.84)   < 1e-3
        assert abs(r - 0.70)   < 1e-3
        assert abs(f1 - 0.7636) < 1e-3

    def test_perfect_predictions(self):
        from classification import compute_metrics
        p, r, f1 = compute_metrics(tp=50, fp=0, fn=0)
        assert p  == pytest.approx(1.0)
        assert r  == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_return_type(self):
        from classification import compute_metrics
        result = compute_metrics(10, 5, 5)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Classification -- compute_mcc
# ---------------------------------------------------------------------------

class TestComputeMCC:

    def test_known_values(self):
        # Same TP/FP/FN as TestComputeMetrics, with TN=132 added.
        # MCC = (42*132 - 8*18) / sqrt(50*60*140*150) = 5400 / sqrt(63_000_000)
        #     = 5400 / 7937.2539... = 0.6803...
        from classification import compute_mcc
        assert abs(compute_mcc(tp=42, tn=132, fp=8, fn=18) - 0.6803) < 1e-3

    def test_perfect_prediction(self):
        from classification import compute_mcc
        assert compute_mcc(tp=10, tn=20, fp=0, fn=0) == pytest.approx(1.0)

    def test_inverted_prediction_returns_minus_one(self):
        # All predictions wrong: every actual positive is predicted negative
        # and vice versa.
        from classification import compute_mcc
        assert compute_mcc(tp=0, tn=0, fp=10, fn=20) == pytest.approx(-1.0)

    def test_zero_denominator_returns_zero(self):
        # Predicts nothing positive AND no actual negatives -> denominator == 0.
        from classification import compute_mcc
        assert compute_mcc(tp=0, tn=0, fp=0, fn=10) == 0.0


# ---------------------------------------------------------------------------
# Classification -- compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:

    COUNTS = {"a": 100, "b": 200, "c": 300, "d": 400}

    def test_rarest_class_has_highest_weight(self):
        from classification import compute_class_weights
        weights = compute_class_weights(self.COUNTS)
        assert max(weights, key=weights.get) == "a"

    def test_balanced_gives_unit_weights(self):
        from classification import compute_class_weights
        balanced = {"a": 100, "b": 100, "c": 100}
        weights = compute_class_weights(balanced)
        assert all(abs(w - 1.0) < 1e-6 for w in weights.values())

    def test_returns_all_keys(self):
        from classification import compute_class_weights
        weights = compute_class_weights(self.COUNTS)
        assert set(weights.keys()) == set(self.COUNTS.keys())


# ---------------------------------------------------------------------------
# Classification -- SimpleCNN
# ---------------------------------------------------------------------------

class TestSimpleCNN:

    def test_output_shape(self):
        from classification import SimpleCNN
        model = SimpleCNN(num_classes=4).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 224, 224))
        assert out.shape == (2, 4)

    def test_num_classes_respected(self):
        from classification import SimpleCNN
        model = SimpleCNN(num_classes=7).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 64, 64))
        assert out.shape == (1, 7)

    def test_is_nn_module(self):
        from classification import SimpleCNN
        assert isinstance(SimpleCNN(), nn.Module)


# ---------------------------------------------------------------------------
# Segmentation -- dice_score
# ---------------------------------------------------------------------------

class TestDiceScore:

    def test_perfect_prediction_returns_one(self):
        from segmentation import dice_score
        target = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
        mean, per_class = dice_score(target.clone(), target, num_classes=3)
        assert mean == pytest.approx(1.0)
        assert per_class == [1.0, 1.0, 1.0]

    def test_collapsed_to_background(self):
        from segmentation import dice_score
        target = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
        pred   = torch.zeros_like(target)
        mean, per_class = dice_score(pred, target, num_classes=3)
        # bg has 3 TP, 6 FP, 0 FN -> Dice = 6/(6+6) = 0.5
        assert per_class[0] == pytest.approx(0.5)
        assert per_class[1] == 0.0
        assert per_class[2] == 0.0


# ---------------------------------------------------------------------------
# Segmentation -- compute_pixel_class_weights
# ---------------------------------------------------------------------------

class TestPixelClassWeights:

    def test_rarest_class_has_highest_weight(self):
        from segmentation import compute_pixel_class_weights
        counts = {"background": 800_000, "cytoplasm": 200_000, "nucleus": 100_000}
        weights = compute_pixel_class_weights(counts)
        assert max(weights, key=weights.get) == "nucleus"


# ---------------------------------------------------------------------------
# Segmentation -- SmallUNet
# ---------------------------------------------------------------------------

class TestSmallUNet:

    def test_output_shape(self):
        from segmentation import SmallUNet
        model = SmallUNet(num_classes=3).eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 128, 128))
        assert out.shape == (2, 3, 128, 128)

    def test_num_classes_respected(self):
        from segmentation import SmallUNet
        model = SmallUNet(num_classes=5).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 64, 64))
        assert out.shape == (1, 5, 64, 64)


# ---------------------------------------------------------------------------
# MIL -- pooling models
# ---------------------------------------------------------------------------

class TestMILPoolers:

    @pytest.fixture
    def bag_features(self):
        # batch of 4 bags, each with 5 instances of 32-dim features
        return torch.randn(4, 5, 32)

    def test_mean_pool_output_shape(self, bag_features):
        from mil import MeanPoolMIL
        model = MeanPoolMIL(feature_dim=32).eval()
        with torch.no_grad():
            logits, weights = model(bag_features)
        assert logits.shape == (4, 2)
        assert weights is None

    def test_max_pool_output_shape(self, bag_features):
        from mil import MaxPoolMIL
        model = MaxPoolMIL(feature_dim=32).eval()
        with torch.no_grad():
            logits, weights = model(bag_features)
        assert logits.shape == (4, 2)
        assert weights is None

    def test_attention_returns_weights_summing_to_one(self, bag_features):
        from mil import AttentionPoolMIL
        model = AttentionPoolMIL(feature_dim=32).eval()
        with torch.no_grad():
            logits, weights = model(bag_features)
        assert logits.shape == (4, 2)
        assert weights.shape == (4, 5)
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Multiple-choice and numeric answers from submission.json
# ---------------------------------------------------------------------------
#
# `submission.json` is auto-written by the export cell at the end of
# `notebook.py`. To regenerate it, open the notebook in marimo, click through
# the radio / number widgets, and the file is updated reactively.


@pytest.fixture(scope="module")
def submission():
    path = pathlib.Path("submission.json")
    if not path.exists():
        pytest.skip(
            "submission.json not found — open notebook.py in marimo, fill "
            "in the widgets, and the export cell will write the file."
        )
    return json.loads(path.read_text())


def _require(submission, key):
    """Return submission[key] or fail loudly if it is missing / unanswered."""
    assert key in submission, f"{key} missing from submission.json"
    val = submission[key]
    assert val is not None, (
        f"{key} is unanswered in submission.json (None). "
        "Fill in the corresponding widget in the notebook."
    )
    return val


class TestAnswersMC:
    """Auto-graded multiple-choice and short-numeric questions."""

    # --- Section A ------------------------------------------------------

    def test_q_cls_taskframing(self, submission):
        assert _require(submission, "Q_CLS_TASKFRAMING") == "a"

    def test_q_cls_split(self, submission):
        assert _require(submission, "Q_CLS_SPLIT") == "b"

    def test_q_cls_formulas(self, submission):
        assert _require(submission, "Q_CLS_FORMULAS") == "b"

    def test_q_cls_f1_calc(self, submission):
        assert _require(submission, "Q_CLS_F1_CALC") == "c"

    def test_q_cls_pr_scenario(self, submission):
        assert _require(submission, "Q_CLS_PR_SCENARIO") == "b"

    def test_q_cls_majority(self, submission):
        assert _require(submission, "Q_CLS_MAJORITY") == "a"

    def test_q_cls_acc_misleading(self, submission):
        assert _require(submission, "Q_CLS_ACC_MISLEADING") == "a"

    def test_q_cls_macro_micro(self, submission):
        assert _require(submission, "Q_CLS_MACRO_MICRO") == "b"

    def test_q_cls_e14a(self, submission):
        assert _require(submission, "Q_CLS_E14A") == "c"

    def test_q_cls_e14b(self, submission):
        assert _require(submission, "Q_CLS_E14B") == "b"

    def test_q_cls_e14c(self, submission):
        assert _require(submission, "Q_CLS_E14C") == "c"

    def test_q_cls_cvbias_dir(self, submission):
        assert _require(submission, "Q_CLS_CVBIAS_DIR") == "a"

    def test_q_cls_cvbias_assum(self, submission):
        assert _require(submission, "Q_CLS_CVBIAS_ASSUM") == "a"

    def test_q_cls_baso_count(self, submission):
        assert abs(_require(submission, "Q_CLS_BASO_COUNT") - 55) <= 5

    def test_q_cls_baso_ppv(self, submission):
        assert abs(_require(submission, "Q_CLS_BASO_PPV") - 0.087) <= 0.02

    def test_q_cls_baso_adopt(self, submission):
        assert _require(submission, "Q_CLS_BASO_ADOPT") == "b"

    def test_q_cls_mcc_vs_f1(self, submission):
        assert _require(submission, "Q_CLS_MCC_VS_F1") == "b"

    def test_q19_f1_lies(self, submission):
        """Ex 19: construct a CM on the referred cohort (N=1000, 90% positive)
        such that F1 >= 0.90 and MCC <= 0.30. Many tuples satisfy this; we
        verify the constraints rather than match a specific answer."""
        import math
        tp = _require(submission, "Q19_TP")
        fn = _require(submission, "Q19_FN")
        fp = _require(submission, "Q19_FP")
        tn = _require(submission, "Q19_TN")
        for name, v in [("Q19_TP", tp), ("Q19_FN", fn),
                        ("Q19_FP", fp), ("Q19_TN", tn)]:
            assert isinstance(v, int) or (isinstance(v, float) and v.is_integer()), \
                f"{name} must be a non-negative integer"
            assert v >= 0, f"{name} must be non-negative"
        tp, fn, fp, tn = int(tp), int(fn), int(fp), int(tn)

        assert tp + fn + fp + tn == 1000, "totals must equal N=1000"
        assert tp + fn == 900, "true abnormals fixed at 900 (TP+FN=900)"
        assert fp + tn == 100, "true normals fixed at 100 (FP+TN=100)"

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = (tp * tn - fp * fn) / math.sqrt(denom_sq) if denom_sq > 0 else 0.0

        assert f1 >= 0.90,  f"F1 = {f1:.4f} but constraint is >= 0.90"
        assert mcc <= 0.30, f"MCC = {mcc:.4f} but constraint is <= 0.30"

    # --- Section B ------------------------------------------------------

    def test_q_seg_taskframing(self, submission):
        assert _require(submission, "Q_SEG_TASKFRAMING") == "a"

    def test_q_seg_pixacc(self, submission):
        assert _require(submission, "Q_SEG_PIXACC") == "a"

    def test_q_seg_aug_asym(self, submission):
        assert _require(submission, "Q_SEG_AUG_ASYM") == "b"

    def test_q_seg_dice_agg(self, submission):
        assert _require(submission, "Q_SEG_DICE_AGG") == "a"

    def test_q_seg_diceloss(self, submission):
        assert _require(submission, "Q_SEG_DICELOSS") == "b"

    def test_q_seg_perclass(self, submission):
        assert _require(submission, "Q_SEG_PERCLASS") == "a"

    def test_q_seg_imbalance(self, submission):
        assert _require(submission, "Q_SEG_IMBALANCE") == "a"

    def test_q_seg_resize(self, submission):
        assert _require(submission, "Q_SEG_RESIZE") == "b"

    def test_q_seg_paired(self, submission):
        assert _require(submission, "Q_SEG_PAIRED") == "c"

    def test_q_seg_nucleus(self, submission):
        assert _require(submission, "Q_SEG_NUCLEUS") == "b"

    def test_q_seg_e15a(self, submission):
        assert _require(submission, "Q_SEG_E15A") == "a"

    def test_q_seg_e15b(self, submission):
        assert _require(submission, "Q_SEG_E15B") == "b"

    # --- Section C ------------------------------------------------------

    def test_q_mil_taskframing(self, submission):
        assert _require(submission, "Q_MIL_TASKFRAMING") == "a"

    def test_q_mil_hidden(self, submission):
        assert _require(submission, "Q_MIL_HIDDEN") == "c"

    def test_q_mil_perm(self, submission):
        assert _require(submission, "Q_MIL_PERM") == "a"

    def test_q_mil_anypos(self, submission):
        assert _require(submission, "Q_MIL_ANYPOS") == "b"

    def test_q_mil_dilution(self, submission):
        assert _require(submission, "Q_MIL_DILUTION") == "a"

    def test_q_mil_attn(self, submission):
        assert _require(submission, "Q_MIL_ATTN") == "a"

    def test_q_mil_pool_params(self, submission):
        assert _require(submission, "Q_MIL_POOL_PARAMS") == "c"

    def test_q_mil_attn_behavior(self, submission):
        assert _require(submission, "Q_MIL_ATTN_BEHAVIOR") == "b"

    def test_q_mil_auroc_rank(self, submission):
        assert _require(submission, "Q_MIL_AUROC_RANK") == "c"

    def test_q_mil_auroc_calib(self, submission):
        assert _require(submission, "Q_MIL_AUROC_CALIB") == "b"

    def test_q_mil_proc_ship(self, submission):
        assert _require(submission, "Q_MIL_PROC_SHIP") == "b"

    def test_q_mil_proc_fpratio(self, submission):
        assert abs(_require(submission, "Q_MIL_PROC_FPRATIO") - 24) <= 2

    def test_q_mil_proc_why(self, submission):
        assert _require(submission, "Q_MIL_PROC_WHY") == "a"

    def test_q_mil_synth(self, submission):
        assert _require(submission, "Q_MIL_SYNTH") == "a"

    def test_q_mil_clam_pseudo(self, submission):
        assert _require(submission, "Q_MIL_CLAM_PSEUDO") == "b"

    def test_q_mil_pool_fingerprints(self, submission):
        # Panel A = soft+peaked (attention), B = uniform (mean), C = one-hot (max)
        assert _require(submission, "Q_MIL_POOL_FINGERPRINTS") == "c"


# Exercise 14 has no numeric grading: the canonical results table is rendered
# directly in the notebook markdown, and students answer three MCQs about it
# (Q_CLS_E14A / E14B / E14C, graded above in TestAnswersMC). No training run
# is required to grade Ex 14.
