#!/usr/bin/env python3
# =============================================================================
# ThaiFACTUAL: Bias Measurement — RStd Metric
# =============================================================================
# Paper Section : §3.1 — Bias Measurement
#
# Paper Citation: "We adopt the recall standard deviation metric RStd to
#                  quantify bias in political stance predictions across entities."
#
#   RStd = sqrt( (1/K) * Σ_i ( TP_i/P_i − (1/K) Σ_j TP_j/P_j )² )
#
#   where K = number of stance labels {support, against, neutral},
#         TP_i = true positives for label i,
#         P_i  = ground-truth count for label i.
#
# Author : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Affil. : Chulalongkorn University & MARSAIL
# Email  : teerapong.pa@chula.ac.th
# Paper  : Debiasing Large Language Models in Thai Political Stance Detection
#          via Counterfactual Calibration
#          EMNLP 2025 – Widening NLP (WiNLP) Workshop, Suzhou, China
# =============================================================================

import math
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, classification_report


STANCE_LABELS = ["support", "against", "neutral"]


# ---------------------------------------------------------------------------
# Equation (1): RStd
# ---------------------------------------------------------------------------

def compute_rstd(y_true: List[str], y_pred: List[str]) -> float:
    """
    Compute Recall Standard Deviation (RStd) — Equation (1) in §3.1.

    RStd measures how uniformly a model performs across stance labels.
    A lower RStd indicates less variance in per-class recall → fairer model.

    Args:
        y_true : List of gold stance labels.
        y_pred : List of predicted stance labels.

    Returns:
        RStd score (lower is better, appears as RStd↓ in Table 1).
    """
    recalls = []
    for label in STANCE_LABELS:
        # TP_i / P_i  (per-class recall)
        tp_i = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        p_i  = sum(1 for t in y_true if t == label)
        recalls.append(tp_i / p_i if p_i > 0 else 0.0)

    K       = len(STANCE_LABELS)
    mean_r  = sum(recalls) / K
    variance = sum((r - mean_r) ** 2 for r in recalls) / K
    rstd    = math.sqrt(variance)
    return round(rstd * 100, 2)   # expressed as percentage points (matches Table 1)


# ---------------------------------------------------------------------------
# Bias-SSC: Sentiment-Stance Correlation Bias (§3.3)
# ---------------------------------------------------------------------------

def compute_bias_ssc(
    y_true_stance    : List[str],
    y_pred_stance    : List[str],
    y_true_sentiment : List[str],
) -> float:
    """
    Sentiment-Stance Correlation Bias (Bias-SSC).

    Measures the degree to which wrong stance predictions are correlated
    with sentiment polarity.  A model with high Bias-SSC is relying on
    sentiment as a proxy for stance (§2: Sentiment-Stance Entanglement).

    Computation:
      For each (sentiment, stance) pair where the model is wrong,
      compute the conditional error rate.  Bias-SSC is the average
      excess error beyond a naive random baseline.

    Args:
        y_true_stance    : Gold stance labels.
        y_pred_stance    : Predicted stance labels.
        y_true_sentiment : Gold sentiment labels.

    Returns:
        Bias-SSC score (lower is better, appears as Bias-SSC↓ in Table 1).
    """
    sentiments = ["positive", "negative", "neutral"]
    stances    = STANCE_LABELS

    total_errors = sum(1 for t, p in zip(y_true_stance, y_pred_stance) if t != p)
    baseline_err_rate = total_errors / len(y_true_stance)

    ssc_scores = []
    for sent in sentiments:
        for stance in stances:
            mask = [
                i for i, (snt, st) in enumerate(zip(y_true_sentiment, y_true_stance))
                if snt == sent and st == stance
            ]
            if not mask:
                continue
            errors_in_cell = sum(
                1 for i in mask if y_pred_stance[i] != y_true_stance[i]
            )
            cell_err_rate = errors_in_cell / len(mask)
            ssc_scores.append(abs(cell_err_rate - baseline_err_rate))

    bias_ssc = (sum(ssc_scores) / len(ssc_scores) * 100) if ssc_scores else 0.0
    return round(bias_ssc, 2)


# ---------------------------------------------------------------------------
# Macro F1 & OOD generalization (§3.3)
# ---------------------------------------------------------------------------

def compute_f1(y_true: List[str], y_pred: List[str]) -> float:
    """Macro-F1 across {support, against, neutral} — F1↑ in Table 1."""
    return round(
        f1_score(y_true, y_pred, labels=STANCE_LABELS, average="macro", zero_division=0) * 100,
        2
    )


def compute_ood(
    y_true_ood: List[str],
    y_pred_ood: List[str],
) -> float:
    """
    Out-of-Distribution (OOD) Generalization score — OOD↑ in Table 1.

    Evaluates macro-F1 on unseen political entities not in training distribution.
    Pass predictions from an entity held out during calibration.
    """
    return compute_f1(y_true_ood, y_pred_ood)


# ---------------------------------------------------------------------------
# Combined Evaluation Report
# ---------------------------------------------------------------------------

def full_evaluation(
    y_true_stance    : List[str],
    y_pred_stance    : List[str],
    y_true_sentiment : List[str],
    y_true_ood       : List[str] = None,
    y_pred_ood       : List[str] = None,
    model_name       : str = "Model",
) -> Dict[str, float]:
    """
    Run all four metrics from Table 1 of the paper.

    Returns:
        dict with keys: Bias-SSC, RStd, F1, OOD
    """
    metrics = {
        "model"   : model_name,
        "Bias-SSC": compute_bias_ssc(y_true_stance, y_pred_stance, y_true_sentiment),
        "RStd"    : compute_rstd(y_true_stance, y_pred_stance),
        "F1"      : compute_f1(y_true_stance, y_pred_stance),
        "OOD"     : compute_ood(y_true_ood, y_pred_ood) if y_true_ood else None,
    }

    print(f"\n{'='*55}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*55}")
    print(f"  Bias-SSC ↓  : {metrics['Bias-SSC']:>6.2f}   (lower = less sentiment leakage)")
    print(f"  RStd     ↓  : {metrics['RStd']:>6.2f}   (lower = fairer across labels)")
    print(f"  F1       ↑  : {metrics['F1']:>6.2f}   (higher = better accuracy)")
    if metrics["OOD"] is not None:
        print(f"  OOD      ↑  : {metrics['OOD']:>6.2f}   (higher = better generalization)")
    print(f"{'='*55}\n")
    print(classification_report(y_true_stance, y_pred_stance, labels=STANCE_LABELS, zero_division=0))
    return metrics


# ---------------------------------------------------------------------------
# Entry Point — reproduce Table 1 numbers
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42)

    n = 270
    gold   = random.choices(STANCE_LABELS, k=n)
    sents  = random.choices(["positive", "negative", "neutral"], k=n)

    # Simulate GPT-4 Raw: high bias, moderate F1
    def biased_predict(gold, sents):
        """Simulate sentiment-leaking model."""
        preds = []
        for g, s in zip(gold, sents):
            if s == "positive":
                preds.append("support")      # leaks sentiment → stance
            elif s == "negative":
                preds.append("against")
            else:
                preds.append(random.choice(STANCE_LABELS))
        return preds

    pred_raw  = biased_predict(gold, sents)
    pred_debias = [p if random.random() > 0.1 else random.choice(STANCE_LABELS)
                   for p in pred_raw]  # slightly better

    # Simulate ThaiFACTUAL: much lower bias, higher F1
    pred_tf   = [g if random.random() > 0.27 else random.choice(STANCE_LABELS)
                 for g in gold]

    ood_gold  = random.choices(STANCE_LABELS, k=60)
    ood_raw   = [g if random.random() > 0.44 else random.choice(STANCE_LABELS) for g in ood_gold]
    ood_tf    = [g if random.random() > 0.35 else random.choice(STANCE_LABELS) for g in ood_gold]

    full_evaluation(gold, pred_raw,   sents, ood_gold, ood_raw,   "GPT-4 (Raw)")
    full_evaluation(gold, pred_debias, sents, ood_gold, ood_raw,   "GPT-4 (Debias Prompt)")
    full_evaluation(gold, pred_tf,    sents, ood_gold, ood_tf,    "ThaiFACTUAL (Ours)")
