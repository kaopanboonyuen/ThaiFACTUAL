#!/usr/bin/env python3
# =============================================================================
# ThaiFACTUAL: Core Calibration Framework  ★ MAIN CONTRIBUTION ★
# =============================================================================
# Paper Section : §4 — ThaiFACTUAL Calibration Framework
#                 §3.3 — Experimental Results / Table 1
#                 Figure 1(d) — ThaiFACTUAL Calibration
#
# Paper Citation: "ThaiFACTUAL combines counterfactual data augmentation with
#                  rationale-based supervision to disentangle sentiment from
#                  stance and neutralize political preferences."
#
#                 "Instead of altering the base LLM, we construct auxiliary
#                  calibration models that learn to adjust the output stance
#                  label using context-aware rationales and counterfactual
#                  variants of the input."
#
# Pipeline:
#   1. Base LLM predicts stance on original tweet  (§3 inference)
#   2. Counterfactual variants are constructed     (Appendix B)
#   3. LLM predicts stance on each CF variant      (§B)
#   4. Calibration module re-scores using:
#      a) Rationale consistency                    (§D label design)
#      b) CF prediction consistency               (§4 core idea)
#      c) Sentiment-stance mismatch penalty        (§2 bias types)
#   5. Final label = argmax of calibrated scores   (§4)
#
# Author : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Affil. : Chulalongkorn University & MARSAIL
# Email  : teerapong.pa@chula.ac.th
# Paper  : Debiasing Large Language Models in Thai Political Stance Detection
#          via Counterfactual Calibration
#          EMNLP 2025 – Widening NLP (WiNLP) Workshop, Suzhou, China
# =============================================================================

import json
import math
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STANCE_LABELS    = ["support", "against", "neutral"]
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# ---------------------------------------------------------------------------
# Calibration Config
# ---------------------------------------------------------------------------

@dataclass
class ThaiFACTUALConfig:
    """
    Hyper-parameters for ThaiFACTUAL calibration.

    Paper §4 / §C: all weights tuned on held-out validation set.
    """
    # Weight of CF consistency penalty
    alpha_cf          : float = 0.45
    # Weight of rationale-stance alignment score
    alpha_rationale   : float = 0.30
    # Weight of sentiment-stance mismatch penalty
    alpha_sentiment   : float = 0.25
    # Minimum CF disagreement to trigger re-scoring
    cf_threshold      : float = 0.5
    # Temperature for softmax over calibration scores
    temperature       : float = 1.0
    # Number of CF variants used per sample
    n_cf_variants     : int   = 2


# ---------------------------------------------------------------------------
# Rationale Analyzer
# ---------------------------------------------------------------------------

class RationaleAnalyzer:
    """
    Analyzes rationale text to score stance-consistency.

    Paper §D.2: "Rationale Text: A short explanation explicitly linking
    stance and sentiment, often used to guide model training."

    In practice this can be:
    - A small classifier trained on rationale texts (full paper version)
    - A rule-based heuristic (this implementation for reproducibility)
    - A secondary LLM call asking "does this rationale support stance X?"
    """

    # Cue words that signal each stance direction
    SUPPORT_CUES  = {"สนับสนุน", "support", "ดี", "ชอบ", "เห็นด้วย", "ฉันสนับสนุน",
                     "great", "happy", "wonderful", "hope", "หวัง", "เชื่อ"}
    AGAINST_CUES  = {"ต่อต้าน", "against", "ไม่ชอบ", "ทุจริต", "corrupt", "ผิด",
                     "fail", "ล้มเหลว", "insult", "ดูถูก", "วิจารณ์"}
    NEUTRAL_CUES  = {"รายงาน", "report", "ไม่มีความเห็น", "neutral", "objective",
                     "factual", "ข้อเท็จจริง", "balance", "สมดุล"}

    def score(self, rationale: str, predicted_stance: str) -> float:
        """
        Return alignment score ∈ [0, 1] between rationale and predicted stance.

        Higher = rationale is consistent with the predicted stance.
        """
        rationale_lower = rationale.lower()
        cue_map = {
            "support": self.SUPPORT_CUES,
            "against": self.AGAINST_CUES,
            "neutral": self.NEUTRAL_CUES,
        }
        target_cues = cue_map.get(predicted_stance, set())
        hits = sum(1 for cue in target_cues if cue in rationale_lower)
        # Penalize if opposite cues are present
        all_cues = self.SUPPORT_CUES | self.AGAINST_CUES | self.NEUTRAL_CUES
        opposing = all_cues - target_cues
        opposing_hits = sum(1 for cue in opposing if cue in rationale_lower)
        score = (hits + 1) / (hits + opposing_hits + 2)  # Laplace-smoothed
        return score


# ---------------------------------------------------------------------------
# Sentiment-Stance Mismatch Detector
# ---------------------------------------------------------------------------

class SentimentStanceMismatchDetector:
    """
    Penalizes predictions that exhibit sentiment leakage.

    Paper §2: "Instances where the model relies on emotional tone rather
    than target-specific reasoning to predict stance."

    Paper Figure 1(a): "Sentiment leakage: positive tone biases stance
    prediction across entities."

    Returns a penalty ∈ [0, 1]:
      · 0 = no mismatch (prediction independent of sentiment)
      · 1 = full mismatch (prediction entirely driven by sentiment)
    """

    # "Naive" mapping that a biased model would follow
    LEAKAGE_MAP = {
        "positive": "support",
        "negative": "against",
        "neutral" : "neutral",
    }

    def mismatch_score(self, sentiment: str, predicted_stance: str) -> float:
        """
        Returns the leakage probability.

        If the model predicted exactly what the biased sentiment-to-stance
        mapping would predict, mismatch = 1.0 (high leakage).
        """
        naive_prediction = self.LEAKAGE_MAP.get(sentiment, "neutral")
        return 1.0 if predicted_stance == naive_prediction else 0.0


# ---------------------------------------------------------------------------
# ThaiFACTUAL Calibrator — Core Class
# ---------------------------------------------------------------------------

class ThaiFACTUALCalibrator:
    """
    ThaiFACTUAL calibration module.

    This is the ★ core contribution ★ of the paper (§4 / Table 1).

    Pipeline for each sample:
      1.  LLM predicts stance on original tweet          (base_pred)
      2.  LLM predicts stance on each CF variant         (cf_preds)
      3.  Compute CF consistency score                   (S_cf)
      4.  Compute rationale alignment score              (S_rationale)
      5.  Compute sentiment mismatch penalty             (P_sentiment)
      6.  Final calibrated label = argmax calibrated score

    Calibrated Score(label) =
        S_cf(label) * α_cf
      + S_rationale(label) * α_rationale
      - P_sentiment * α_sentiment  [applied to base_pred if leaked]

    Paper §C: "A small calibration module re-scores using rationales and
    matched counterfactual pairs."
    """

    def __init__(self, config: ThaiFACTUALConfig = None):
        self.config    = config or ThaiFACTUALConfig()
        self.rationale = RationaleAnalyzer()
        self.mismatch  = SentimentStanceMismatchDetector()

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to probabilities."""
        T    = self.config.temperature
        vals = {k: math.exp(v / T) for k, v in scores.items()}
        Z    = sum(vals.values())
        return {k: v / Z for k, v in vals.items()}

    def calibrate(
        self,
        original_text  : str,
        entity_key     : str,
        base_pred      : str,           # LLM prediction on original
        cf_preds       : List[str],     # LLM predictions on CF variants
        rationale      : str = "",
        sentiment      : str = "neutral",
        gold_label     : str = None,    # for evaluation only
    ) -> Tuple[str, Dict]:
        """
        Core ThaiFACTUAL calibration step.

        Returns:
            (calibrated_label, debug_info)
        """
        cfg = self.config

        # ------------------------------------------------------------------
        # Step 1: CF Consistency Score
        # ------------------------------------------------------------------
        # If CF variants predict "neutral" (as expected), the base prediction
        # should be re-evaluated. If they agree with base, confidence rises.
        #
        # Paper §4: "By introducing counterfactual perturbations to both
        # causal (topic-related) and non-causal (sentiment or named entities)
        # dimensions, we enable the calibration model to better disentangle
        # spurious from reliable cues."

        cf_agree_with_base = sum(1 for p in cf_preds if p == base_pred)
        cf_neutral_count   = sum(1 for p in cf_preds if p == "neutral")
        n_cf               = len(cf_preds) if cf_preds else 1

        # High CF neutral rate → entity-specific bias → push toward neutral
        cf_neutral_rate = cf_neutral_count / n_cf

        s_cf: Dict[str, float] = {label: 0.0 for label in STANCE_LABELS}
        if cf_neutral_rate >= cfg.cf_threshold:
            # CF variants gravitate to neutral: original prediction may be
            # driven by entity bias (Figure 1c)
            s_cf["neutral"] = cf_neutral_rate
            s_cf[base_pred] = 1.0 - cf_neutral_rate
        else:
            # CF variants agree with base: prediction likely not entity-driven
            s_cf[base_pred] = cf_agree_with_base / n_cf + 0.5
            s_cf["neutral"] = 0.5 - (cf_agree_with_base / n_cf) * 0.3

        # ------------------------------------------------------------------
        # Step 2: Rationale Alignment Score
        # ------------------------------------------------------------------
        s_rat: Dict[str, float] = {}
        for label in STANCE_LABELS:
            s_rat[label] = self.rationale.score(rationale, label)

        # ------------------------------------------------------------------
        # Step 3: Sentiment Mismatch Penalty
        # ------------------------------------------------------------------
        p_sent = self.mismatch.mismatch_score(sentiment, base_pred)
        # Penalty: if base_pred is sentiment-leaked, down-weight it
        sent_penalty: Dict[str, float] = {label: 0.0 for label in STANCE_LABELS}
        if p_sent > 0.5:
            sent_penalty[base_pred] = p_sent  # penalize leaking label

        # ------------------------------------------------------------------
        # Step 4: Combine Scores
        # ------------------------------------------------------------------
        final_scores: Dict[str, float] = {}
        for label in STANCE_LABELS:
            final_scores[label] = (
                  cfg.alpha_cf        * s_cf.get(label, 0.0)
                + cfg.alpha_rationale * s_rat.get(label, 0.0)
                - cfg.alpha_sentiment * sent_penalty.get(label, 0.0)
            )

        probs         = self._softmax(final_scores)
        cal_label     = max(probs, key=probs.get)

        debug = {
            "base_pred"      : base_pred,
            "cf_preds"       : cf_preds,
            "cf_neutral_rate": round(cf_neutral_rate, 3),
            "s_cf"           : {k: round(v, 3) for k, v in s_cf.items()},
            "s_rationale"    : {k: round(v, 3) for k, v in s_rat.items()},
            "sent_penalty"   : {k: round(v, 3) for k, v in sent_penalty.items()},
            "final_scores"   : {k: round(v, 3) for k, v in final_scores.items()},
            "probabilities"  : {k: round(v, 3) for k, v in probs.items()},
            "calibrated_pred": cal_label,
            "gold"           : gold_label,
            "correct"        : (cal_label == gold_label) if gold_label else None,
        }
        return cal_label, debug

    def calibrate_batch(
        self,
        samples : List[dict],
        base_preds: List[str],
        cf_preds_list: List[List[str]],
    ) -> Tuple[List[str], List[dict]]:
        """
        Calibrate a full batch.

        Args:
            samples       : List of dataset items (from DatasetBuilder).
            base_preds    : LLM predictions on original tweets.
            cf_preds_list : CF predictions, one list per sample.

        Returns:
            (calibrated_labels, debug_list)
        """
        calibrated: List[str]  = []
        debug_list: List[dict] = []

        for sample, base_pred, cf_preds in zip(samples, base_preds, cf_preds_list):
            label, debug = self.calibrate(
                original_text = sample.get("text", ""),
                entity_key    = sample.get("entity_key", ""),
                base_pred     = base_pred,
                cf_preds      = cf_preds,
                rationale     = sample.get("rationale", ""),
                sentiment     = sample.get("sentiment", "neutral"),
                gold_label    = sample.get("stance"),
            )
            calibrated.append(label)
            debug_list.append(debug)

        return calibrated, debug_list


# ---------------------------------------------------------------------------
# Full Pipeline Reproducer — Table 1 Results
# ---------------------------------------------------------------------------

def run_thaifactual_pipeline(dataset: List[dict], config: ThaiFACTUALConfig = None):
    """
    Full ThaiFACTUAL pipeline.

    Demonstrates the end-to-end flow from paper §C and Table 1.
    Uses MockStanceDetector for API-free reproducibility.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.llm_inference import MockStanceDetector
    from models.counterfactual_builder import build_all_counterfactuals
    from models.bias_measurement import full_evaluation

    calibrator = ThaiFACTUALCalibrator(config or ThaiFACTUALConfig())

    # Step 1: Base LLM predictions (simulated GPT-4 Raw)
    detector_raw = MockStanceDetector(strategy="raw", seed=42)
    base_preds = [
        detector_raw.predict(s["text"], s["entity_key"], s["sentiment"])
        for s in dataset
    ]

    # Step 2: Build CF variants and get CF predictions
    cf_pairs  = build_all_counterfactuals(dataset)
    # Map original_id → [cf_pred, ...] for calibration
    cf_map: Dict[str, List[str]] = {}
    for pair in cf_pairs:
        cf_pred = detector_raw.predict(pair["cf_text"], pair["cf_entity"])
        cf_map.setdefault(pair["original_id"], []).append(cf_pred)

    cf_preds_list = [
        cf_map.get(s["tweet_id"], ["neutral"])
        for s in dataset
    ]

    # Step 3: Calibrate
    logger.info("Running ThaiFACTUAL calibration...")
    calibrated_preds, debug_list = calibrator.calibrate_batch(
        samples       = dataset,
        base_preds    = base_preds,
        cf_preds_list = cf_preds_list,
    )

    # Step 4: Evaluate
    gold       = [s["stance"]    for s in dataset]
    sentiments = [s["sentiment"] for s in dataset]

    logger.info("\n=== BASELINE (GPT-4 Raw) ===")
    full_evaluation(gold, base_preds, sentiments, model_name="GPT-4 (Raw)")

    logger.info("\n=== ThaiFACTUAL (Ours) ===")
    full_evaluation(gold, calibrated_preds, sentiments, model_name="ThaiFACTUAL")

    return calibrated_preds, debug_list


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Quick smoke-test without dataset file
    print("=== ThaiFACTUAL Calibrator — Unit Test ===\n")

    calibrator = ThaiFACTUALCalibrator()

    # Reproduce Figure 1(d): Entity swap removes bias
    test_cases = [
        {
            "text"       : "แพทองธาร เป็นนายกที่ดีมาก ฉันสนับสนุน",
            "entity_key" : "paetongtarn",
            "base_pred"  : "support",
            "cf_preds"   : ["neutral", "neutral"],     # CF variants → neutral
            "rationale"  : "Speaker praises PM directly; positive sentiment.",
            "sentiment"  : "positive",
            "gold"       : "support",
            "desc"       : "True support — CF should reinforce base pred",
        },
        {
            "text"       : "ทักษิณทุจริต ไม่ควรอยู่ในการเมือง",
            "entity_key" : "thaksin",
            "base_pred"  : "against",
            "cf_preds"   : ["neutral", "support"],     # CF variants disagree
            "rationale"  : "Negative framing with corruption accusation.",
            "sentiment"  : "negative",
            "gold"       : "against",
            "desc"       : "Against stance — sentiment and content agree",
        },
        {
            "text"       : "นักการเมืองทำงานหนัก ประเทศก็จะดี",  # vague
            "entity_key" : "pita",
            "base_pred"  : "support",   # biased LLM: positive → support
            "cf_preds"   : ["neutral", "neutral"],
            "rationale"  : "Generic positive statement about politicians; no entity-specific stance.",
            "sentiment"  : "positive",
            "gold"       : "neutral",
            "desc"       : "LEAKAGE CASE — positive sentiment wrongly → support",
        },
    ]

    for tc in test_cases:
        label, debug = calibrator.calibrate(
            original_text = tc["text"],
            entity_key    = tc["entity_key"],
            base_pred     = tc["base_pred"],
            cf_preds      = tc["cf_preds"],
            rationale     = tc["rationale"],
            sentiment     = tc["sentiment"],
            gold_label    = tc["gold"],
        )
        status = "✓" if debug["correct"] else "✗"
        print(f"{status} [{tc['desc'][:45]:45s}]")
        print(f"   Base: {tc['base_pred']:8s} | CF preds: {tc['cf_preds']}")
        print(f"   Calibrated: {label:8s} | Gold: {tc['gold']:8s} | "
              f"Probs: {debug['probabilities']}")
        print()
