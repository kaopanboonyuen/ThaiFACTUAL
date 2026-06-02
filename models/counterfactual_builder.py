#!/usr/bin/env python3
# =============================================================================
# ThaiFACTUAL: Counterfactual Construction Process
# =============================================================================
# Paper Section : Appendix B — Counterfactual Construction Process
#                 §4 — ThaiFACTUAL Framework (plug-and-play debiasing)
#
# Paper Citation: "We generate counterfactual variants by replacing political
#                  entities while preserving sentiment structure and tone.
#                  This substitution forces the model to focus on the political
#                  target rather than reusing learned sentiment-to-stance
#                  correlations."
#
# Paper Examples (§B):
#   Original : "Pita did a great job. I'm happy to see his vision for Thailand."
#   CF Variant: "Thaksin did a great job. I'm happy to see his vision for Thailand."
#
#   Original : "Thaksin is corrupt. His return is an insult to justice."
#   CF Variant: "Paetongtarn is corrupt. Her rise is an insult to justice."
#
# Author : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Affil. : Chulalongkorn University & MARSAIL
# Email  : teerapong.pa@chula.ac.th
# Paper  : Debiasing Large Language Models in Thai Political Stance Detection
#          via Counterfactual Calibration
#          EMNLP 2025 – Widening NLP (WiNLP) Workshop, Suzhou, China
# =============================================================================

import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Entity Lexicon — Name × Pronoun × Keywords (Thai + English)
# ---------------------------------------------------------------------------

ENTITY_LEXICON: Dict[str, Dict] = {
    "paetongtarn": {
        "full_name_en"  : "Paetongtarn Shinawatra",
        "full_name_th"  : "แพทองธาร ชินวัตร",
        "nicknames_th"  : ["อุ๊งอิ๊ง", "แพทองธาร"],
        "pronoun_th"    : "เธอ",   # she/her in Thai
        "pronoun_en"    : "her",
        "party"         : "เพื่อไทย",
        "role_th"       : "นายกฯ หญิง",
    },
    "thaksin": {
        "full_name_en"  : "Thaksin Shinawatra",
        "full_name_th"  : "ทักษิณ ชินวัตร",
        "nicknames_th"  : ["ทักษิณ", "แม้ว"],
        "pronoun_th"    : "เขา",
        "pronoun_en"    : "his",
        "party"         : "เพื่อไทย",
        "role_th"       : "อดีตนายกฯ",
    },
    "pita": {
        "full_name_en"  : "Pita Limjaroenrat",
        "full_name_th"  : "พิธา ลิ้มเจริญรัตน์",
        "nicknames_th"  : ["พิธา"],
        "pronoun_th"    : "เขา",
        "pronoun_en"    : "his",
        "party"         : "ก้าวไกล",
        "role_th"       : "หัวหน้าพรรคก้าวไกล",
    },
}


@dataclass
class CounterfactualPair:
    """
    A (original, counterfactual) pair used in ThaiFACTUAL calibration.

    Paper §B: "We maintain lexical polarity (e.g., 'corrupt', 'insult')
    while altering the referenced entity."
    """
    original_id       : str
    original_text     : str
    original_entity   : str
    original_stance   : str   # gold label of original
    cf_text           : str
    cf_entity         : str
    expected_cf_label : str   # "neutral" if entity swap removes stance signal
    sentiment_preserved: bool = True


def _build_entity_patterns(entity_key: str) -> List[str]:
    """Return all Thai/English surface forms for an entity."""
    meta = ENTITY_LEXICON[entity_key]
    patterns = [
        meta["full_name_en"],
        meta["full_name_th"],
    ] + meta["nicknames_th"]
    return patterns


def swap_entity(text: str, source_key: str, target_key: str) -> Tuple[str, bool]:
    """
    Replace all surface forms of `source_key` entity with `target_key` entity.

    Preserves surrounding text (sentiment, grammar) intact.

    Returns:
        (modified_text, was_modified)
    """
    src_patterns = _build_entity_patterns(source_key)
    tgt_meta     = ENTITY_LEXICON[target_key]
    tgt_name_th  = tgt_meta["full_name_th"]

    modified = text
    was_modified = False

    for pattern in sorted(src_patterns, key=len, reverse=True):  # longest first
        if pattern in modified:
            modified     = modified.replace(pattern, tgt_name_th)
            was_modified = True

    # Pronoun swap (Thai)
    src_pronoun = ENTITY_LEXICON[source_key]["pronoun_th"]
    tgt_pronoun = ENTITY_LEXICON[target_key]["pronoun_th"]
    if src_pronoun != tgt_pronoun and src_pronoun in modified:
        modified = modified.replace(src_pronoun, tgt_pronoun)

    return modified, was_modified


def build_counterfactual_pairs(
    tweet_id    : str,
    text        : str,
    entity_key  : str,
    stance      : str,
    sentiment   : str,
    cf_targets  : Optional[List[str]] = None,
) -> List[CounterfactualPair]:
    """
    Generate counterfactual variants for a single tweet.

    For each target entity (excluding the source), produce a CF pair.

    Paper §B: "By constructing counterfactual inputs—swapping political
    entities while preserving sentiment—and conditioning predictions on
    neutral rationales, ThaiFACTUAL forces the model to disentangle causal
    stance features from confounding sentiment or entity signals."

    Args:
        tweet_id   : Source tweet ID.
        text       : Original tweet text.
        entity_key : Source entity (political figure in original tweet).
        stance     : Gold stance of the original tweet.
        sentiment  : Gold sentiment of the original tweet.
        cf_targets : List of target entity keys to swap to (default: all others).

    Returns:
        List of CounterfactualPair objects.
    """
    if cf_targets is None:
        cf_targets = [k for k in ENTITY_LEXICON if k != entity_key]

    pairs: List[CounterfactualPair] = []
    for target_key in cf_targets:
        cf_text, modified = swap_entity(text, entity_key, target_key)
        if not modified:
            # Entity name not found in text — skip this CF
            continue

        # Expected CF label: "neutral" unless entity swap is semantically trivial
        # Paper §B (Example): support→neutral for entity swap when stance was
        # driven by entity preference, not content.
        expected_label = "neutral" if stance in ("support", "against") else "neutral"

        pairs.append(CounterfactualPair(
            original_id        = tweet_id,
            original_text      = text,
            original_entity    = entity_key,
            original_stance    = stance,
            cf_text            = cf_text,
            cf_entity          = target_key,
            expected_cf_label  = expected_label,
            sentiment_preserved= True,  # by design: only entity is replaced
        ))

    return pairs


def build_all_counterfactuals(dataset: List[dict]) -> List[CounterfactualPair]:
    """
    Build the full counterfactual augmentation set from the dataset.

    Paper §C: "Counterfactual data was injected as an auxiliary correction
    layer—LLMs predict, then a small calibration module re-scores using
    rationales and matched counterfactual pairs."
    """
    all_pairs: List[CounterfactualPair] = []
    for item in dataset:
        pairs = build_counterfactual_pairs(
            tweet_id   = item["tweet_id"],
            text       = item["text"],
            entity_key = item["entity_key"],
            stance     = item["stance"],
            sentiment  = item["sentiment"],
        )
        all_pairs.extend(pairs)

    print(f"[CounterfactualBuilder] Generated {len(all_pairs)} CF pairs "
          f"from {len(dataset)} original tweets.")
    return all_pairs


def save_counterfactuals(pairs: List[CounterfactualPair], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pairs], f, ensure_ascii=False, indent=2)
    print(f"[CounterfactualBuilder] Saved {len(pairs)} CF pairs → {path}")


# ---------------------------------------------------------------------------
# Entry Point — reproduce Appendix B examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Counterfactual Construction — Appendix B Examples ===\n")

    # Example 1 (§B): Support → Neutral Shift
    ex1 = CounterfactualPair(
        original_id       = "ex_001",
        original_text     = "Pita did a great job. I'm happy to see his vision for Thailand.",
        original_entity   = "pita",
        original_stance   = "support",
        cf_text           = "Thaksin did a great job. I'm happy to see his vision for Thailand.",
        cf_entity         = "thaksin",
        expected_cf_label = "neutral",
        sentiment_preserved = True,
    )

    # Example 2 (§B): Against + Negative
    ex2 = CounterfactualPair(
        original_id       = "ex_002",
        original_text     = "Thaksin is corrupt. His return is an insult to justice.",
        original_entity   = "thaksin",
        original_stance   = "against",
        cf_text           = "Paetongtarn is corrupt. Her rise is an insult to justice.",
        cf_entity         = "paetongtarn",
        expected_cf_label = "neutral",
        sentiment_preserved = True,
    )

    for ex in [ex1, ex2]:
        print(f"Original  [{ex.original_entity:15s}]: {ex.original_text}")
        print(f"CF Variant [{ex.cf_entity:15s}]: {ex.cf_text}")
        print(f"Orig Stance: {ex.original_stance} | Expected CF: {ex.expected_cf_label}")
        print(f"Sentiment preserved: {ex.sentiment_preserved}")
        print("-" * 70)

    # Test Thai entity swap
    print("\n--- Thai Entity Swap Test ---")
    thai_tweets = [
        ("แพทองธาร เป็นนายกที่ดี เธอทำได้ดีมาก",    "paetongtarn", "pita"),
        ("ทักษิณกลับมาแล้ว เขาจะเปลี่ยนแปลงอะไร",   "thaksin",     "pita"),
        ("พิธาพูดถึงปัญหาการศึกษา เขาเข้าใจคนหนุ่ม", "pita",        "thaksin"),
    ]
    for text, src, tgt in thai_tweets:
        swapped, ok = swap_entity(text, src, tgt)
        status = "✓ swapped" if ok else "✗ no match"
        print(f"{status} | {src:15s}→{tgt:15s} | '{swapped}'")
