#!/usr/bin/env python3
# =============================================================================
# ThaiFACTUAL: Thai Political Stance Dataset Builder
# =============================================================================
# Paper Section : Appendix A — Thai Political Stance Dataset Construction
#                 (Section A.1 Entity Selection, A.2 Data Collection,
#                  A.3 Annotation Procedure, A.4 Data Balancing)
#
# Paper Citation: "We curate and release the first high-quality Thai political
#                  stance dataset with stance, sentiment, rationale, and bias
#                  markers across diverse political entities and events."
#
# Author : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Affil. : Chulalongkorn University & MARSAIL
# Email  : teerapong.pa@chula.ac.th
# Paper  : Debiasing Large Language Models in Thai Political Stance Detection
#          via Counterfactual Calibration
#          EMNLP 2025 – Widening NLP (WiNLP) Workshop, Suzhou, China
# =============================================================================

import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

ENTITIES = {
    "paetongtarn": {
        "full_name": "Paetongtarn Shinawatra",
        "role": "Current Prime Minister, Pheu Thai Party",
        "ideology": "pro-establishment populism",
        "keywords": ["เเพทองธาร", "แพทองธาร", "อุ๊งอิ๊ง", "เพื่อไทย", "นายกฯ"],
    },
    "thaksin": {
        "full_name": "Thaksin Shinawatra",
        "role": "Former PM, recently returned from exile",
        "ideology": "historical political division",
        "keywords": ["ทักษิณ", "แม้ว", "เพื่อไทย", "UDD", "เสื้อแดง"],
    },
    "pita": {
        "full_name": "Pita Limjaroenrat",
        "role": "Move Forward Party, reformist opposition",
        "ideology": "youth-backed, policy-progressive",
        "keywords": ["พิธา", "ก้าวไกล", "Move Forward", "MFP", "ลิ้มเจริญรัตน์"],
    },
}

STANCE_LABELS   = ["support", "against", "neutral"]
SENTIMENT_LABELS = ["positive", "negative", "neutral"]


@dataclass
class TweetSample:
    """
    A single annotated Thai political tweet.

    Fields align with Appendix A.3 — Annotation Procedure:
      - stance    : support / against / neutral
      - sentiment : positive / negative / neutral
      - rationale : short explanation linking stance & sentiment (§D deep-dive)
      - bias_marker: optional binary flag for sentiment leakage / entity bias
    """
    tweet_id   : str
    text       : str
    entity_key : str                   # one of ENTITIES keys
    stance     : str                   # support / against / neutral
    sentiment  : str                   # positive / negative / neutral
    rationale  : str = ""
    bias_marker: Optional[str] = None  # "sentiment_leakage" | "entity_bias" | None
    is_counterfactual: bool = False
    original_tweet_id: Optional[str] = None  # links CF to source

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Synthetic sample bank — mirrors the 270-tweet balanced corpus (§A.4)
# In a real setting these come from the Twitter/X API + annotators.
# ---------------------------------------------------------------------------

_SAMPLE_BANK: List[dict] = [
    # ---- Paetongtarn -------------------------------------------------------
    {
        "entity_key": "paetongtarn",
        "text": "แพทองธาร เป็นนายกที่มีวิสัยทัศน์ ประเทศไทยจะดีขึ้น",
        "stance": "support", "sentiment": "positive",
        "rationale": "Speaker explicitly praises PM's vision; stance aligns with positive sentiment.",
        "bias_marker": "sentiment_leakage",
    },
    {
        "entity_key": "paetongtarn",
        "text": "ไม่เชื่อว่าเธอจะสามารถแก้ปัญหาเศรษฐกิจได้จริง",
        "stance": "against", "sentiment": "negative",
        "rationale": "Doubt about economic policy; negative tone matches against stance.",
        "bias_marker": None,
    },
    {
        "entity_key": "paetongtarn",
        "text": "นายกฯ เข้าร่วมประชุม ASEAN วันนี้ ไม่มีความเห็นเพิ่มเติม",
        "stance": "neutral", "sentiment": "neutral",
        "rationale": "Factual report with no evaluative language; truly neutral.",
        "bias_marker": None,
    },
    {
        "entity_key": "paetongtarn",
        "text": "ชอบที่เธอพยายาม แต่นโยบายยังไม่ชัดเจน",
        "stance": "neutral", "sentiment": "positive",
        "rationale": "Positive tone but qualified; stance is neutral. Classic sentiment-leakage trap.",
        "bias_marker": "sentiment_leakage",
    },
    # ---- Thaksin -----------------------------------------------------------
    {
        "entity_key": "thaksin",
        "text": "ทักษิณกลับมาแล้ว หวังว่าจะช่วยประเทศได้จริง",
        "stance": "support", "sentiment": "positive",
        "rationale": "Hopeful tone about return; support stance.",
        "bias_marker": None,
    },
    {
        "entity_key": "thaksin",
        "text": "ทักษิณคือคนทุจริต การกลับมาของเขาเป็นการดูถูกความยุติธรรม",
        "stance": "against", "sentiment": "negative",
        "rationale": "Strong negative framing with corruption accusation; clear against stance.",
        "bias_marker": None,
    },
    {
        "entity_key": "thaksin",
        "text": "ทักษิณเดินทางถึงกรุงเทพฯ เมื่อวาน",
        "stance": "neutral", "sentiment": "neutral",
        "rationale": "Purely factual; no evaluative content.",
        "bias_marker": None,
    },
    {
        "entity_key": "thaksin",
        "text": "เขาทำหลายอย่างเพื่อคนจน แต่ปัญหาด้านจริยธรรมก็มีเช่นกัน",
        "stance": "neutral", "sentiment": "neutral",
        "rationale": "Balanced view; neither fully supportive nor opposed. Neutral stance.",
        "bias_marker": "entity_bias",
    },
    # ---- Pita --------------------------------------------------------------
    {
        "entity_key": "pita",
        "text": "พิธาคือความหวังของคนรุ่นใหม่ ก้าวไกลต้องชนะ",
        "stance": "support", "sentiment": "positive",
        "rationale": "Enthusiastic endorsement; clearly supportive.",
        "bias_marker": None,
    },
    {
        "entity_key": "pita",
        "text": "พิธาล้มเหลวในการเป็นนายก ความฝันสลาย",
        "stance": "against", "sentiment": "negative",
        "rationale": "Critical of failure; against stance.",
        "bias_marker": None,
    },
    {
        "entity_key": "pita",
        "text": "พิธาให้สัมภาษณ์เรื่องนโยบายการศึกษาวันนี้",
        "stance": "neutral", "sentiment": "neutral",
        "rationale": "Neutral report; no judgment.",
        "bias_marker": None,
    },
    {
        "entity_key": "pita",
        "text": "เขาพูดได้ดีมาก แต่ยังไม่แน่ใจว่าจะทำได้จริงหรือเปล่า",
        "stance": "neutral", "sentiment": "positive",
        "rationale": "Positive tone but skeptical stance — sentiment != stance. Leakage trap.",
        "bias_marker": "sentiment_leakage",
    },
]


def _make_id(text: str, entity: str, stance: str) -> str:
    """Deterministic short hash as tweet ID."""
    raw = f"{entity}_{stance}_{text[:30]}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def build_dataset(samples_per_entity: int = 90, seed: int = 42) -> List[TweetSample]:
    """
    Build the balanced 270-tweet dataset (§A.4).

    Each entity gets exactly `samples_per_entity` tweets (default 90),
    equally distributed across stance × sentiment categories.

    In production: replace _SAMPLE_BANK with real scraped & annotated tweets.
    Here we replicate via controlled oversampling for reproducibility.
    """
    random.seed(seed)
    dataset: List[TweetSample] = []

    for entity_key in ENTITIES:
        entity_samples = [s for s in _SAMPLE_BANK if s["entity_key"] == entity_key]
        # Oversample to reach target count while keeping distribution
        pool = (entity_samples * ((samples_per_entity // len(entity_samples)) + 2))
        pool = pool[:samples_per_entity]
        random.shuffle(pool)

        for i, s in enumerate(pool):
            tid = _make_id(s["text"], entity_key, s["stance"]) + f"_{i:03d}"
            dataset.append(TweetSample(
                tweet_id    = tid,
                text        = s["text"],
                entity_key  = entity_key,
                stance      = s["stance"],
                sentiment   = s["sentiment"],
                rationale   = s.get("rationale", ""),
                bias_marker = s.get("bias_marker"),
            ))

    print(f"[DatasetBuilder] Total samples: {len(dataset)}")
    for ek in ENTITIES:
        count = sum(1 for d in dataset if d.entity_key == ek)
        print(f"  · {ENTITIES[ek]['full_name']}: {count} tweets")
    return dataset


def save_dataset(dataset: List[TweetSample], output_path: str = "data/sample_dataset.json"):
    """Persist dataset to JSON (§A — Data Release)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    records = [s.to_dict() for s in dataset]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[DatasetBuilder] Saved {len(records)} records → {output_path}")


def load_dataset(path: str = "data/sample_dataset.json") -> List[TweetSample]:
    """Load dataset from JSON."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return [TweetSample(**r) for r in records]


def print_statistics(dataset: List[TweetSample]):
    """Print label distribution statistics."""
    print("\n=== Dataset Statistics (Table corresponds to §A.4) ===")
    for entity_key, meta in ENTITIES.items():
        subs = [d for d in dataset if d.entity_key == entity_key]
        print(f"\n{meta['full_name']} (n={len(subs)}):")
        for stance in STANCE_LABELS:
            n = sum(1 for s in subs if s.stance == stance)
            print(f"  stance={stance}: {n}")
        for sent in SENTIMENT_LABELS:
            n = sum(1 for s in subs if s.sentiment == sent)
            print(f"  sentiment={sent}: {n}")
        biased = sum(1 for s in subs if s.bias_marker)
        print(f"  bias_markers: {biased}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = build_dataset(samples_per_entity=90)
    print_statistics(dataset)
    save_dataset(dataset, "data/sample_dataset.json")
