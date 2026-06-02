#!/usr/bin/env python3
# =============================================================================
# ThaiFACTUAL: Thai Text Preprocessing Utilities
# =============================================================================
# Paper Section : Appendix D.4 — Why Thai Political Language Challenges LLMs
#
# Paper Citation: "Several linguistic and cultural factors make Thai political
#                  stance detection particularly challenging:
#                  - Indirect Expression: sarcasm, irony, metaphor, rhetorical
#                    understatement
#                  - Entity Sensitivity: identical structures may imply different
#                    stances depending on the political figure
#                  - Emotionally Encoded Stance: implicit stance signaling
#                    embedded in emotional or moral appeals"
#
# Author : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Affil. : Chulalongkorn University & MARSAIL
# Email  : teerapong.pa@chula.ac.th
# Paper  : Debiasing Large Language Models in Thai Political Stance Detection
#          via Counterfactual Calibration
#          EMNLP 2025 – Widening NLP (WiNLP) Workshop, Suzhou, China
# =============================================================================

import re
import unicodedata
from typing import List, Optional


# ---------------------------------------------------------------------------
# Thai Unicode Ranges
# ---------------------------------------------------------------------------

THAI_RANGE = (0x0E00, 0x0E7F)


def is_thai_char(char: str) -> bool:
    """Check if a character is in the Thai Unicode block."""
    return THAI_RANGE[0] <= ord(char) <= THAI_RANGE[1]


def thai_ratio(text: str) -> float:
    """Return the fraction of Thai characters in the text."""
    if not text:
        return 0.0
    thai_count = sum(1 for c in text if is_thai_char(c))
    return thai_count / len(text)


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

def normalize_thai(text: str) -> str:
    """
    Normalize Thai political tweet text.

    Steps:
      1. Unicode NFC normalization
      2. Remove zero-width characters (common in Thai social media)
      3. Collapse repeated characters (e.g., "ดีมากกกกก" → "ดีมาก")
      4. Strip leading/trailing whitespace
      5. Normalize whitespace
    """
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse repeated Thai/Latin consonants (social media emphasis)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text).strip()


def remove_hashtags(text: str, keep_text: bool = True) -> str:
    """
    Remove or clean hashtags.

    Args:
        keep_text: If True, keep the hashtag text (remove '#' only).
    """
    if keep_text:
        return re.sub(r"#(\S+)", r"\1", text)
    return re.sub(r"#\S+", "", text).strip()


def remove_mentions(text: str) -> str:
    """Remove @mentions."""
    return re.sub(r"@\S+", "", text).strip()


def clean_tweet(text: str) -> str:
    """
    Full tweet cleaning pipeline for Thai political tweets.

    Paper §A.2: "Tweets were de-duplicated and normalized."
    """
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_hashtags(text, keep_text=True)
    text = normalize_thai(text)
    return text


# ---------------------------------------------------------------------------
# Thai Sentiment Lexicon (political domain)
# ---------------------------------------------------------------------------

POSITIVE_WORDS_TH = {
    "ดี", "เยี่ยม", "สนับสนุน", "รัก", "ชอบ", "ดีใจ", "หวัง", "เชื่อ",
    "ฉลาด", "เก่ง", "สำเร็จ", "เปลี่ยนแปลง", "ก้าวหน้า", "ความหวัง",
    "ประสบความสำเร็จ", "พัฒนา", "มีวิสัยทัศน์", "ยอดเยี่ยม",
}

NEGATIVE_WORDS_TH = {
    "ทุจริต", "ไม่ดี", "เสียใจ", "ต่อต้าน", "โกง", "โกหก", "ล้มเหลว",
    "ผิดพลาด", "น่าผิดหวัง", "เลว", "เสียหาย", "ปัญหา", "วิกฤต",
    "ดูถูก", "ไม่ยุติธรรม", "คอร์รัปชัน", "ทรยศ", "ทำลาย",
}

IRONY_INDICATORS_TH = {
    "555", "ฮ่าๆ", "เหอๆ", "งง", "สุดยอดมากเลย",  # sarcasm markers
    "ใช่เลย", "โอ้โห", "ทำได้ดีมากเนาะ",
}


def lexicon_sentiment(text: str) -> str:
    """
    Rule-based sentiment from Thai political lexicon.

    Paper §D.4: "Emotionally Encoded Stance: Open confrontation is culturally
    discouraged, leading to highly implicit stance signaling embedded in
    emotional or moral appeals."

    Returns: "positive" | "negative" | "neutral"
    """
    pos = sum(1 for w in POSITIVE_WORDS_TH if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS_TH if w in text)
    # Irony: positive words + irony marker → actually negative
    irony = any(w in text for w in IRONY_INDICATORS_TH)
    if irony:
        pos, neg = neg, pos  # flip polarity

    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# Code-Switching Detection (Thai + English mix — common in Thai political tweets)
# ---------------------------------------------------------------------------

def detect_code_switching(text: str) -> dict:
    """
    Detect code-switching between Thai and English.

    Paper §D.4: "code-switching" is listed as a challenge for Thai NLP.
    """
    tokens = text.split()
    thai_tokens  = [t for t in tokens if any(is_thai_char(c) for c in t)]
    en_tokens    = [t for t in tokens if t.isascii() and t.isalpha()]
    total        = max(len(tokens), 1)
    return {
        "thai_ratio"   : len(thai_tokens) / total,
        "english_ratio": len(en_tokens) / total,
        "is_code_switched": (len(thai_tokens) > 0 and len(en_tokens) > 0),
        "thai_tokens"  : thai_tokens,
        "english_tokens": en_tokens,
    }


# ---------------------------------------------------------------------------
# Entity Mention Extractor
# ---------------------------------------------------------------------------

ENTITY_MENTIONS = {
    "paetongtarn": ["แพทองธาร", "อุ๊งอิ๊ง", "Paetongtarn", "นายกฯ หญิง"],
    "thaksin"    : ["ทักษิณ", "แม้ว", "Thaksin"],
    "pita"       : ["พิธา", "Pita", "ก้าวไกล", "Move Forward"],
}


def extract_mentioned_entities(text: str) -> List[str]:
    """
    Find which political entities are mentioned in a tweet.

    Returns list of entity keys mentioned (e.g., ["paetongtarn", "pita"]).
    """
    found = []
    for entity_key, mentions in ENTITY_MENTIONS.items():
        if any(m in text for m in mentions):
            found.append(entity_key)
    return found


# ---------------------------------------------------------------------------
# Entry Point — demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "แพทองธาร เป็นนายกที่ดีมาก ฉันสนับสนุน 100% #เพื่อไทย",
        "ทักษิณทุจริต ไม่ควรอยู่ในการเมืองไทยอีกต่อไป",
        "พิธาพูดเก่งมากเลยนะ 555 ทำได้ดีมากเนาะ #ก้าวไกล",
        "Paetongtarn met with ASEAN leaders today. No comment.",
        "ประเทศไทยต้องการการเปลี่ยนแปลง Move Forward คือความหวัง",
    ]

    print("=== Thai Text Utilities Demo (§D.4) ===\n")
    for text in samples:
        cleaned   = clean_tweet(text)
        sentiment = lexicon_sentiment(cleaned)
        entities  = extract_mentioned_entities(cleaned)
        cs        = detect_code_switching(text)
        ratio     = thai_ratio(text)

        print(f"Original : {text}")
        print(f"Cleaned  : {cleaned}")
        print(f"Thai ratio    : {ratio:.2f}")
        print(f"Sentiment     : {sentiment}")
        print(f"Entities      : {entities}")
        print(f"Code-switched : {cs['is_code_switched']} "
              f"(TH={cs['thai_ratio']:.2f}, EN={cs['english_ratio']:.2f})")
        print("-" * 60)
