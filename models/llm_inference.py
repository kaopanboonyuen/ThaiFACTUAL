#!/usr/bin/env python3
# =============================================================================
# ThaiFACTUAL: LLM Inference Wrapper
# =============================================================================
# Paper Section : §3 — Biases of LLMs in Thai Political Stance Detection
#                 §C — Implementation Details
#
# Paper Citation: "LLMs evaluated via OpenAI and HuggingFace APIs
#                  (GPT-4, GPT-3.5, LLaMA-3-8B-chat).
#                  All prompting uses temperature=0.0 to ensure determinism."
#
# Author : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Affil. : Chulalongkorn University & MARSAIL
# Email  : teerapong.pa@chula.ac.th
# Paper  : Debiasing Large Language Models in Thai Political Stance Detection
#          via Counterfactual Calibration
#          EMNLP 2025 – Widening NLP (WiNLP) Workshop, Suzhou, China
# =============================================================================

import os
import json
import time
import logging
from typing import Optional, Literal
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (see also utils/prompts.py for full library)
# ---------------------------------------------------------------------------

SYSTEM_STANCE = (
    "You are a Thai political stance detection assistant. "
    "Your task is to classify the stance of a Thai-language tweet toward a given "
    "political figure as one of: support, against, or neutral. "
    "Output ONLY one word: support, against, or neutral."
)

SYSTEM_DEBIAS = (
    "You are a fairness-aware Thai political stance detection assistant. "
    "IMPORTANT: Do NOT let the emotional tone of the text influence your prediction. "
    "Focus strictly on whether the author supports, opposes, or is neutral toward "
    "the named political figure. "
    "Output ONLY one word: support, against, or neutral."
)

SYSTEM_COT = (
    "You are a Thai political stance detection assistant. "
    "Think step-by-step: first identify the political target, then evaluate whether "
    "the author endorses, criticizes, or is indifferent to that target. "
    "Finally output the stance as: support, against, or neutral."
)


def build_user_prompt(tweet: str, entity_name: str) -> str:
    return (
        f"Tweet (Thai): {tweet}\n"
        f"Target political figure: {entity_name}\n"
        f"Stance:"
    )


# ---------------------------------------------------------------------------
# OpenAI Wrapper
# ---------------------------------------------------------------------------

class OpenAIStanceDetector:
    """
    GPT-4 / GPT-3.5-turbo stance detector.

    Implements three prompt strategies evaluated in Table 1:
      · "raw"    — zero-shot, no debiasing instruction
      · "debias" — debiasing system prompt (GPT-4 Debias Prompt in Table 1)
      · "cot"    — chain-of-thought (LLaMA-3 CoT Prompt equivalent)
    """

    VALID_LABELS = {"support", "against", "neutral"}

    def __init__(
        self,
        model: str = "gpt-4",
        strategy: Literal["raw", "debias", "cot"] = "raw",
        temperature: float = 0.0,   # §C: temperature=0.0 for determinism
        max_tokens: int = 10,
    ):
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Run: pip install openai")

        self.model       = model
        self.strategy    = strategy
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._system_map = {
            "raw"   : SYSTEM_STANCE,
            "debias": SYSTEM_DEBIAS,
            "cot"   : SYSTEM_COT,
        }

    def predict(self, tweet: str, entity_name: str, retries: int = 3) -> str:
        """
        Predict stance for a single tweet.

        Returns one of: "support", "against", "neutral", or "unknown".
        """
        system = self._system_map[self.strategy]
        user   = build_user_prompt(tweet, entity_name)

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model       = self.model,
                    temperature = self.temperature,
                    max_tokens  = self.max_tokens,
                    messages    = [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                raw_output = response.choices[0].message.content.strip().lower()
                # Normalize to valid label
                for label in self.VALID_LABELS:
                    if label in raw_output:
                        return label
                logger.warning(f"Unparseable output: '{raw_output}' — defaulting to neutral")
                return "neutral"

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        return "unknown"

    def predict_batch(self, tweets: list, entity_names: list) -> list:
        """Batch prediction over a list of tweets."""
        preds = []
        for i, (tweet, entity) in enumerate(zip(tweets, entity_names)):
            pred = self.predict(tweet, entity)
            preds.append(pred)
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(tweets)} tweets")
        return preds


# ---------------------------------------------------------------------------
# HuggingFace / LLaMA-3 Wrapper
# ---------------------------------------------------------------------------

class HuggingFaceStanceDetector:
    """
    LLaMA-3 (or any HuggingFace chat model) stance detector.

    Implements CoT prompting strategy as described in Table 1.
    Paper §C: "LLaMA-3-8B-chat evaluated via HuggingFace APIs."
    """

    VALID_LABELS = {"support", "against", "neutral"}

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        strategy: Literal["raw", "cot"] = "cot",
        temperature: float = 0.0,
        max_new_tokens: int = 50,
    ):
        try:
            from transformers import pipeline, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        token = os.getenv("HUGGINGFACE_TOKEN")
        self.pipe = pipeline(
            "text-generation",
            model          = model_id,
            token          = token,
            torch_dtype    = "auto",
            device_map     = "auto",
        )
        self.strategy       = strategy
        self.temperature    = temperature
        self.max_new_tokens = max_new_tokens
        self._system_map    = {"raw": SYSTEM_STANCE, "cot": SYSTEM_COT}

    def _format_chat(self, system: str, user: str) -> str:
        """LLaMA-3 chat format."""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

    def predict(self, tweet: str, entity_name: str) -> str:
        system  = self._system_map[self.strategy]
        user    = build_user_prompt(tweet, entity_name)
        prompt  = self._format_chat(system, user)
        outputs = self.pipe(
            prompt,
            max_new_tokens = self.max_new_tokens,
            do_sample      = (self.temperature > 0),
            temperature    = self.temperature or None,
        )
        raw = outputs[0]["generated_text"].split("assistant")[-1].strip().lower()
        for label in self.VALID_LABELS:
            if label in raw:
                return label
        return "neutral"

    def predict_batch(self, tweets: list, entity_names: list) -> list:
        return [self.predict(t, e) for t, e in zip(tweets, entity_names)]


# ---------------------------------------------------------------------------
# Mock Detector (for unit testing without API keys)
# ---------------------------------------------------------------------------

class MockStanceDetector:
    """
    Deterministic mock detector for testing the full pipeline offline.
    Simulates realistic biased behavior for baseline comparisons.
    """

    def __init__(self, strategy: str = "raw", seed: int = 42):
        import random
        self.rng      = random.Random(seed)
        self.strategy = strategy

    def predict(self, tweet: str, entity_name: str, sentiment: str = None) -> str:
        if self.strategy == "raw" and sentiment:
            # Simulate sentiment leakage (§2: Sentiment-Stance Entanglement)
            if sentiment == "positive":
                return self.rng.choices(["support", "neutral"], weights=[0.8, 0.2])[0]
            elif sentiment == "negative":
                return self.rng.choices(["against", "neutral"], weights=[0.8, 0.2])[0]
        return self.rng.choice(["support", "against", "neutral"])

    def predict_batch(self, tweets: list, entity_names: list, sentiments: list = None) -> list:
        if sentiments:
            return [self.predict(t, e, s) for t, e, s in zip(tweets, entity_names, sentiments)]
        return [self.predict(t, e) for t, e in zip(tweets, entity_names)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_detector(
    backend  : Literal["openai", "huggingface", "mock"] = "mock",
    strategy : str = "raw",
    **kwargs,
):
    """
    Factory function to get the right detector.

    Usage:
        detector = get_detector("openai", strategy="debias", model="gpt-4")
        pred = detector.predict(tweet, entity_name)
    """
    if backend == "openai":
        return OpenAIStanceDetector(strategy=strategy, **kwargs)
    elif backend == "huggingface":
        return HuggingFaceStanceDetector(strategy=strategy, **kwargs)
    elif backend == "mock":
        return MockStanceDetector(strategy=strategy, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Entry Point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== LLM Inference Wrapper — Smoke Test (Mock Mode) ===")
    print("Paper §C: temperature=0.0, deterministic inference\n")

    samples = [
        ("แพทองธาร เป็นนายกที่ดีมาก ฉันสนับสนุน", "Paetongtarn Shinawatra", "positive"),
        ("ทักษิณคือคนทุจริต ไม่ควรกลับมา",         "Thaksin Shinawatra",     "negative"),
        ("พิธาให้สัมภาษณ์เรื่องนโยบายการศึกษา",    "Pita Limjaroenrat",       "neutral"),
    ]

    for strategy in ["raw", "debias"]:
        detector = get_detector("mock", strategy=strategy)
        print(f"Strategy: {strategy.upper()}")
        for tweet, entity, sentiment in samples:
            pred = detector.predict(tweet, entity, sentiment)
            print(f"  [{entity[:15]:15s}] sentiment={sentiment:8s} → pred={pred}")
        print()
