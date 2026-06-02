#!/usr/bin/env python3
# =============================================================================
# ThaiFACTUAL: Prompt Templates Library
# =============================================================================
# Paper Section : §3 — Biases of LLMs in Thai Political Stance Detection
#                 Table 1 — prompt strategies
#
# Paper Citation: "All prompting uses temperature=0.0 to ensure determinism.
#                  For ThaiFACTUAL, counterfactual data was injected as an
#                  auxiliary correction layer."
#
# Prompt Strategies:
#   1. RAW       — Zero-shot, no debiasing      (GPT-4 Raw in Table 1)
#   2. DEBIAS    — Explicit fairness instruction (GPT-4 Debias in Table 1)
#   3. COT       — Chain-of-Thought             (LLaMA-3 CoT in Table 1)
#   4. THAIFACTUAL — CF-augmented calibration   (ThaiFACTUAL in Table 1)
#
# Author : Teerapong Panboonyuen (aka Kao Panboonyuen)
# Affil. : Chulalongkorn University & MARSAIL
# Email  : teerapong.pa@chula.ac.th
# Paper  : Debiasing Large Language Models in Thai Political Stance Detection
#          via Counterfactual Calibration
#          EMNLP 2025 – Widening NLP (WiNLP) Workshop, Suzhou, China
# =============================================================================

from typing import Optional


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

SYSTEM_RAW = """You are a Thai political stance detection assistant.
Classify the stance of a Thai-language tweet toward a given political figure.
Output ONLY one word: support, against, or neutral."""

SYSTEM_DEBIAS = """You are a fairness-aware Thai political stance detection assistant.
IMPORTANT INSTRUCTIONS:
- Do NOT let emotional tone or sentiment bias your prediction.
- Focus strictly on whether the author supports, opposes, or is neutral toward the named figure.
- Ignore positive/negative language if it does not reflect a clear political position.
Output ONLY one word: support, against, or neutral."""

SYSTEM_COT = """You are a Thai political stance detection assistant.
Think step-by-step:
  Step 1: Identify the political target mentioned in the tweet.
  Step 2: Determine the author's political position toward that target.
  Step 3: Ask yourself: is this stance driven by content or just tone?
  Step 4: Output the stance as exactly one word.
Output ONLY one word: support, against, or neutral."""

SYSTEM_THAIFACTUAL = """You are a Thai political stance detection assistant operating with
counterfactual calibration. You will receive the original tweet AND a counterfactual variant
where the political entity has been swapped.

Rules:
  1. If swapping the entity changes your prediction, your original prediction may be driven
     by entity bias (not content). Re-evaluate carefully.
  2. If sentiment is positive/negative but the content does not explicitly endorse or oppose
     the target, output neutral.
  3. Use the provided rationale to check whether stance and content align.

Output ONLY one word: support, against, or neutral."""


# ---------------------------------------------------------------------------
# User Prompt Builders
# ---------------------------------------------------------------------------

def build_raw_prompt(tweet: str, entity_name: str) -> str:
    """Zero-shot stance prompt (GPT-4 Raw, Table 1)."""
    return (
        f"Tweet (Thai): {tweet}\n"
        f"Political figure: {entity_name}\n"
        f"Stance:"
    )


def build_debias_prompt(tweet: str, entity_name: str) -> str:
    """Debiased prompt (GPT-4 Debias Prompt, Table 1)."""
    return (
        f"Tweet (Thai): {tweet}\n"
        f"Political figure: {entity_name}\n"
        f"Remember: do NOT let sentiment drive your answer.\n"
        f"Stance:"
    )


def build_cot_prompt(tweet: str, entity_name: str) -> str:
    """Chain-of-thought prompt (LLaMA-3 CoT Prompt, Table 1)."""
    return (
        f"Tweet (Thai): {tweet}\n"
        f"Political figure: {entity_name}\n\n"
        f"Now think step-by-step:\n"
        f"1. Who is the political target?\n"
        f"2. What is the author's position toward them?\n"
        f"3. Is this stance content-based or sentiment-based?\n"
        f"Final stance (one word):"
    )


def build_thaifactual_prompt(
    tweet          : str,
    entity_name    : str,
    cf_tweet       : str,
    cf_entity_name : str,
    rationale      : str = "",
    sentiment      : str = "",
) -> str:
    """
    ThaiFACTUAL counterfactual-augmented prompt (Figure 1d, §4, Table 1).

    This is the core prompt innovation of the paper:
    By showing the model the CF variant alongside the original,
    entity-bias becomes visible and can be corrected.

    Paper §4: "By introducing counterfactual perturbations to both causal
    (topic-related) and non-causal (sentiment or named entities) dimensions,
    we enable the calibration model to better disentangle spurious from
    reliable cues."
    """
    parts = [
        f"=== Original Tweet ===",
        f"Tweet (Thai): {tweet}",
        f"Political figure: {entity_name}",
    ]
    if sentiment:
        parts.append(f"Sentiment: {sentiment}")
    parts += [
        f"\n=== Counterfactual Variant (Entity Swapped) ===",
        f"Tweet (Thai): {cf_tweet}",
        f"Political figure: {cf_entity_name}",
    ]
    if rationale:
        parts += [
            f"\n=== Rationale ===",
            f"{rationale}",
        ]
    parts += [
        f"\n=== Your Task ===",
        f"Based on the original tweet and the counterfactual comparison above,",
        f"what is the stance of the ORIGINAL tweet toward {entity_name}?",
        f"Output ONLY one word: support, against, or neutral.",
        f"Stance:",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt Registry
# ---------------------------------------------------------------------------

PROMPT_REGISTRY = {
    "raw"         : (SYSTEM_RAW,          build_raw_prompt),
    "debias"      : (SYSTEM_DEBIAS,        build_debias_prompt),
    "cot"         : (SYSTEM_COT,           build_cot_prompt),
    "thaifactual" : (SYSTEM_THAIFACTUAL,   build_thaifactual_prompt),
}


def get_prompt(
    strategy     : str,
    tweet        : str,
    entity_name  : str,
    **kwargs,
) -> tuple:
    """
    Retrieve (system_prompt, user_prompt) for a given strategy.

    Usage:
        system, user = get_prompt("thaifactual", tweet, entity,
                                  cf_tweet=cf_text, cf_entity_name=cf_entity,
                                  rationale=rationale, sentiment=sentiment)
    """
    if strategy not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: {list(PROMPT_REGISTRY.keys())}")
    system, builder = PROMPT_REGISTRY[strategy]
    user = builder(tweet, entity_name, **kwargs)
    return system, user


# ---------------------------------------------------------------------------
# Entry Point — print all prompt examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tweet_th  = "แพทองธาร เป็นนายกที่มีวิสัยทัศน์ ประเทศไทยจะดีขึ้น"
    tweet_cf  = "ทักษิณ เป็นนายกที่มีวิสัยทัศน์ ประเทศไทยจะดีขึ้น"
    entity    = "Paetongtarn Shinawatra"
    cf_entity = "Thaksin Shinawatra"
    rationale = "Speaker explicitly praises PM's vision; positive sentiment."

    print("=" * 60)
    print("PROMPT TEMPLATES — ThaiFACTUAL Paper (All Strategies)")
    print("=" * 60)

    for strategy in ["raw", "debias", "cot"]:
        system, user = get_prompt(strategy, tweet_th, entity)
        print(f"\n{'─'*60}")
        print(f"Strategy: {strategy.upper()} (Table 1)")
        print(f"{'─'*60}")
        print(f"[SYSTEM]\n{system}")
        print(f"\n[USER]\n{user}")

    # ThaiFACTUAL requires extra args
    system, user = get_prompt(
        "thaifactual", tweet_th, entity,
        cf_tweet=tweet_cf, cf_entity_name=cf_entity,
        rationale=rationale, sentiment="positive"
    )
    print(f"\n{'─'*60}")
    print("Strategy: THAIFACTUAL (★ Core Contribution — Figure 1d, Table 1)")
    print(f"{'─'*60}")
    print(f"[SYSTEM]\n{system}")
    print(f"\n[USER]\n{user}")
