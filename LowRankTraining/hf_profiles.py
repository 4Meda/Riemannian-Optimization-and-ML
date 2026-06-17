"""Architecture-specific low-rank conversion profiles for Hugging Face models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TransformerLowRankProfile:
    """Names of modules to target or exclude during low-rank conversion."""

    target_keywords: Tuple[str, ...]
    exclude_keywords: Tuple[str, ...]


BERT_PROFILE = TransformerLowRankProfile(
    target_keywords=(
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ),
    exclude_keywords=(
        "classifier",
        "cls",
        "pooler",
        "embeddings",
        "layernorm",
        "layer_norm",
        "norm",
    ),
)


LLAMA_MISTRAL_PROFILE = TransformerLowRankProfile(
    target_keywords=(
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ),
    exclude_keywords=(
        "embed_tokens",
        "lm_head",
        "norm",
        "layernorm",
        "layer_norm",
    ),
)


PROFILES: Dict[str, TransformerLowRankProfile] = {
    "bert": BERT_PROFILE,
    "roberta": BERT_PROFILE,
    "deberta": BERT_PROFILE,
    "llama": LLAMA_MISTRAL_PROFILE,
    "mistral": LLAMA_MISTRAL_PROFILE,
}


def get_profile(name: str) -> TransformerLowRankProfile:
    """Return a predefined conversion profile."""
    name = name.lower()

    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile '{name}'. Available profiles: {available}")

    return PROFILES[name]
