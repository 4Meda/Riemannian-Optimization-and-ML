"""Utilities for applying low-rank layers to Transformer architectures."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn

try:
    from .layers import LowRankLinear
except ImportError:  # Allows running this file directly.
    from layers import LowRankLinear


DEFAULT_TARGET_KEYWORDS = (
    # BERT / ViT style
    "query",
    "key",
    "value",
    "dense",
    # LLaMA / Mistral style
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # PyTorch TransformerEncoderLayer style
    "out_proj",
    "linear1",
    "linear2",
    # Generic MLP naming
    "fc",
    "fc1",
    "fc2",
)

DEFAULT_EXCLUDE_KEYWORDS = (
    # Usually keep task heads dense at first.
    "classifier",
    "classification_head",
    "score",
    "lm_head",
    "head",
    # Usually avoid poolers for BERT-style classifiers.
    "pooler",
    # Usually not Linear, but included for safety.
    "embedding",
    "embeddings",
    "embed_tokens",
    "wte",
    "wpe",
    # Usually not Linear, but included for safety.
    "norm",
    "layernorm",
    "layer_norm",
    "ln_",
)


@dataclass
class LowRankConversionReport:
    """Summary of a low-rank conversion."""

    converted: List[Dict[str, object]]
    skipped: List[Dict[str, object]]

    @property
    def num_converted(self) -> int:
        """Number of converted modules."""
        return len(self.converted)

    def print(self) -> None:
        """Print a human-readable conversion summary."""
        print(f"Converted modules: {len(self.converted)}")
        for item in self.converted:
            print(
                "  "
                f"{item['name']}: "
                f"{item['in_features']} -> {item['out_features']}, "
                f"rank={item['rank']}, "
                f"params {item['dense_params']} -> {item['lowrank_params']}"
            )

        if self.skipped:
            print()
            print(f"Skipped modules: {len(self.skipped)}")
            for item in self.skipped:
                print(f"  {item['name']}: {item['reason']}")


def _contains_any(name: str, keywords: Optional[Iterable[str]]) -> bool:
    """Return True if ``name`` contains any keyword."""
    if keywords is None:
        return True

    name_lower = name.lower()

    return any(keyword.lower() in name_lower for keyword in keywords)


def _resolve_rank(
    module_name: str,
    linear: nn.Linear,
    rank: Optional[int],
    rank_ratio: Optional[float],
    rank_map: Optional[Mapping[str, int]],
    min_rank: int,
    max_rank: Optional[int],
) -> int:
    """Resolve the low-rank dimension for one linear layer."""
    full_rank = min(linear.in_features, linear.out_features)

    if rank_map is not None:
        module_name_lower = module_name.lower()

        for key, value in rank_map.items():
            if key.lower() in module_name_lower:
                rank = int(value)
                break

    if rank is not None and rank_ratio is not None:
        raise ValueError("Use either rank or rank_ratio, not both.")

    if rank is None:
        if rank_ratio is None:
            raise ValueError("Either rank, rank_ratio, or rank_map must be provided.")

        if not 0 < rank_ratio <= 1:
            raise ValueError("rank_ratio must satisfy 0 < rank_ratio <= 1.")

        rank = math.ceil(rank_ratio * full_rank)

    rank = int(rank)
    rank = max(rank, int(min_rank))

    if max_rank is not None:
        rank = min(rank, int(max_rank))

    rank = min(rank, full_rank)

    if rank <= 0:
        raise ValueError("Resolved rank must be positive.")

    return rank


def _parameter_count_linear(linear: nn.Linear) -> int:
    """Count parameters of a dense linear layer."""
    count = linear.in_features * linear.out_features

    if linear.bias is not None:
        count += linear.out_features

    return count


def _parameter_count_lowrank(linear: nn.Linear, rank: int) -> int:
    """Count parameters of the corresponding low-rank linear layer."""
    count = rank * (linear.in_features + linear.out_features)

    if linear.bias is not None:
        count += linear.out_features

    return count


def replace_transformer_linear_layers(
    model: nn.Module,
    rank: Optional[int] = None,
    rank_ratio: Optional[float] = None,
    rank_map: Optional[Mapping[str, int]] = None,
    target_keywords: Optional[Iterable[str]] = DEFAULT_TARGET_KEYWORDS,
    exclude_keywords: Iterable[str] = DEFAULT_EXCLUDE_KEYWORDS,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
    inplace: bool = True,
    copy_weights: bool = True,
    skip_if_no_compression: bool = True,
) -> Tuple[nn.Module, LowRankConversionReport]:
    """Replace selected transformer ``nn.Linear`` layers by ``LowRankLinear``.

    Args:
        model: Transformer model.
        rank: Fixed rank for every selected layer.
        rank_ratio: Rank as a fraction of each layer's full rank.
        rank_map: Optional dictionary mapping name substrings to custom ranks.
            For example: ``{"q_proj": 64, "down_proj": 128}``.
        target_keywords: Only modules whose full name contains one of these
            keywords are converted. Use ``None`` to consider all linear layers.
        exclude_keywords: Modules whose full name contains one of these
            keywords are skipped.
        min_rank: Minimum rank.
        max_rank: Optional maximum rank.
        inplace: If True, modify the input model directly.
        copy_weights: If True, initialize low-rank layers from dense weights
            using truncated SVD.
        skip_if_no_compression: If True, skip layers where the low-rank
            representation would not reduce parameter count.

    Returns:
        Tuple ``(model, report)``.
    """
    if not inplace:
        model = copy.deepcopy(model)

    converted: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    def convert_children(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear):
                if _contains_any(full_name, exclude_keywords):
                    skipped.append(
                        {
                            "name": full_name,
                            "reason": "matched exclude keyword",
                        }
                    )
                    continue

                if not _contains_any(full_name, target_keywords):
                    skipped.append(
                        {
                            "name": full_name,
                            "reason": "did not match target keyword",
                        }
                    )
                    continue

                layer_rank = _resolve_rank(
                    module_name=full_name,
                    linear=child,
                    rank=rank,
                    rank_ratio=rank_ratio,
                    rank_map=rank_map,
                    min_rank=min_rank,
                    max_rank=max_rank,
                )

                dense_params = _parameter_count_linear(child)
                lowrank_params = _parameter_count_lowrank(child, layer_rank)

                if skip_if_no_compression and lowrank_params >= dense_params:
                    skipped.append(
                        {
                            "name": full_name,
                            "reason": (
                                "low-rank layer would not reduce parameter count "
                                f"({dense_params} -> {lowrank_params})"
                            ),
                        }
                    )
                    continue

                if copy_weights:
                    lowrank_child = LowRankLinear.from_linear(child, layer_rank)
                else:
                    lowrank_child = LowRankLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        rank=layer_rank,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )

                setattr(parent, child_name, lowrank_child)

                converted.append(
                    {
                        "name": full_name,
                        "in_features": child.in_features,
                        "out_features": child.out_features,
                        "rank": layer_rank,
                        "dense_params": dense_params,
                        "lowrank_params": lowrank_params,
                    }
                )
            else:
                convert_children(child, full_name)

    convert_children(model)

    return model, LowRankConversionReport(converted=converted, skipped=skipped)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {"total": total, "trainable": trainable}


def count_lowrank_layers(model: nn.Module) -> int:
    """Count low-rank linear layers in a model."""
    return sum(1 for module in model.modules() if isinstance(module, LowRankLinear))


def lowrank_parameter_groups(
    model: nn.Module,
    weight_decay: float = 1e-2,
    lowrank_weight_decay: Optional[float] = None,
    bias_weight_decay: float = 0.0,
):
    """Create optimizer parameter groups for low-rank transformer training.

    The returned groups include role labels:

    - ``role='U'`` for low-rank U factors;
    - ``role='Y'`` for low-rank Y factors;
    - ``role=None`` for standard parameters.

    Returns:
        Tuple ``(parameter_groups, qr_pairs)``.
    """
    if lowrank_weight_decay is None:
        lowrank_weight_decay = weight_decay

    U_parameters: List[torch.nn.Parameter] = []
    Y_parameters: List[torch.nn.Parameter] = []
    decay_parameters: List[torch.nn.Parameter] = []
    no_decay_parameters: List[torch.nn.Parameter] = []

    qr_pairs = []
    seen_parameter_ids = set()

    for module in model.modules():
        if isinstance(module, LowRankLinear):
            if module.U.requires_grad:
                U_parameters.append(module.U)
                seen_parameter_ids.add(id(module.U))

            if module.Y.requires_grad:
                Y_parameters.append(module.Y)
                seen_parameter_ids.add(id(module.Y))

            if module.bias is not None and module.bias.requires_grad:
                no_decay_parameters.append(module.bias)
                seen_parameter_ids.add(id(module.bias))

            qr_pairs.append((module.U, module.Y))

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if id(parameter) in seen_parameter_ids:
            continue

        name_lower = name.lower()

        if (
            parameter.ndim == 1
            or name_lower.endswith("bias")
            or "norm" in name_lower
            or "layernorm" in name_lower
            or "layer_norm" in name_lower
        ):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    parameter_groups = []

    if U_parameters:
        parameter_groups.append(
            {
                "params": U_parameters,
                "role": "U",
                "weight_decay": lowrank_weight_decay,
            }
        )

    if Y_parameters:
        parameter_groups.append(
            {
                "params": Y_parameters,
                "role": "Y",
                "weight_decay": lowrank_weight_decay,
            }
        )

    if decay_parameters:
        parameter_groups.append(
            {
                "params": decay_parameters,
                "role": None,
                "weight_decay": weight_decay,
            }
        )

    if no_decay_parameters:
        parameter_groups.append(
            {
                "params": no_decay_parameters,
                "role": None,
                "weight_decay": bias_weight_decay,
            }
        )

    return parameter_groups, qr_pairs
