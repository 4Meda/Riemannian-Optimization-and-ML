"""Convert selected Hugging Face transformer layers to low-rank layers.

Examples:

    python LowRankTraining/convert_hf_model.py \
      --model bert-base-uncased \
      --task masked-lm \
      --profile bert \
      --rank 64

    python LowRankTraining/convert_hf_model.py \
      --model hf-internal-testing/tiny-random-LlamaForCausalLM \
      --task causal-lm \
      --profile llama \
      --rank 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LowRankTraining.hf_profiles import get_profile
from LowRankTraining.transformer_utils import (
    count_lowrank_layers,
    count_parameters,
    replace_transformer_linear_layers,
)


MODEL_LOADERS = {
    "base": AutoModel,
    "masked-lm": AutoModelForMaskedLM,
    "causal-lm": AutoModelForCausalLM,
    "sequence-classification": AutoModelForSequenceClassification,
}


def parse_dtype(dtype: str):
    """Parse a dtype string for Hugging Face model loading."""
    dtype = dtype.lower()

    if dtype == "auto":
        return "auto"

    if dtype == "float32":
        return torch.float32

    if dtype == "float16":
        return torch.float16

    if dtype == "bfloat16":
        return torch.bfloat16

    raise ValueError("dtype must be one of: auto, float32, float16, bfloat16")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert selected Hugging Face transformer layers to low rank.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="base",
        choices=sorted(MODEL_LOADERS),
        help="Model class/task to load.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        choices=["bert", "roberta", "deberta", "llama", "mistral"],
        help="Architecture profile.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Fixed low-rank dimension.",
    )
    parser.add_argument(
        "--rank-ratio",
        type=float,
        default=None,
        help="Rank as a fraction of the full matrix rank.",
    )
    parser.add_argument(
        "--min-rank",
        type=int,
        default=1,
        help="Minimum allowed rank.",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Maximum allowed rank.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to move the model to when not using device_map='auto'.",
    )
    parser.add_argument(
        "--device-map-auto",
        action="store_true",
        help="Use Hugging Face device_map='auto'. Useful for larger models.",
    )
    parser.add_argument(
        "--no-copy-weights",
        action="store_true",
        help="Do not initialize low-rank layers from dense weights.",
    )
    parser.add_argument(
        "--no-skip-if-no-compression",
        action="store_true",
        help="Convert even if the low-rank layer has more parameters.",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Optional directory for saving a low-rank state_dict checkpoint.",
    )

    return parser.parse_args()


def load_model(args: argparse.Namespace):
    """Load a Hugging Face model."""
    loader = MODEL_LOADERS[args.task]
    torch_dtype = parse_dtype(args.dtype)

    load_kwargs = {}

    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    if args.device_map_auto:
        load_kwargs["device_map"] = "auto"

    model = loader.from_pretrained(args.model, **load_kwargs)

    if not args.device_map_auto:
        model = model.to(args.device)

    return model


def save_lowrank_checkpoint(
    model,
    args: argparse.Namespace,
    metadata,
) -> None:
    """Save converted model state_dict and conversion metadata."""
    output_dir = Path(args.save_checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "lowrank_state_dict.pt"
    metadata_path = output_dir / "conversion_metadata.json"

    torch.save(model.state_dict(), checkpoint_path)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print()
    print(f"Saved state_dict to: {checkpoint_path}")
    print(f"Saved metadata to:   {metadata_path}")
    print()
    print("Note: to reload this checkpoint, first load the same Hugging Face")
    print("model, apply the same low-rank conversion, then load this state_dict.")


def main() -> None:
    """Run conversion."""
    args = parse_args()

    if args.rank is None and args.rank_ratio is None:
        raise ValueError("Provide either --rank or --rank-ratio.")

    if args.rank is not None and args.rank_ratio is not None:
        raise ValueError("Use either --rank or --rank-ratio, not both.")

    profile = get_profile(args.profile)

    print(f"Loading model: {args.model}")
    model = load_model(args)

    before = count_parameters(model)

    print()
    print("Parameter count before conversion:")
    print(before)

    model, report = replace_transformer_linear_layers(
        model,
        rank=args.rank,
        rank_ratio=args.rank_ratio,
        target_keywords=profile.target_keywords,
        exclude_keywords=profile.exclude_keywords,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        inplace=True,
        copy_weights=not args.no_copy_weights,
        skip_if_no_compression=not args.no_skip_if_no_compression,
    )

    after = count_parameters(model)

    print()
    report.print()

    print()
    print("Parameter count after conversion:")
    print(after)

    print()
    print(f"Number of LowRankLinear layers: {count_lowrank_layers(model)}")

    metadata = {
        "model": args.model,
        "task": args.task,
        "profile": args.profile,
        "rank": args.rank,
        "rank_ratio": args.rank_ratio,
        "min_rank": args.min_rank,
        "max_rank": args.max_rank,
        "copy_weights": not args.no_copy_weights,
        "parameter_count_before": before,
        "parameter_count_after": after,
        "num_lowrank_layers": count_lowrank_layers(model),
        "converted_modules": report.converted,
    }

    if args.save_checkpoint is not None:
        save_lowrank_checkpoint(model, args, metadata)


if __name__ == "__main__":
    main()
