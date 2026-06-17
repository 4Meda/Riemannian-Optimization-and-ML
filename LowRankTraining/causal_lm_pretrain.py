"""Causal language-model pretraining for low-rank transformer models.

Example using a saved tiny Mistral low-rank checkpoint:

    python LowRankTraining/causal_lm_pretrain.py \
      --model hf-internal-testing/tiny-random-MistralForCausalLM \
      --tokenizer hf-internal-testing/llama-tokenizer \
      --profile mistral \
      --rank 4 \
      --lowrank-checkpoint LowRankTraining/checkpoints/tiny_mistral_rank4/lowrank_state_dict.pt \
      --dataset wikitext \
      --dataset-config wikitext-2-raw-v1 \
      --max-steps 5 \
      --logging-steps 1 \
      --eval-steps 5 \
      --save-steps 5 \
      --output-dir LowRankTraining/outputs/tiny_mistral_rank4_test

Resume example:

    python LowRankTraining/causal_lm_pretrain.py \
      --model hf-internal-testing/tiny-random-MistralForCausalLM \
      --tokenizer hf-internal-testing/llama-tokenizer \
      --profile mistral \
      --rank 4 \
      --resume-from-checkpoint LowRankTraining/outputs/tiny_mistral_rank4_test/checkpoint-step-5 \
      --max-steps 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Reduce noisy TensorFlow-related imports/messages when using transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure terminal logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_dtype(dtype: str):
    """Parse a dtype string."""
    dtype = dtype.lower()

    if dtype == "float32":
        return torch.float32

    if dtype == "float16":
        return torch.float16

    if dtype == "bfloat16":
        return torch.bfloat16

    raise ValueError("dtype must be one of: float32, float16, bfloat16")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Continue causal LM pretraining with low-rank transformer layers.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Original Hugging Face model name or local path.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name/path. Defaults to --model.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        choices=["llama", "mistral"],
        help="Low-rank conversion profile.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Fixed low-rank dimension. Must match checkpoint if loading one.",
    )
    parser.add_argument(
        "--rank-ratio",
        type=float,
        default=None,
        help="Rank as a fraction of the full matrix rank.",
    )
    parser.add_argument(
        "--lowrank-checkpoint",
        type=str,
        default=None,
        help="Optional model-only low-rank state_dict checkpoint to initialise from.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory to resume from. Loads model, optimizer, and metadata if available.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Optional Hugging Face dataset config.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Dataset split used for training.",
    )
    parser.add_argument(
        "--validation-split",
        type=str,
        default="validation",
        help="Dataset split used for evaluation.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column in the dataset.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Token sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches per optimizer step.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of optimizer steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every this many optimizer steps.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=0,
        help="Evaluate every this many optimizer steps. Use 0 to disable evaluation.",
    )
    parser.add_argument(
        "--eval-before-training",
        action="store_true",
        help="Run validation evaluation before training starts.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=50,
        help="Maximum number of validation batches per evaluation. Use 0 for all.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save every this many optimizer steps. Use 0 to disable periodic saving.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="LowRankTraining/outputs/causal_lm_pretrain",
        help="Output directory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Random seed.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.rank is None and args.rank_ratio is None:
        raise ValueError("Provide either --rank or --rank-ratio.")

    if args.rank is not None and args.rank_ratio is not None:
        raise ValueError("Use either --rank or --rank-ratio, not both.")

    if args.lowrank_checkpoint is not None and args.resume_from_checkpoint is not None:
        raise ValueError("Use either --lowrank-checkpoint or --resume-from-checkpoint, not both.")

    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    if args.logging_steps <= 0:
        raise ValueError("--logging-steps must be positive.")

    if args.eval_steps < 0:
        raise ValueError("--eval-steps cannot be negative.")

    if args.save_steps < 0:
        raise ValueError("--save-steps cannot be negative.")


def load_hf_dataset(args: argparse.Namespace, split: str):
    """Load a Hugging Face dataset split."""
    if args.dataset_config:
        return load_dataset(args.dataset, args.dataset_config, split=split)

    return load_dataset(args.dataset, split=split)


def prepare_lm_dataset(args: argparse.Namespace, tokenizer, split: str):
    """Load, tokenize, and chunk a text dataset."""
    LOGGER.info(
        "Loading dataset split: dataset=%s config=%s split=%s",
        args.dataset,
        args.dataset_config,
        split,
    )

    dataset = load_hf_dataset(args, split=split)

    if args.text_column not in dataset.column_names:
        raise ValueError(
            f"Text column '{args.text_column}' not found. "
            f"Available columns: {dataset.column_names}"
        )

    def tokenize_function(examples):
        return tokenizer(examples[args.text_column])

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {split} split",
    )

    if tokenizer.model_max_length and tokenizer.model_max_length < 1_000_000:
        block_size = min(args.block_size, tokenizer.model_max_length)
    else:
        block_size = args.block_size

    def group_texts(examples):
        concatenated = {key: sum(examples[key], []) for key in examples.keys()}
        total_length = len(concatenated["input_ids"])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            total_length = 0

        return {
            key: [
                values[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping {split} split into blocks of {block_size}",
    )

    LOGGER.info("%s split contains %d language-model blocks", split, len(lm_dataset))

    return lm_dataset


def make_dataloader(
    dataset,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    """Create a language-modelling DataLoader."""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_perplexity(loss: float) -> float:
    """Compute perplexity safely from cross-entropy loss."""
    if loss >= 20:
        return float("inf")

    return float(math.exp(loss))


@torch.no_grad()
def evaluate_model(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    max_eval_batches: int = 50,
) -> Dict[str, float]:
    """Evaluate the model on a validation DataLoader."""
    model.eval()

    losses = []
    n_batches = 0

    for batch in eval_loader:
        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = model(**batch)
        losses.append(float(outputs.loss.item()))

        n_batches += 1

        if max_eval_batches > 0 and n_batches >= max_eval_batches:
            break

    model.train()

    if not losses:
        return {"eval_loss": float("nan"), "eval_perplexity": float("nan")}

    eval_loss = sum(losses) / len(losses)
    eval_perplexity = compute_perplexity(eval_loss)

    return {
        "eval_loss": eval_loss,
        "eval_perplexity": eval_perplexity,
    }


def checkpoint_paths(checkpoint: str | Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Resolve checkpoint paths.

    Args:
        checkpoint: Either a checkpoint directory or a model state_dict file.

    Returns:
        Tuple ``(model_state_path, optimizer_state_path, metadata_path)``.
    """
    checkpoint = Path(checkpoint)

    if checkpoint.is_dir():
        return (
            checkpoint / "lowrank_state_dict.pt",
            checkpoint / "optimizer_state.pt",
            checkpoint / "training_metadata.json",
        )

    return (
        checkpoint,
        None,
        None,
    )


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to the selected device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_model_state(model, checkpoint: str | Path, strict: bool = True) -> None:
    """Load a model state_dict checkpoint."""
    model_state_path, _, _ = checkpoint_paths(checkpoint)

    if not model_state_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_state_path}")

    LOGGER.info("Loading model checkpoint: %s", model_state_path)
    state_dict = torch.load(model_state_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=strict)


def load_resume_state(
    model,
    optimizer: torch.optim.Optimizer,
    checkpoint: str | Path,
    device: torch.device,
):
    """Load model, optimizer, and metadata from a resume checkpoint."""
    model_state_path, optimizer_state_path, metadata_path = checkpoint_paths(checkpoint)

    if not model_state_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_state_path}")

    LOGGER.info("Resuming model from: %s", model_state_path)
    state_dict = torch.load(model_state_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    start_step = 0
    history = []
    metadata = {}

    if metadata_path is not None and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        start_step = int(metadata.get("step", 0))
        history = metadata.get("history", [])

        LOGGER.info("Loaded resume metadata from: %s", metadata_path)
        LOGGER.info("Resuming from optimizer step: %d", start_step)
    else:
        LOGGER.warning("No training metadata found. Resuming from step 0.")

    if optimizer_state_path is not None and optimizer_state_path.exists():
        LOGGER.info("Loading optimizer state from: %s", optimizer_state_path)
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
        move_optimizer_state_to_device(optimizer, device)
    else:
        LOGGER.warning("No optimizer state found. Optimizer will restart from scratch.")

    return start_step, history, metadata


def save_training_state(
    model,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
    history,
    conversion_metadata,
) -> Path:
    """Save model, optimizer, tokenizer, and training metadata."""
    checkpoint_dir = output_dir / f"checkpoint-step-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_state_path = checkpoint_dir / "lowrank_state_dict.pt"
    optimizer_state_path = checkpoint_dir / "optimizer_state.pt"
    metadata_path = checkpoint_dir / "training_metadata.json"

    torch.save(model.state_dict(), model_state_path)
    torch.save(optimizer.state_dict(), optimizer_state_path)
    tokenizer.save_pretrained(checkpoint_dir)

    metadata = {
        "step": step,
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "profile": args.profile,
        "rank": args.rank,
        "rank_ratio": args.rank_ratio,
        "lowrank_checkpoint": args.lowrank_checkpoint,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "text_column": args.text_column,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size or args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "history": history,
        "conversion": conversion_metadata,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    LOGGER.info("Saved checkpoint to %s", checkpoint_dir)

    return checkpoint_dir


def main() -> None:
    """Run low-rank causal LM pretraining."""
    setup_logging()
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    dtype = parse_dtype(args.dtype)

    if device.type == "cpu" and dtype != torch.float32:
        LOGGER.warning("Using %s on CPU may be slow or unsupported.", args.dtype)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_name = args.tokenizer or args.model

    LOGGER.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )

    profile = get_profile(args.profile)

    LOGGER.info("Applying low-rank conversion")
    before = count_parameters(model)

    should_copy_weights = args.lowrank_checkpoint is None and args.resume_from_checkpoint is None

    model, report = replace_transformer_linear_layers(
        model,
        rank=args.rank,
        rank_ratio=args.rank_ratio,
        target_keywords=profile.target_keywords,
        exclude_keywords=profile.exclude_keywords,
        inplace=True,
        copy_weights=should_copy_weights,
    )

    after = count_parameters(model)

    conversion_metadata = {
        "parameter_count_before": before,
        "parameter_count_after": after,
        "num_lowrank_layers": count_lowrank_layers(model),
        "converted_modules": report.converted,
    }

    LOGGER.info("Parameter count before conversion: %s", before)
    LOGGER.info("Parameter count after conversion:  %s", after)
    LOGGER.info("LowRankLinear layers: %s", count_lowrank_layers(model))

    if args.lowrank_checkpoint is not None:
        load_model_state(model, args.lowrank_checkpoint, strict=True)

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    optimizer_step = 0
    history = []

    if args.resume_from_checkpoint is not None:
        optimizer_step, history, _ = load_resume_state(
            model=model,
            optimizer=optimizer,
            checkpoint=args.resume_from_checkpoint,
            device=device,
        )

    LOGGER.info("Preparing training data")
    train_dataset = prepare_lm_dataset(args, tokenizer, split=args.train_split)

    train_loader = make_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    eval_loader = None

    if args.eval_steps > 0 or args.eval_before_training:
        LOGGER.info("Preparing validation data")
        eval_dataset = prepare_lm_dataset(args, tokenizer, split=args.validation_split)

        eval_loader = make_dataloader(
            eval_dataset,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    LOGGER.info("Starting training")
    LOGGER.info("Device: %s", device)
    LOGGER.info("Starting optimizer step: %d", optimizer_step)
    LOGGER.info("Target max optimizer steps: %d", args.max_steps)

    if args.eval_before_training and eval_loader is not None:
        metrics = evaluate_model(
            model,
            eval_loader,
            device=device,
            max_eval_batches=args.max_eval_batches,
        )

        LOGGER.info(
            "pretrain_eval eval_loss=%.6f eval_perplexity=%.4f",
            metrics["eval_loss"],
            metrics["eval_perplexity"],
        )

        history.append(
            {
                "step": optimizer_step,
                "type": "eval",
                **metrics,
                "elapsed_seconds": 0.0,
            }
        )

    start_time = time.time()
    micro_step = 0
    running_loss = 0.0
    running_micro_batches = 0
    last_saved_step = None

    optimizer.zero_grad(set_to_none=True)

    while optimizer_step < args.max_steps:
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / args.gradient_accumulation_steps
            loss.backward()

            micro_step += 1
            running_loss += float(raw_loss.item())
            running_micro_batches += 1

            if micro_step % args.gradient_accumulation_steps != 0:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1

            if optimizer_step % args.logging_steps == 0:
                avg_loss = running_loss / max(running_micro_batches, 1)
                perplexity = compute_perplexity(avg_loss)
                elapsed = time.time() - start_time

                LOGGER.info(
                    "step=%d train_loss=%.6f train_perplexity=%.4f elapsed=%.1fs",
                    optimizer_step,
                    avg_loss,
                    perplexity,
                    elapsed,
                )

                history.append(
                    {
                        "step": optimizer_step,
                        "type": "train",
                        "train_loss": avg_loss,
                        "train_perplexity": perplexity,
                        "elapsed_seconds": elapsed,
                    }
                )

                running_loss = 0.0
                running_micro_batches = 0

            if (
                args.eval_steps > 0
                and eval_loader is not None
                and optimizer_step % args.eval_steps == 0
            ):
                metrics = evaluate_model(
                    model,
                    eval_loader,
                    device=device,
                    max_eval_batches=args.max_eval_batches,
                )

                elapsed = time.time() - start_time

                LOGGER.info(
                    "step=%d eval_loss=%.6f eval_perplexity=%.4f elapsed=%.1fs",
                    optimizer_step,
                    metrics["eval_loss"],
                    metrics["eval_perplexity"],
                    elapsed,
                )

                history.append(
                    {
                        "step": optimizer_step,
                        "type": "eval",
                        **metrics,
                        "elapsed_seconds": elapsed,
                    }
                )

            if args.save_steps > 0 and optimizer_step % args.save_steps == 0:
                save_training_state(
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    output_dir=output_dir,
                    step=optimizer_step,
                    args=args,
                    history=history,
                    conversion_metadata=conversion_metadata,
                )

                last_saved_step = optimizer_step

            if optimizer_step >= args.max_steps:
                break

    if last_saved_step != optimizer_step:
        save_training_state(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            output_dir=output_dir,
            step=optimizer_step,
            args=args,
            history=history,
            conversion_metadata=conversion_metadata,
        )
    else:
        LOGGER.info(
            "Final checkpoint already saved at step %d; skipping duplicate save.",
            optimizer_step,
        )

    LOGGER.info("Training complete")


if __name__ == "__main__":
    main()
