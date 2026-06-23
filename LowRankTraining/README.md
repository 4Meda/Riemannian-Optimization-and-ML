# LowRank Training

This directory contains PyTorch utilities for low-rank training and pretraining of neural networks, including transformer architectures such as BERT and LLaMA/Mistral-style causal language models.

The main idea is to replace selected dense linear layers with low-rank factors:

    W = U Yᵀ

where:

- `U` has shape `(out_features, rank)`;
- `Y` has shape `(in_features, rank)`;
- the represented dense weight is `U @ Y.T`.

For a linear layer, the forward pass is computed as:

    output = input @ Y @ Uᵀ + bias

This can reduce the number of trainable parameters and provides a natural connection with Riemannian low-rank optimization.

## Files

    layers.py
        Defines LowRankLinear, a low-rank replacement for torch.nn.Linear.

    models.py
        Contains full-rank and low-rank fully connected models for MNIST-style experiments.

    transformer_utils.py
        Provides utilities for converting selected transformer linear layers to LowRankLinear.

    hf_profiles.py
        Defines architecture-specific conversion profiles for Hugging Face models.

    convert_hf_model.py
        Command-line utility for converting Hugging Face transformer models to low-rank form.

    causal_lm_pretrain.py
        Causal language-model pretraining script with checkpoint saving, evaluation, and resume support.

## Supported Transformer Families

The current utilities support Hugging Face transformer models whose projections are implemented as explicit `torch.nn.Linear` modules.

### BERT-style models

The BERT profile targets modules such as:

    attention.self.query
    attention.self.key
    attention.self.value
    attention.output.dense
    intermediate.dense
    output.dense

Typical usage:

    python LowRankTraining/convert_hf_model.py \
      --model bert-base-uncased \
      --task masked-lm \
      --profile bert \
      --rank 64

### LLaMA/Mistral-style models

The LLaMA/Mistral profile targets modules such as:

    self_attn.q_proj
    self_attn.k_proj
    self_attn.v_proj
    self_attn.o_proj
    mlp.gate_proj
    mlp.up_proj
    mlp.down_proj

Tiny Mistral-style test conversion:

    python LowRankTraining/convert_hf_model.py \
      --model hf-internal-testing/tiny-random-MistralForCausalLM \
      --task causal-lm \
      --profile mistral \
      --rank 4

## Saving a Low-Rank Checkpoint

A converted low-rank model can be saved with:

    python LowRankTraining/convert_hf_model.py \
      --model hf-internal-testing/tiny-random-MistralForCausalLM \
      --task causal-lm \
      --profile mistral \
      --rank 4 \
      --save-checkpoint LowRankTraining/checkpoints/tiny_mistral_rank4

This creates:

    LowRankTraining/checkpoints/tiny_mistral_rank4/
    ├── lowrank_state_dict.pt
    └── conversion_metadata.json

The saved file is a PyTorch `state_dict`. To reload it, first load the original Hugging Face model, apply the same low-rank conversion, and then load the saved `state_dict`.

## Causal Language-Model Pretraining

The script `causal_lm_pretrain.py` supports low-rank causal language-model pretraining.

Example using a saved tiny Mistral-style low-rank checkpoint:

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
      --max-eval-batches 2 \
      --save-steps 5 \
      --output-dir LowRankTraining/outputs/tiny_mistral_rank4_test

The script will:

1. load the original Hugging Face model;
2. apply the same low-rank conversion;
3. load the saved low-rank checkpoint if provided;
4. load and tokenize the text dataset;
5. train for the requested number of optimizer steps;
6. optionally evaluate on a validation split;
7. save a checkpoint containing model state, optimizer state, tokenizer files, and metadata.

## Resuming Training

To resume from a saved training checkpoint:

    python LowRankTraining/causal_lm_pretrain.py \
      --model hf-internal-testing/tiny-random-MistralForCausalLM \
      --tokenizer hf-internal-testing/llama-tokenizer \
      --profile mistral \
      --rank 4 \
      --resume-from-checkpoint LowRankTraining/outputs/tiny_mistral_rank4_test/checkpoint-step-5 \
      --dataset wikitext \
      --dataset-config wikitext-2-raw-v1 \
      --max-steps 10 \
      --logging-steps 1 \
      --eval-steps 5 \
      --max-eval-batches 2 \
      --save-steps 5 \
      --output-dir LowRankTraining/outputs/tiny_mistral_rank4_resume_test

When resuming, the script restores:

- the low-rank model state;
- the optimizer state, if available;
- the previous optimizer step;
- the training history stored in metadata.

## Checkpoint Contents

Each training checkpoint directory contains files such as:

    lowrank_state_dict.pt
    optimizer_state.pt
    training_metadata.json
    tokenizer_config.json
    tokenizer files

The most important files are:

- `lowrank_state_dict.pt`: low-rank model parameters;
- `optimizer_state.pt`: optimizer state for resuming training;
- `training_metadata.json`: training configuration and logged metrics.

## Requirements

The main dependencies are:

    torch
    transformers
    datasets
    accelerate

Install all repository dependencies with:

    pip install -r requirements.txt

PyTorch installation may depend on your CPU/CUDA setup. If needed, use the official PyTorch installation command from:

    https://pytorch.org/get-started/locally/

## Generated Files

The following files and directories are generated during experiments and should not be committed:

    LowRankTraining/checkpoints/
    LowRankTraining/outputs/
    LowRankTraining/runs/
    LowRankTraining/data/
    *.pt
    *.pth
    *.ckpt

These should be listed in `.gitignore`.

## Notes and Limitations

- The converter replaces explicit `torch.nn.Linear` modules.
- Packed attention implementations, such as some uses of `torch.nn.MultiheadAttention.in_proj_weight`, are not fully factorized by the current converter.
- SVD-based conversion of large models can be computationally expensive.
- Real LLaMA/Mistral-scale models may require GPU memory management, reduced precision, and Hugging Face authentication.
- The current causal language-model pretraining script uses standard `torch.optim.AdamW`. A custom Riemannian optimizer can be added later for low-rank factors.
- The tiny Hugging Face internal models are useful for testing code paths, but their losses and perplexities are not meaningful scientific results.

