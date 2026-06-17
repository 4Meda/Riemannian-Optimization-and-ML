# Riemannian Optimization for Machine Learning

This repository collects research code developed during my PhD on Riemannian optimization and low-rank methods for machine learning. It includes Python modules for optimization on matrix manifolds, experiments on matrix completion and data-pair learning, and PyTorch utilities for low-rank neural network and transformer pretraining.

## Overview

The project provides tools for:

- working with matrix factorizations;
- defining tangent-space operations on matrix manifolds;
- applying retraction operators;
- implementing Riemannian gradient-based optimization methods;
- running numerical experiments for machine learning problems.

## Repository Structure

```text
.
├── Factorizations.py
├── Manifold.py
├── optimizers.py
├── utils.py
├── DataPairs
│   └── Main.py
├── MatrixCompletion
│   ├── Main.py
│   └── README.md
├── AccRGDhistories.json
├── ArmijoRGDhistories.json
├── FixStpRGDhistories.json
├── NesRGDhistories.json
└── README.md
```

## Main Modules

### `Factorizations.py`

Contains functionality related to matrix factorizations and low-rank matrix representations.

This includes routines such as:

- singular value decomposition based operations;
- projections associated with low-rank matrix structures;
- helper functions for factorized matrix variables.

### `Manifold.py`

Contains tools for defining and working with matrix manifolds.

This includes functionality such as:

- tangent-space operations;
- projection of ambient gradients onto tangent spaces;
- retraction operators;
- manifold-related helper routines.

### `optimizers.py`

Implements Riemannian optimization algorithms, including:

- fixed-step Riemannian Gradient Descent;
- Armijo line-search Riemannian Gradient Descent;
- accelerated Riemannian Gradient Descent with momentum;
- Nesterov-type Riemannian updates.

### `utils.py`

Contains general utility functions used by the project. These functions are not specific to the mathematical definition of the manifold, but support implementation, experimentation, and data handling.

## Example Applications

### Matrix Completion

The `MatrixCompletion/` directory contains scripts and outputs related to matrix completion experiments.

To run the matrix completion example from the repository root:

```bash
python MatrixCompletion/Main.py
```

Alternatively:

```bash
cd MatrixCompletion
python Main.py
```

### Data Pairs

The `DataPairs/` directory contains an additional experiment script.

To run it from the repository root:

```bash
python DataPairs/Main.py
```

Alternatively:

```bash
cd DataPairs
python Main.py
```

## Optimisation History Files

The JSON files store optimisation histories produced by different methods:

```text
AccRGDhistories.json
ArmijoRGDhistories.json
FixStpRGDhistories.json
NesRGDhistories.json
```

These files may contain information such as objective values, gradient norms, iteration counts, or convergence data generated during experiments.

## Requirements

The project is written in Python and uses standard scientific computing libraries.

Typical dependencies include:

- NumPy
- SciPy
- Matplotlib

If these packages are not already installed, they can be installed with:

```bash
pip install numpy scipy matplotlib
```

## Usage

From the repository root, run one of the example scripts:

```bash
python MatrixCompletion/Main.py
```

or:

```bash
python DataPairs/Main.py
```

The main implementation files can also be imported into other Python scripts for custom Riemannian optimization experiments.

## Notes

This repository is intended for research and experimentation with Riemannian optimization methods for machine learning. The code is organized so that manifold operations, factorization routines, optimization algorithms, and experiment scripts are separated into different modules.

## LowRank Training

The `LowRankTraining/` directory contains PyTorch utilities for low-rank neural network training and transformer pretraining.

It includes:

- a `LowRankLinear` layer based on the factorization `W = U Yᵀ`;
- utilities for converting selected Hugging Face transformer layers to low-rank form;
- BERT and LLaMA/Mistral conversion profiles;
- checkpoint saving and loading utilities;
- a causal language-model pretraining script with evaluation and resume support.

Example tiny Mistral-style conversion:

    python LowRankTraining/convert_hf_model.py \
      --model hf-internal-testing/tiny-random-MistralForCausalLM \
      --task causal-lm \
      --profile mistral \
      --rank 4

Example low-rank causal LM smoke test:

    python LowRankTraining/causal_lm_pretrain.py \
      --model hf-internal-testing/tiny-random-MistralForCausalLM \
      --tokenizer hf-internal-testing/llama-tokenizer \
      --profile mistral \
      --rank 4 \
      --dataset wikitext \
      --dataset-config wikitext-2-raw-v1 \
      --max-steps 5
