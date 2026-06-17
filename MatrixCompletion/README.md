# Matrix Completion Experiment

This directory contains the main experiment script for solving a low-rank matrix completion problem using the Riemannian optimization methods implemented in the repository.

## Problem Description

The goal is to recover or approximate a low-rank matrix from a subset of its observed entries.

The reference matrix is generated as a low-rank product:

```text
W* = A Bᵀ
```

where `A` and `B` are randomly generated Gaussian matrices.

Only a subset of the entries of `W*` is observed. This subset is represented by a binary sampling mask. The optimization problem is then solved over a low-rank factorized representation:

```text
W = U Yᵀ
```

where:

- `U` is a Stiefel-type factor;
- `Y` is a full-rank factor;
- the rank is chosen by the user.

## Main Script

The main script is:

```bash
Main.py
```

It imports the core modules from the repository root:

```text
Factorizations.py
Manifold.py
optimizers.py
utils.py
```

## Optimization Methods

The experiment compares several Riemannian optimization methods:

- fixed-step Riemannian Gradient Descent;
- Armijo line-search Riemannian Gradient Descent;
- accelerated Riemannian Gradient Descent;
- damped Nesterov-type Riemannian Gradient Descent.

## Running the Experiment

From the repository root, run:

```bash
python MatrixCompletion/Main.py
```

To run without displaying the plot window:

```bash
python MatrixCompletion/Main.py --no-plot
```

A custom experiment can be run with command-line arguments, for example:

```bash
python MatrixCompletion/Main.py \
  --rank 5 \
  --d1 100 \
  --d2 600 \
  --n 1000 \
  --tol 1e-15 \
  --max-it 2048 \
  --seed 44 \
  --no-plot
```

## Command-Line Arguments

| Argument | Description | Default |
|---|---|---:|
| `--rank` | Rank of the target matrix | `5` |
| `--d1` | Number of matrix rows | `100` |
| `--d2` | Number of matrix columns | `600` |
| `--n` | Number of observed entries | `1000` |
| `--tol` | Stopping tolerance | `1e-15` |
| `--max-it` | Maximum number of optimizer iterations | `2048` |
| `--seed` | Random seed | `44` |
| `--no-plot` | Disable plotting | off |

Use a negative seed value to disable deterministic seeding:

```bash
python MatrixCompletion/Main.py --seed -1 --no-plot
```

## Output Files

The optimization methods save their iteration and cost histories as JSON files:

```text
FixStpRGDhistories.json
ArmijoRGDhistories.json
AccRGDhistories.json
NesRGDhistories.json
```

Each file stores:

- `it_hist`: iteration numbers;
- `cost_hist`: corresponding cost values.

These histories are loaded at the end of the script and plotted on a semilogarithmic scale.

## Notes

The Nesterov-type method may be more sensitive to the step size than the other methods. In the main experiment, a damped step size is used for this method.
