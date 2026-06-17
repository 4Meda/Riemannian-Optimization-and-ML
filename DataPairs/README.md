# Data Pairs Experiment

This directory contains an experiment for learning a low-rank matrix from data pairs using Riemannian optimization.

## Problem Description

The experiment considers training samples of the form:

    (x_i, z_i, y_i)

The goal is to learn a low-rank matrix:

    W = U Vᵀ

such that:

    x_iᵀ W z_i ≈ y_i

for all observed data pairs. The optimization problem becomes:

$$
\min_{U, V} f(U, V) =\frac{1}{n}\sum_{i=1}^{n}\left(x_i^\top U V^\top z_i - y_i\right)^2
$$

The optimization is performed over the factorized representation `W = U Vᵀ`, where:

- `U` is a Stiefel-type factor;
- `V` is a full-rank factor;
- the rank is chosen by the user.

## Main Script

The main script is:

    Main.py

It imports the core modules from the repository root:

    Factorizations.py
    Manifold.py
    optimizers.py
    utils.py

## Optimization Methods

The experiment compares several Riemannian optimization methods:

- fixed-step Riemannian Gradient Descent;
- Armijo line-search Riemannian Gradient Descent;
- accelerated Riemannian Gradient Descent;
- Nesterov-type Riemannian Gradient Descent.

## Running the Experiment

From the repository root, run:

    python DataPairs/Main.py

To run without displaying the plot window:

    python DataPairs/Main.py --no-plot

A custom experiment can be run with command-line arguments, for example:

    python DataPairs/Main.py \
      --rank 2 \
      --d1 10 \
      --d2 10 \
      --n 100 \
      --tol 1e-3 \
      --max-it 2048 \
      --step-size 1e-6 \
      --seed 44 \
      --no-plot

## Command-Line Arguments

| Argument | Description | Default |
|---|---|---:|
| `--rank` | Rank of the model matrix | `2` |
| `--d1` | Dimension of the `x` samples | `10` |
| `--d2` | Dimension of the `z` samples | `10` |
| `--n` | Number of data pairs | `100` |
| `--tol` | Stopping tolerance | `1e-3` |
| `--max-it` | Maximum number of optimizer iterations | `2048` |
| `--step-size` | Step size used by the optimization methods | `1e-6` |
| `--seed` | Random seed | `44` |
| `--no-plot` | Disable plotting | off |

Use a negative seed value to disable deterministic seeding:

    python DataPairs/Main.py --seed -1 --no-plot

## Output Files

The optimization methods save their iteration and cost histories as JSON files:

    FixStpRGDhistories.json
    ArmijoRGDhistories.json
    AccRGDhistories.json
    NesRGDhistories.json

Each file stores:

- `it_hist`: iteration numbers;
- `cost_hist`: corresponding cost values.

These histories are loaded at the end of the script and plotted on a semilogarithmic scale.

## Notes

This experiment is intended for numerical testing of Riemannian optimization methods on a low-rank learning problem.
