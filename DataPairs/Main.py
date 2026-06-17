"""Learning from data pairs using Riemannian optimization.

The experiment considers samples of the form ``(x_i, z_i, y_i)`` and optimizes
a low-rank matrix ``W = U V.T`` so that ``x_i.T @ W @ z_i`` approximates
``y_i``.

Run from the repository root with:

    python DataPairs/Main.py

For a non-interactive run without displaying the plot:

    python DataPairs/Main.py --no-plot
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Add the repository root to the Python path when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Factorizations import FixedRankFactorizations
from optimizers import Optimizer
import Manifold
import utils


DEFAULT_SEED = 44

HISTORY_FILES = {
    "fixed": "FixStpRGDhistories.json",
    "armijo": "ArmijoRGDhistories.json",
    "accelerated": "AccRGDhistories.json",
    "nesterov": "NesRGDhistories.json",
}

# Backward-compatible global manifold object.
manifold = Manifold.SubspaceProjectionmanifold()


def set_seed(seed: int | None) -> None:
    """Set random seeds for reproducible experiments."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)


def pair_prediction(W: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Compute the scalar prediction ``x.T @ W @ z``."""
    return float(x.T @ W @ z)


def pair_loss(W: np.ndarray, x: np.ndarray, z: np.ndarray, y) -> float:
    """Compute the squared loss for one data pair."""
    target = float(np.asarray(y).squeeze())
    residual = pair_prediction(W, x, z) - target

    return residual**2


def pair_loss_derivative(W: np.ndarray, x: np.ndarray, z: np.ndarray, y) -> float:
    """Derivative of the squared loss with respect to the scalar prediction."""
    target = float(np.asarray(y).squeeze())
    residual = pair_prediction(W, x, z) - target

    return 2.0 * residual


def l(W: np.ndarray, x: np.ndarray, z: np.ndarray, y) -> float:
    """Backward-compatible alias for ``pair_loss``."""
    return pair_loss(W, x, z, y)


def lprime(W: np.ndarray, x: np.ndarray, z: np.ndarray, y) -> float:
    """Backward-compatible alias for ``pair_loss_derivative``."""
    return pair_loss_derivative(W, x, z, y)


def data_pair_predictions(problem, W: np.ndarray) -> np.ndarray:
    """Compute predictions for all data pairs.

    For each sample index ``i``, this computes:

    ```text
    prediction_i = X[:, i].T @ W @ Z[:, i]
    ```
    """
    return np.sum(problem.X * (W @ problem.Z), axis=0)


def data_pairs_cost(problem, U: np.ndarray, V: np.ndarray) -> float:
    """Compute the average squared prediction error."""
    W = U @ V.T

    predictions = data_pair_predictions(problem, W)
    residuals = predictions - problem.labels

    return float(np.mean(residuals**2))


def cost(problem, U: np.ndarray, V: np.ndarray) -> float:
    """Backward-compatible alias for ``data_pairs_cost``."""
    return data_pairs_cost(problem, U, V)


class DataPairsDerivatives:
    """Derivative calculations for the data-pair objective."""

    @staticmethod
    def parW(problem, W: np.ndarray) -> np.ndarray:
        """Compute the Euclidean gradient with respect to ``W``."""
        predictions = data_pair_predictions(problem, W)
        residuals = predictions - problem.labels

        coefficients = (2.0 / problem.n) * residuals

        return (problem.X * coefficients) @ problem.Z.T

    @staticmethod
    def parU(problem, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Compute the Euclidean partial derivative with respect to ``U``."""
        W = U @ V.T
        grad_W = DataPairsDerivatives.parW(problem, W)

        return grad_W @ V

    @staticmethod
    def parY(problem, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Compute the Euclidean partial derivative with respect to ``V``.

        The method name ``parY`` is kept for compatibility with ``optimizers.py``.
        In this experiment, the second factor is denoted by ``V``.
        """
        W = U @ V.T
        grad_W = DataPairsDerivatives.parW(problem, W)

        return grad_W.T @ U

    @staticmethod
    def DgradPhiU(problem, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Compute the Stiefel component of the Riemannian gradient."""
        dU = DataPairsDerivatives.parU(problem, U, V)

        return dU - U @ Manifold.sym(U.T @ dU)

    @staticmethod
    def DgradPhiY(problem, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Compute the full-rank factor component of the Riemannian gradient."""
        dV = DataPairsDerivatives.parY(problem, U, V)

        return dV @ V.T @ V


# Backward-compatible name used by optimizers.Optimizer.__init__.
derivatives = DataPairsDerivatives


class DataPairsExperiment(Optimizer):
    """Run the data-pair learning experiment with Riemannian optimizers."""

    def __init__(
        self,
        rank: int,
        d1: int,
        d2: int,
        n: int,
        tol: float,
        max_it: int,
        derivatives_class=DataPairsDerivatives,
        step_size: float = 1e-6,
        seed: int | None = DEFAULT_SEED,
        show_plot: bool = True,
    ):
        super().__init__(derivatives=derivatives_class)

        self.rank = int(rank)
        self.d1 = int(d1)
        self.d2 = int(d2)
        self.n = int(n)
        self.tol = float(tol)
        self.max_it = int(max_it)
        self.stp = float(step_size)
        self.seed = seed

        self._validate_parameters()

        set_seed(seed)

        self.manifold = Manifold.SubspaceProjectionmanifold()

        self._generate_data()
        self._initialise_factors()

        self.results = self.run_optimizers()
        self.histories = self.load_histories()

        if show_plot:
            self.plot_histories()

    def _validate_parameters(self) -> None:
        """Validate experiment dimensions and algorithm parameters."""
        if self.rank <= 0:
            raise ValueError("rank must be positive.")

        if self.rank > min(self.d1, self.d2):
            raise ValueError("rank cannot exceed min(d1, d2).")

        if self.d1 <= 0 or self.d2 <= 0:
            raise ValueError("d1 and d2 must be positive.")

        if self.n <= 0:
            raise ValueError("n must be positive.")

        if self.tol <= 0:
            raise ValueError("tol must be positive.")

        if self.max_it <= 0:
            raise ValueError("max_it must be positive.")

        if self.stp <= 0:
            raise ValueError("step_size must be positive.")

    def _generate_data(self) -> None:
        """Generate random data pairs and labels."""
        self.X = np.random.rand(self.d1, self.n)
        self.Z = np.random.rand(self.d2, self.n)

        self.labels = np.random.rand(self.n)

        # Backward-compatible alias used in the original script.
        self.Y = self.labels

    def _initialise_factors(self) -> None:
        """Create the initial low-rank matrix and factorize it as ``U @ V.T``."""
        self.Win = np.random.rand(self.d1, self.rank) @ np.random.rand(
            self.rank,
            self.d2,
        )

        factorizer = FixedRankFactorizations(self.rank, self.Win.shape)

        self.Uin, self.Yin, self.Sigma = factorizer.subspproj_factorization(self.Win)

    def run_optimizers(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Run all optimization methods used in the experiment."""
        Ufix, Vfix = self.FixedStepRGD(
            data_pairs_cost,
            self.Uin,
            self.Yin,
            self.stp,
        )

        Uarm, Varm = self.ArmijoRuleRGD(
            data_pairs_cost,
            self.Uin,
            self.Yin,
            self.stp,
            0.9,
            1.0 / self.d1,
        )

        Uacc, Vacc = self.AccRGD(
            data_pairs_cost,
            self.Uin,
            self.Yin,
            self.stp,
            np.pi / 4,
        )

        Unes, Vnes = self.Nesterov(
            data_pairs_cost,
            self.Uin,
            self.Yin,
            self.stp,
        )

        return {
            "fixed": (Ufix, Vfix),
            "armijo": (Uarm, Varm),
            "accelerated": (Uacc, Vacc),
            "nesterov": (Unes, Vnes),
        }

    def load_histories(self):
        """Load optimization histories saved by the optimizer methods."""
        histories = {}

        for method_name, filename in HISTORY_FILES.items():
            histories[method_name] = utils.load_histories(filename)

        return histories

    def plot_histories(self) -> None:
        """Plot cost histories for all methods on a semilog scale."""
        fix_it, fix_cost = self.histories["fixed"]
        arm_it, arm_cost = self.histories["armijo"]
        acc_it, acc_cost = self.histories["accelerated"]
        nes_it, nes_cost = self.histories["nesterov"]

        title = (
            f", d1={self.d1}, d2={self.d2}, "
            f"rank={self.rank}, n={self.n}"
        )

        utils.semiplotItCost(
            title,
            (fix_it, fix_cost, "Fixed-step RGD"),
            (arm_it, arm_cost, "Armijo RGD"),
            (acc_it, acc_cost, "Accelerated RGD"),
            (nes_it, nes_cost, "Nesterov"),
        )


# Backward-compatible class name.
main = DataPairsExperiment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the data-pair Riemannian optimization experiment.",
    )

    parser.add_argument("--rank", type=int, default=2, help="Rank of the model matrix.")
    parser.add_argument("--d1", type=int, default=10, help="Dimension of x samples.")
    parser.add_argument("--d2", type=int, default=10, help="Dimension of z samples.")
    parser.add_argument("--n", type=int, default=100, help="Number of data pairs.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Stopping tolerance.")
    parser.add_argument(
        "--max-it",
        type=int,
        default=2**11,
        help="Maximum number of optimizer iterations.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1e-6,
        help="Step size used by fixed-step and accelerated methods.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed. Use a negative value to disable seeding.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the experiment without displaying the plot.",
    )

    return parser.parse_args()


def run_from_cli() -> DataPairsExperiment:
    """Run the experiment from command-line arguments."""
    args = parse_args()

    seed = None if args.seed < 0 else args.seed

    return DataPairsExperiment(
        rank=args.rank,
        d1=args.d1,
        d2=args.d2,
        n=args.n,
        tol=args.tol,
        max_it=args.max_it,
        derivatives_class=DataPairsDerivatives,
        step_size=args.step_size,
        seed=seed,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    run_from_cli()
