"""Matrix completion experiment using Riemannian optimization.

Run from the repository root with:

    python MatrixCompletion/Main.py

For a non-interactive run without displaying the plot:

    python MatrixCompletion/Main.py --no-plot
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

from Factorizations import FixedRankFactorizations, FixedRankMatrix
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


def create_cai_matrix(rank: int, d1: int, d2: int) -> np.ndarray:
    """Create a random rank-``rank`` matrix as in Cai et al. style examples.

    The matrix is generated as ``A @ B.T``, where ``A`` and ``B`` are Gaussian
    random matrices of shapes ``(d1, rank)`` and ``(d2, rank)``.
    """
    matrix_A = FixedRankMatrix(rank, (d1, rank))
    matrix_B = FixedRankMatrix(rank, (d2, rank))

    A = matrix_A.random_matrix()
    B = matrix_B.random_matrix()

    return A @ B.T


def create_Cai_matrix(rank: int, d1: int, d2: int) -> np.ndarray:
    """Backward-compatible alias for ``create_cai_matrix``."""
    return create_cai_matrix(rank, d1, d2)


def create_mask_matrix(d1: int, d2: int, n: int) -> np.ndarray:
    """Create a binary mask with ``n`` randomly sampled observed entries."""
    total_entries = d1 * d2

    if n <= 0:
        raise ValueError("n must be positive.")

    if n > total_entries:
        raise ValueError("n cannot exceed d1 * d2.")

    mask = np.zeros((d1, d2), dtype=float)

    indices = np.random.choice(total_entries, n, replace=False)
    rows, cols = np.unravel_index(indices, (d1, d2))

    mask[rows, cols] = 1.0

    return mask


def orthogonal_sampling(mask: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply the sampling operator by masking the entries of ``W``."""
    return mask * W


def OrtSampling(mask: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for ``orthogonal_sampling``."""
    return orthogonal_sampling(mask, W)


def matrix_completion_cost(problem, U: np.ndarray, Y: np.ndarray) -> float:
    """Compute the normalized matrix completion cost."""
    sampled_prediction = orthogonal_sampling(problem.mask, U @ Y.T)
    sampled_reference = orthogonal_sampling(problem.mask, problem.Ws)

    residual = sampled_prediction - sampled_reference

    return float(np.linalg.norm(residual, ord="fro") ** 2 / problem.n)


def cost(problem, U: np.ndarray, Y: np.ndarray) -> float:
    """Backward-compatible alias for ``matrix_completion_cost``."""
    return matrix_completion_cost(problem, U, Y)


class MatrixCompletionDerivatives:
    """Derivative calculations for the matrix completion objective."""

    @staticmethod
    def parU(problem, U: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the partial derivative with respect to ``U``."""
        residual = orthogonal_sampling(problem.mask, U @ Y.T) - orthogonal_sampling(
            problem.mask,
            problem.Ws,
        )

        S = (2.0 / problem.n) * residual

        return S @ Y

    @staticmethod
    def parY(problem, U: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the partial derivative with respect to ``Y``."""
        residual = orthogonal_sampling(problem.mask, U @ Y.T) - orthogonal_sampling(
            problem.mask,
            problem.Ws,
        )

        S = (2.0 / problem.n) * residual

        return S.T @ U

    @staticmethod
    def DgradPhiU(problem, U: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the Stiefel component of the Riemannian gradient."""
        dU = MatrixCompletionDerivatives.parU(problem, U, Y)

        return dU - U @ Manifold.sym(U.T @ dU)

    @staticmethod
    def DgradPhiY(problem, U: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the full-rank factor component of the Riemannian gradient."""
        dY = MatrixCompletionDerivatives.parY(problem, U, Y)

        return dY @ Y.T @ Y


# Backward-compatible name used by optimizers.Optimizer.__init__.
derivatives = MatrixCompletionDerivatives


class MatrixCompletionExperiment(Optimizer):
    """Run matrix completion experiments with several Riemannian optimizers."""

    def __init__(
        self,
        rank: int,
        d1: int,
        d2: int,
        n: int,
        tol: float,
        max_it: int,
        derivatives_class=MatrixCompletionDerivatives,
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
        self.seed = seed

        self._validate_parameters()

        set_seed(seed)

        self.manifold = Manifold.SubspaceProjectionmanifold()

        self.Ws = create_cai_matrix(self.rank, self.d1, self.d2)
        self.mask = create_mask_matrix(self.d1, self.d2, self.n)

        self._initialise_factors()
        self._set_step_size()

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

        if self.n > self.d1 * self.d2:
            raise ValueError("n cannot exceed d1 * d2.")

        if self.tol <= 0:
            raise ValueError("tol must be positive.")

        if self.max_it <= 0:
            raise ValueError("max_it must be positive.")

    def _initialise_factors(self) -> None:
        """Create the initial low-rank matrix and factorize it as ``U @ Y.T``."""
        self.Win = np.random.rand(self.d1, self.rank) @ np.random.rand(
            self.rank,
            self.d2,
        )

        factorizer = FixedRankFactorizations(self.rank, self.Win.shape)

        self.Uin, self.Yin, self.Sigma = factorizer.subspproj_factorization(self.Win)

    def _set_step_size(self) -> None:
        """Set the baseline step size from the retained singular values."""
        retained_singular_values = np.asarray(self.Sigma[: self.rank], dtype=float)

        L = float(np.max(retained_singular_values))
        mu = float(np.min(retained_singular_values))

        if L + mu <= 0:
            raise ValueError("Cannot compute a positive step size from singular values.")

        self.stp = 1.0 / (L + mu)

    def run_optimizers(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Run all optimization methods used in the experiment."""
        Ufix, Yfix = self.FixedStepRGD(
            matrix_completion_cost,
            self.Uin,
            self.Yin,
            self.stp,
        )

        Uarm, Yarm = self.ArmijoRuleRGD(
            matrix_completion_cost,
            self.Uin,
            self.Yin,
            self.stp,
            0.9,
            1.0 / self.d1,
        )

        Uacc, Yacc = self.AccRGD(
            matrix_completion_cost,
            self.Uin,
            self.Yin,
            self.stp,
            np.pi / 4,
        )

        Unes, Ynes = self.Nesterov(
            matrix_completion_cost,
            self.Uin,
            self.Yin,
            self.stp / 10,
        )

        return {
            "fixed": (Ufix, Yfix),
            "armijo": (Uarm, Yarm),
            "accelerated": (Uacc, Yacc),
            "nesterov": (Unes, Ynes),
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
            (nes_it, nes_cost, "Nesterov damped"),
        )


# Backward-compatible class name.
main = MatrixCompletionExperiment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the matrix completion Riemannian optimization experiment.",
    )

    parser.add_argument("--rank", type=int, default=5, help="Rank of the target matrix.")
    parser.add_argument("--d1", type=int, default=100, help="Number of matrix rows.")
    parser.add_argument("--d2", type=int, default=600, help="Number of matrix columns.")
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of observed matrix entries.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-15,
        help="Stopping tolerance.",
    )
    parser.add_argument(
        "--max-it",
        type=int,
        default=2**11,
        help="Maximum number of optimizer iterations.",
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


def run_from_cli() -> MatrixCompletionExperiment:
    """Run the experiment from command-line arguments."""
    args = parse_args()

    seed = None if args.seed < 0 else args.seed

    return MatrixCompletionExperiment(
        rank=args.rank,
        d1=args.d1,
        d2=args.d2,
        n=args.n,
        tol=args.tol,
        max_it=args.max_it,
        derivatives_class=MatrixCompletionDerivatives,
        seed=seed,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    run_from_cli()
