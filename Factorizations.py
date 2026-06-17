"""Matrix factorization utilities.

This module contains helper classes for generating fixed-rank matrices and
computing low-rank matrix factorizations using the singular value decomposition.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import svd


Array = np.ndarray
Shape = Tuple[int, int]


class FixedRankMatrix:
    """Utility class for fixed-rank matrix experiments.

    Args:
        rank: Target rank.
        shape: Matrix shape as ``(rows, columns)``.
    """

    def __init__(self, rank: int, shape: Shape):
        self.rank = int(rank)
        self.shape = tuple(shape)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate rank and matrix shape."""
        if len(self.shape) != 2:
            raise ValueError("shape must contain exactly two dimensions.")

        if self.rank <= 0:
            raise ValueError("rank must be positive.")

        if self.rank > min(self.shape):
            raise ValueError("rank cannot exceed min(shape).")

    def random_matrix(
        self,
        shape: Optional[Shape] = None,
        random_state: Optional[int] = None,
    ) -> Array:
        """Generate a random Gaussian matrix.

        Args:
            shape: Optional output shape. If omitted, ``self.shape`` is used.
            random_state: Optional seed for reproducible generation.

        Returns:
            Random matrix with independent standard normal entries.
        """
        output_shape = self.shape if shape is None else tuple(shape)

        if random_state is None:
            return np.random.normal(loc=0.0, scale=1.0, size=output_shape)

        rng = np.random.default_rng(random_state)
        return rng.normal(loc=0.0, scale=1.0, size=output_shape)


class FixedRankFactorizations:
    """Low-rank factorizations based on the singular value decomposition.

    Args:
        rank: Target rank used in truncated factorizations.
        shape: Matrix shape as ``(rows, columns)``.
    """

    def __init__(self, rank: int, shape: Shape):
        self.rank = int(rank)
        self.shape = tuple(shape)

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate rank and matrix shape."""
        if len(self.shape) != 2:
            raise ValueError("shape must contain exactly two dimensions.")

        if self.rank <= 0:
            raise ValueError("rank must be positive.")

        if self.rank > min(self.shape):
            raise ValueError("rank cannot exceed min(shape).")

    def _truncated_svd(self, matrix: Array):
        """Compute the truncated singular value decomposition.

        Args:
            matrix: Input matrix.

        Returns:
            Tuple ``(U_r, s_r, Vt_r, s)`` where ``s`` contains all singular
            values and the other factors are truncated to ``self.rank``.
        """
        U, singular_values, Vt = svd(matrix, full_matrices=False)

        U_r = U[:, : self.rank]
        s_r = singular_values[: self.rank]
        Vt_r = Vt[: self.rank, :]

        return U_r, s_r, Vt_r, singular_values

    def full_rank_factorization(self, matrix: Array):
        """Compute a rank-truncated full-rank factorization.

        For an input matrix with truncated SVD components, this returns factors
        ``G`` and ``H`` such that ``G @ H`` is the rank-``self.rank``
        approximation.

        Args:
            matrix: Input matrix.

        Returns:
            Tuple ``(G, H)`` with shapes ``(rows, rank)`` and
            ``(rank, columns)``.
        """
        U_r, s_r, Vt_r, _ = self._truncated_svd(matrix)

        sqrt_s = np.diag(np.sqrt(s_r))

        G = U_r @ sqrt_s
        H = sqrt_s @ Vt_r

        return G, H

    def subspproj_factorization(self, matrix: Array):
        """Compute the subspace-projection factorization.

        This returns factors ``U`` and ``Y`` such that the rank-truncated matrix
        approximation can be represented as ``U @ Y.T``.

        Args:
            matrix: Input matrix.

        Returns:
            Tuple ``(U, Y, singular_values)`` where ``U`` has shape
            ``(rows, rank)``, ``Y`` has shape ``(columns, rank)``, and
            ``singular_values`` contains all singular values of the input matrix.
        """
        U_r, s_r, Vt_r, singular_values = self._truncated_svd(matrix)

        Y_transpose = np.diag(s_r) @ Vt_r
        Y = Y_transpose.T

        return U_r, Y, singular_values
