"""Manifold operations for Riemannian optimization.

This module defines basic operations used by the Riemannian optimization
algorithms implemented in this repository, including tangent-space projections,
horizontal-space projections, retractions, and Riemannian gradients.
"""

import numpy as np
import control as ct

import utils


def skew(A):
    """Return the skew-symmetric part of a matrix.

    Note:
        The sign convention is kept consistent with the original implementation.
    """
    return (A.T - A) / 2


def sym(A):
    """Return the symmetric part of a matrix."""
    return (A.T + A) / 2


class SubspaceProjectionmanifold:
    """Quotient geometry for the Stiefel-times-full-rank decomposition.

    The represented matrix variable is decomposed into factors ``U`` and ``Y``.
    The methods below implement tangent-space projection, horizontal projection,
    retraction, and Riemannian gradient calculations.
    """

    def Psi(self, Zu, Zy, U):
        """Project an ambient vector onto the tangent space.

        Args:
            Zu: Ambient component associated with ``U``.
            Zy: Ambient component associated with ``Y``.
            U: Current Stiefel factor.

        Returns:
            Tuple ``(etaU, etaY)`` representing the tangent vector.
        """
        etaU = Zu - U @ utils.sym(U.T @ Zu)
        etaY = Zy
        return etaU, etaY

    def LyapunovSys(self, etaU, etaY, U, Y):
        """Solve the Lyapunov/Sylvester system for the vertical correction.

        Args:
            etaU: Tangent component associated with ``U``.
            etaY: Tangent component associated with ``Y``.
            U: Current Stiefel factor.
            Y: Current full-rank factor.

        Returns:
            Matrix ``Omega`` used to project onto the horizontal space.
        """
        yty = Y.T @ Y

        A = utils.skew(yty @ (U.T @ etaU) @ yty)
        B = utils.skew((etaY.T @ Y) @ yty)

        omega_tilde = ct.lyap(yty, yty, -2 * A + 2 * B)
        omega = ct.lyap(yty, yty, -omega_tilde)

        return omega

    def PI(self, etaU, etaY, U, Y):
        """Project a tangent vector onto the horizontal space."""
        omega = self.LyapunovSys(etaU, etaY, U, Y)

        horizontal_U = etaU - U @ omega
        horizontal_Y = etaY - Y @ omega

        return horizontal_U, horizontal_Y

    def Retraction(self, step, etaU, etaY, U, Y):
        """Retract a tangent vector back onto the manifold.

        Args:
            step: Step size.
            etaU: Tangent direction associated with ``U``.
            etaY: Tangent direction associated with ``Y``.
            U: Current Stiefel factor.
            Y: Current full-rank factor.

        Returns:
            Updated factors ``(U_new, Y_new)``.
        """
        U_new, _ = np.linalg.qr(U - step * etaU, mode="reduced")
        Y_new = Y - step * etaY

        return U_new, Y_new

    def Grad(self, dU, dY, U, Y):
        """Compute the Riemannian gradient components."""
        grad_U = dU - U @ utils.sym(U.T @ dU)
        grad_Y = dY @ Y.T @ Y

        return grad_U, grad_Y

    def RiemConnection(self, w):
        """Placeholder for a Riemannian connection implementation."""
        return None


class FullRankManifold:
    """Manifold operations for the full-rank factorized representation."""

    def Psi(self, Zu, Zy, U):
        """Project an ambient vector onto the tangent space."""
        etaU = Zu - U @ utils.sym(U.T @ Zu)
        etaY = Zy

        return etaU, etaY

    def LyapunovSys(self, etaU, etaY, U, Y):
        """Solve the Lyapunov/Sylvester system for the vertical correction."""
        yty = Y.T @ Y

        A = utils.skew(yty @ (U.T @ etaU) @ yty)
        B = utils.skew((etaY.T @ Y) @ yty)

        omega_tilde = ct.lyap(yty, yty, 2 * A - 2 * B)
        omega = ct.lyap(yty, yty, omega_tilde)

        return omega

    def PI(self, etaU, etaY, U, Y):
        """Project a tangent vector onto the horizontal space."""
        omega = self.LyapunovSys(etaU, etaY, U, Y)

        horizontal_U = etaU - U @ omega
        horizontal_Y = etaY - Y @ omega

        return horizontal_U, horizontal_Y

    def Retraction(self, step, etaU, etaY, U, Y):
        """Retract a tangent vector back onto the manifold."""
        U_new = utils.uf(U - step * etaU)
        Y_new = Y - step * etaY

        return U_new, Y_new

    def Grad(self, dU, dY, U, Y):
        """Compute the Riemannian gradient components."""
        grad_U = dU - U @ utils.sym(U.T @ dU)
        grad_Y = dY @ Y.T @ Y

        return grad_U, grad_Y

    def RiemConnection(self, w):
        """Placeholder for a Riemannian connection implementation."""
        return None
