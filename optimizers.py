"""Riemannian optimization algorithms.

This module implements the optimization methods used in the experiments,
including fixed-step Riemannian Gradient Descent, Armijo line-search RGD,
accelerated RGD, and a Nesterov-type update.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Tuple

import numpy as np

try:
    from alive_progress import alive_bar
except ImportError:
    @contextmanager
    def alive_bar(total, title="Progress"):
        """Fallback progress bar used when alive_progress is unavailable."""
        def bar():
            return None

        bar.text = ""
        yield bar

import utils


Array = np.ndarray
CostFunction = Callable[[object, Array, Array], float]


class Optimizer:
    """Base optimizer class for Riemannian optimization experiments.

    The concrete experiment class is expected to define attributes such as:

    - ``max_it``: maximum number of iterations;
    - ``tol``: stopping tolerance for the cost function;
    - ``rank``: expected rank of the factor ``Y``.

    By default, derivative routines are imported from ``MatrixCompletion.Main``.
    A custom derivatives object can also be provided.
    """

    def __init__(self, derivatives=None):
        if derivatives is None:
            from MatrixCompletion.Main import derivatives

        self.derivatives = derivatives

    @staticmethod
    def _update_progress(bar, current_cost: float) -> None:
        """Update the progress bar display."""
        bar.text = f"Cost: {current_cost:.2e}"
        bar()

    def _evaluate_cost(self, cost: CostFunction, U: Array, Y: Array) -> float:
        """Evaluate the cost function as a Python float."""
        return float(cost(self, U, Y))

    def _riemannian_gradient(self, U: Array, Y: Array) -> Tuple[Array, Array]:
        """Return the Riemannian gradient components."""
        grad_U = self.derivatives.DgradPhiU(self, U, Y)
        grad_Y = self.derivatives.DgradPhiY(self, U, Y)

        return grad_U, grad_Y

    @staticmethod
    def _take_rgd_step(
        U: Array,
        Y: Array,
        grad_U: Array,
        grad_Y: Array,
        step: float,
    ) -> Tuple[Array, Array]:
        """Take one QR-retracted Riemannian gradient step."""
        U_new, _ = np.linalg.qr(U - step * grad_U, mode="reduced")
        Y_new = Y - step * grad_Y

        return U_new, Y_new

    def _warn_if_rank_drops(self, Y: Array) -> None:
        """Print a warning if ``Y`` does not have the expected rank."""
        expected_rank = getattr(self, "rank", None)

        if expected_rank is None:
            return

        actual_rank = np.linalg.matrix_rank(Y)

        if actual_rank != expected_rank:
            print(f"rank(Y) = {actual_rank}")

    @staticmethod
    def _gradient_norm_squared(grad_U: Array, grad_Y: Array, Y: Array) -> float:
        """Compute the squared norm used in the Armijo condition.

        The metric used here is consistent with the original implementation:

        - Euclidean contribution for the ``U`` component;
        - ``inv(Y.T @ Y)`` weighted contribution for the ``Y`` component.
        """
        gram_Y = Y.T @ Y

        try:
            inv_gram_Y = np.linalg.inv(gram_Y)
        except np.linalg.LinAlgError:
            inv_gram_Y = np.linalg.pinv(gram_Y)

        norm_U = np.trace(grad_U.T @ grad_U)
        norm_Y = np.trace(inv_gram_Y @ grad_Y.T @ grad_Y)

        return float(np.real(norm_U + norm_Y))

    def FixedStepRGD(
        self,
        cost: CostFunction,
        Uin: Array,
        Yin: Array,
        stp: float,
    ) -> Tuple[Array, Array]:
        """Run fixed-step Riemannian Gradient Descent.

        Args:
            cost: Cost function.
            Uin: Initial ``U`` factor.
            Yin: Initial ``Y`` factor.
            stp: Fixed step size.

        Returns:
            Optimized factors ``(U, Y)``.
        """
        U = Uin.copy()
        Y = Yin.copy()

        it = 0
        current_cost = Optimizer._evaluate_cost(self, cost, U, Y)

        it_hist = [it]
        cost_hist = [current_cost]

        with alive_bar(self.max_it, title="Fixed-step RGD") as bar:
            while current_cost > self.tol and it < self.max_it:
                grad_U, grad_Y = Optimizer._riemannian_gradient(self, U, Y)
                U, Y = Optimizer._take_rgd_step(U, Y, grad_U, grad_Y, stp)

                it += 1
                current_cost = Optimizer._evaluate_cost(self, cost, U, Y)

                it_hist.append(it)
                cost_hist.append(current_cost)

                Optimizer._warn_if_rank_drops(self, Y)
                Optimizer._update_progress(bar, current_cost)

        print(f"\nCost: {current_cost:.5e}")

        utils.save_histories(
            it_hist,
            cost_hist,
            filename="FixStpRGDhistories.json",
        )

        return U, Y

    def Armijo_step(
        self,
        cost: CostFunction,
        U: Array,
        Y: Array,
        alpha: float,
        beta: float,
        sigma: float,
        max_search: int = 8,
    ) -> float:
        """Compute an Armijo line-search step size.

        Args:
            cost: Cost function.
            U: Current ``U`` factor.
            Y: Current ``Y`` factor.
            alpha: Initial step-size parameter.
            beta: Backtracking reduction factor, usually in ``(0, 1)``.
            sigma: Sufficient decrease parameter, usually in ``(0, 1)``.
            max_search: Maximum number of backtracking reductions.

        Returns:
            Accepted step size.
        """
        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        if not 0 < beta < 1:
            raise ValueError("beta must satisfy 0 < beta < 1.")

        if not 0 < sigma < 1:
            raise ValueError("sigma must satisfy 0 < sigma < 1.")

        current_cost = Optimizer._evaluate_cost(self, cost, U, Y)
        grad_U, grad_Y = Optimizer._riemannian_gradient(self, U, Y)

        grad_norm_squared = Optimizer._gradient_norm_squared(grad_U, grad_Y, Y)

        if grad_norm_squared <= 0:
            return 0.0

        step = 2 * alpha

        for _ in range(max_search + 1):
            U_candidate, Y_candidate = Optimizer._take_rgd_step(
                U,
                Y,
                grad_U,
                grad_Y,
                step,
            )

            candidate_cost = Optimizer._evaluate_cost(self, cost, U_candidate, Y_candidate)

            sufficient_decrease = (
                candidate_cost <= current_cost - sigma * step * grad_norm_squared
            )

            if sufficient_decrease:
                return step

            step *= beta

        return step

    def ArmijoRuleRGD(
        self,
        cost: CostFunction,
        Uin: Array,
        Yin: Array,
        alpha: float,
        beta: float,
        sigma: float,
    ) -> Tuple[Array, Array]:
        """Run Riemannian Gradient Descent with Armijo backtracking.

        Args:
            cost: Cost function.
            Uin: Initial ``U`` factor.
            Yin: Initial ``Y`` factor.
            alpha: Initial step-size parameter.
            beta: Backtracking reduction factor.
            sigma: Sufficient decrease parameter.

        Returns:
            Optimized factors ``(U, Y)``.
        """
        U = Uin.copy()
        Y = Yin.copy()

        it = 0
        current_cost = Optimizer._evaluate_cost(self, cost, U, Y)

        it_hist = [it]
        cost_hist = [current_cost]

        step = alpha

        with alive_bar(self.max_it, title="Armijo RGD") as bar:
            while current_cost > self.tol and it < self.max_it:
                step = Optimizer.Armijo_step(self, cost, U, Y, alpha, beta, sigma)

                if step == 0:
                    print("Armijo step is zero. Stopping early.")
                    break

                grad_U, grad_Y = Optimizer._riemannian_gradient(self, U, Y)
                U, Y = Optimizer._take_rgd_step(U, Y, grad_U, grad_Y, step)

                it += 1
                current_cost = Optimizer._evaluate_cost(self, cost, U, Y)

                it_hist.append(it)
                cost_hist.append(current_cost)

                Optimizer._warn_if_rank_drops(self, Y)
                Optimizer._update_progress(bar, current_cost)

        print(f"\nCost: {current_cost:.5e}")
        print(f"step = {step}")

        utils.save_histories(
            it_hist,
            cost_hist,
            filename="ArmijoRGDhistories.json",
        )

        return U, Y

    def AccRGD(
        self,
        cost: CostFunction,
        Uin: Array,
        Yin: Array,
        stp: float,
        beta: float,
    ) -> Tuple[Array, Array]:
        """Run accelerated Riemannian Gradient Descent with momentum.

        Args:
            cost: Cost function.
            Uin: Initial ``U`` factor.
            Yin: Initial ``Y`` factor.
            stp: Step size.
            beta: Momentum parameter.

        Returns:
            Optimized factors ``(U, Y)``.
        """
        U = Uin.copy()
        Y = Yin.copy()

        velocity_U = np.zeros_like(U)
        velocity_Y = np.zeros_like(Y)

        it = 0
        current_cost = Optimizer._evaluate_cost(self, cost, U, Y)

        it_hist = [it]
        cost_hist = [current_cost]

        with alive_bar(self.max_it, title="Accelerated RGD") as bar:
            while current_cost > self.tol and it < self.max_it:
                grad_U, grad_Y = Optimizer._riemannian_gradient(self, U, Y)

                velocity_U = beta * velocity_U - stp * grad_U
                velocity_Y = beta * velocity_Y - stp * grad_Y

                U, _ = np.linalg.qr(U + velocity_U, mode="reduced")
                Y = Y + velocity_Y

                it += 1
                current_cost = Optimizer._evaluate_cost(self, cost, U, Y)

                it_hist.append(it)
                cost_hist.append(current_cost)

                Optimizer._warn_if_rank_drops(self, Y)
                Optimizer._update_progress(bar, current_cost)

        print(f"\nCost: {current_cost:.5e}")

        utils.save_histories(
            it_hist,
            cost_hist,
            filename="AccRGDhistories.json",
        )

        return U, Y

    def Nesterov(
        self,
        cost: CostFunction,
        Uin: Array,
        Yin: Array,
        stp: float,
    ) -> Tuple[Array, Array]:
        """Run a Nesterov-type Riemannian gradient method.

        Args:
            cost: Cost function.
            Uin: Initial ``U`` factor.
            Yin: Initial ``Y`` factor.
            stp: Step size.

        Returns:
            Optimized factors ``(U, Y)``.
        """
        U = Uin.copy()
        Y = Yin.copy()

        lookahead_U = U.copy()
        lookahead_Y = Y.copy()

        it = 0
        current_cost = Optimizer._evaluate_cost(self, cost, U, Y)

        it_hist = [it]
        cost_hist = [current_cost]

        with alive_bar(self.max_it, title="Nesterov RGD") as bar:
            while current_cost > self.tol and it < self.max_it:
                grad_U, grad_Y = Optimizer._riemannian_gradient(self, lookahead_U, lookahead_Y)

                U_new, Y_new = Optimizer._take_rgd_step(
                    lookahead_U,
                    lookahead_Y,
                    grad_U,
                    grad_Y,
                    stp,
                )

                it += 1
                momentum = (it - 1) / (it + 2)

                lookahead_U = U_new + momentum * (U_new - U)
                lookahead_Y = Y_new + momentum * (Y_new - Y)

                U, Y = U_new, Y_new

                current_cost = Optimizer._evaluate_cost(self, cost, U, Y)

                it_hist.append(it)
                cost_hist.append(current_cost)

                Optimizer._warn_if_rank_drops(self, Y)
                Optimizer._update_progress(bar, current_cost)

        print(f"\nCost: {current_cost:.5e}")

        utils.save_histories(
            it_hist,
            cost_hist,
            filename="NesRGDhistories.json",
        )

        return U, Y
