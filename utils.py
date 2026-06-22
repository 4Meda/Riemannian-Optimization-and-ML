"""Utility functions for Riemannian optimization experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

try:
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None
    tty = None


def sym(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a matrix."""
    return (A.T + A) / 2


def skew(A: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric part of a matrix.

    The sign convention is kept consistent with the original implementation.
    """
    return (A.T - A) / 2


def uf(A: np.ndarray) -> np.ndarray:
    """Return the orthogonal polar factor of a full-column-rank matrix.

    For a matrix ``A`` with singular value decomposition ``A = U S V.T``, this
    returns ``U V.T``. The result has orthonormal columns and is commonly used
    as a retraction onto the Stiefel manifold.
    """
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    return U @ Vt


def _to_json_compatible(value: Any) -> Any:
    """Convert NumPy objects to JSON-compatible Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(key): _to_json_compatible(val) for key, val in value.items()}

    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]

    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]

    return value


def save_histories(it_hist, cost_hist, filename: str | Path) -> None:
    """Save iteration and cost histories to a JSON file.

    Args:
        it_hist: Iteration history.
        cost_hist: Cost-function history.
        filename: Output JSON filename.
    """
    path = Path(filename)

    data = {
        "it_hist": _to_json_compatible(it_hist),
        "cost_hist": _to_json_compatible(cost_hist),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_histories(filename: str | Path):
    """Load iteration and cost histories from a JSON file.

    Args:
        filename: Input JSON filename.

    Returns:
        Tuple ``(it_hist, cost_hist)``.
    """
    path = Path(filename)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["it_hist"], data["cost_hist"]


def wait_for_spacebar() -> None:
    """Pause execution until the user presses the spacebar.

    On systems where raw terminal input is unavailable, this falls back to a
    standard Enter-key prompt.
    """
    print("Press Spacebar to continue...")

    if termios is None or tty is None or not sys.stdin.isatty():
        input("Press Enter to continue...")
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while True:
            ch = sys.stdin.read(1)

            if ch == " ":
                break

            if ch in ("\x03", "\x04"):
                raise KeyboardInterrupt

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def semilog_plot_it_cost(txt: str, *args, show: bool = True, save_path: str | Path | None = None):
    """Plot one or more cost histories with a logarithmic y-axis.

    Args:
        txt: Title text for the plot.
        *args: Triples of ``(x, y, label)``.
        show: Whether to display the plot immediately.
        save_path: Optional path where the figure should be saved.

    Returns:
        Tuple ``(fig, ax)``.
    """
    fig, ax = plt.subplots()

    ax.set_title(f"Semilog plot {txt}")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_yscale("log")

    for x, y, label in args:
        ax.plot(x, y, "-", label=str(label))

    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def loglog_plot_it_cost(txt: str, *args, show: bool = True, save_path: str | Path | None = None):
    """Plot one or more cost histories with logarithmic x- and y-axes.

    Args:
        txt: Title text for the plot.
        *args: Triples of ``(x, y, label)``.
        show: Whether to display the plot immediately.
        save_path: Optional path where the figure should be saved.

    Returns:
        Tuple ``(fig, ax)``.
    """
    fig, ax = plt.subplots()

    ax.set_title(f"Log-log plot {txt}")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cost", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for x, y, label in args:
        ax.plot(x, y, "-", label=str(label))

    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

