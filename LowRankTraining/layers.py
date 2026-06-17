"""Low-rank PyTorch layers."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class LowRankLinear(nn.Module):
    """Low-rank replacement for ``torch.nn.Linear``.

    The dense weight matrix is represented as

        W = U Y.T

    where ``U`` has shape ``(out_features, rank)`` and ``Y`` has shape
    ``(in_features, rank)``.

    The forward pass computes

        output = input @ Y @ U.T + bias

    which is equivalent to using the dense weight ``W`` in a standard linear
    layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")

        if self.rank <= 0:
            raise ValueError("rank must be positive.")

        if self.rank > min(self.in_features, self.out_features):
            raise ValueError("rank cannot exceed min(in_features, out_features).")

        factory_kwargs = {"device": device, "dtype": dtype}

        self.U = nn.Parameter(
            torch.empty(self.out_features, self.rank, **factory_kwargs)
        )
        self.Y = nn.Parameter(
            torch.empty(self.in_features, self.rank, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        """Return the materialized dense weight matrix ``W = U @ Y.T``."""
        return self.U @ self.Y.T

    def reset_parameters(self) -> None:
        """Initialize the factors from a randomly initialized dense layer."""
        dense_weight = torch.empty(
            self.out_features,
            self.in_features,
            device=self.U.device,
            dtype=self.U.dtype,
        )

        nn.init.kaiming_uniform_(dense_weight, a=math.sqrt(5))

        with torch.no_grad():
            self.set_from_dense_weight_(dense_weight)

            if self.bias is not None:
                bound = 1 / math.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def set_from_dense_weight_(self, weight: torch.Tensor) -> "LowRankLinear":
        """Set the low-rank factors from a dense weight matrix using SVD."""
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                "weight has incompatible shape: "
                f"expected {(self.out_features, self.in_features)}, "
                f"got {tuple(weight.shape)}"
            )

        svd_weight = weight.detach().to(device=self.U.device)

        original_dtype = svd_weight.dtype

        if svd_weight.dtype in (torch.float16, torch.bfloat16):
            svd_weight = svd_weight.float()

        U_svd, singular_values, Vh = torch.linalg.svd(
            svd_weight,
            full_matrices=False,
        )

        U_factor = U_svd[:, : self.rank]
        Y_factor = Vh[: self.rank, :].T * singular_values[: self.rank]

        self.U.copy_(U_factor.to(dtype=self.U.dtype))

        if self.Y.dtype != original_dtype:
            Y_factor = Y_factor.to(dtype=self.Y.dtype)

        self.Y.copy_(Y_factor)

        return self

    @torch.no_grad()
    def set_from_linear_(self, linear: nn.Linear) -> "LowRankLinear":
        """Set the low-rank layer from an existing dense ``nn.Linear`` layer."""
        self.set_from_dense_weight_(linear.weight)

        if self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias)

        return self

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> "LowRankLinear":
        """Create a low-rank layer from an existing dense ``nn.Linear`` layer."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        layer.set_from_linear_(linear)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the low-rank linear transformation."""
        hidden = x.matmul(self.Y)
        return F.linear(hidden, self.U, self.bias)

    @torch.no_grad()
    def orthogonalize_(self, preserve_weight: bool = True) -> "LowRankLinear":
        """Orthogonalize ``U`` using QR retraction.

        If ``preserve_weight=True``, ``Y`` is updated so that ``U @ Y.T`` is
        preserved.
        """
        Q, R = torch.linalg.qr(self.U, mode="reduced")

        self.U.copy_(Q)

        if preserve_weight:
            self.Y.copy_(self.Y @ R.T)

        return self

    def to_linear(self) -> nn.Linear:
        """Convert this low-rank layer back to a dense ``nn.Linear`` layer."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.U.device,
            dtype=self.U.dtype,
        )

        with torch.no_grad():
            linear.weight.copy_(self.weight)

            if self.bias is not None:
                linear.bias.copy_(self.bias)

        return linear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"bias={self.bias is not None}"
        )


# Backward-compatible name with your previous implementation.
LowRankLayer = LowRankLinear
