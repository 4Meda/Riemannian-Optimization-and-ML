"""Full-rank and low-rank neural network models for MNIST."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .layers import LowRankLinear
except ImportError:  # Allows running this file directly.
    from layers import LowRankLinear


class FullRankNet(nn.Module):
    """Fully connected full-rank network for MNIST.

    Args:
        no_hidden_layers: Number of hidden layers after the input layer.
        hidden_size: Width of the hidden layers.
        input_size: Flattened MNIST input size.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        no_hidden_layers: int = 5,
        hidden_size: int = 512,
        input_size: int = 28 * 28,
        num_classes: int = 10,
    ):
        super().__init__()

        self.no_hidden_layers = int(no_hidden_layers)
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)

        if self.no_hidden_layers < 0:
            raise ValueError("no_hidden_layers must be non-negative.")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)

        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.no_hidden_layers)
            ]
        )

        self.output_layer = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits."""
        x = x.flatten(start_dim=1)

        x = F.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return self.output_layer(x)


class LowRankNet(nn.Module):
    """Low-rank version of ``FullRankNet``.

    The input and output layers are kept dense. Hidden layers from
    ``first_layer_to_low`` onward are replaced by low-rank layers.

    Args:
        full_rank_net: Full-rank network whose architecture is copied.
        low_rank: Rank used for low-rank hidden layers.
        first_layer_to_low: Index of the first hidden layer converted to low rank.
    """

    def __init__(
        self,
        full_rank_net: FullRankNet,
        low_rank: int = 10,
        first_layer_to_low: int = 0,
    ):
        super().__init__()

        self.low_rank = int(low_rank)
        self.first_layer_to_low = int(first_layer_to_low)

        self.input_size = full_rank_net.input_size
        self.hidden_size = full_rank_net.hidden_size
        self.num_classes = full_rank_net.num_classes
        self.no_hidden_layers = len(full_rank_net.hidden_layers)

        if self.low_rank <= 0:
            raise ValueError("low_rank must be positive.")

        if self.low_rank > self.hidden_size:
            raise ValueError("low_rank cannot exceed hidden_size.")

        if not 0 <= self.first_layer_to_low <= self.no_hidden_layers:
            raise ValueError(
                "first_layer_to_low must be between 0 and no_hidden_layers."
            )

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)

        hidden_layers = []

        for layer_index in range(self.no_hidden_layers):
            if layer_index < self.first_layer_to_low:
                hidden_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            else:
                hidden_layers.append(
                    LowRankLinear(
                        self.hidden_size,
                        self.hidden_size,
                        self.low_rank,
                    )
                )

        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits."""
        x = x.flatten(start_dim=1)

        x = F.relu(self.input_layer(x))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return self.output_layer(x)


def convert_to_low_rank(
    full_rank_net: FullRankNet,
    low_rank_net: LowRankNet,
) -> LowRankNet:
    """Copy a full-rank network into a low-rank network.

    Dense layers are copied directly. Low-rank layers are initialized by the
    truncated SVD of the corresponding full-rank hidden layer.

    Args:
        full_rank_net: Source full-rank network.
        low_rank_net: Target low-rank network.

    Returns:
        The updated low-rank network.
    """
    if len(full_rank_net.hidden_layers) != len(low_rank_net.hidden_layers):
        raise ValueError("The two networks have incompatible hidden-layer counts.")

    with torch.no_grad():
        low_rank_net.input_layer.weight.copy_(full_rank_net.input_layer.weight)
        low_rank_net.input_layer.bias.copy_(full_rank_net.input_layer.bias)

        low_rank_net.output_layer.weight.copy_(full_rank_net.output_layer.weight)
        low_rank_net.output_layer.bias.copy_(full_rank_net.output_layer.bias)

        for source_layer, target_layer in zip(
            full_rank_net.hidden_layers,
            low_rank_net.hidden_layers,
        ):
            if isinstance(target_layer, LowRankLinear):
                target_layer.set_from_linear_(source_layer)
            elif isinstance(target_layer, nn.Linear):
                target_layer.weight.copy_(source_layer.weight)
                target_layer.bias.copy_(source_layer.bias)
            else:
                raise TypeError(f"Unsupported target layer type: {type(target_layer)}")

    return low_rank_net


def get_model_parameters(model: nn.Module):
    """Return optimizer parameter groups for low-rank training.

    The first two groups contain low-rank factors and are labelled with roles
    that can be used by a Riemannian optimizer:

    - ``role='U'`` for Stiefel-side factors;
    - ``role='Y'`` for full-rank factors.

    The third group contains all remaining dense weights and biases.
    """
    U_parameters: List[nn.Parameter] = []
    Y_parameters: List[nn.Parameter] = []
    other_parameters: List[nn.Parameter] = []

    lowrank_parameter_ids = set()

    for module in model.modules():
        if isinstance(module, LowRankLinear):
            U_parameters.append(module.U)
            Y_parameters.append(module.Y)

            lowrank_parameter_ids.add(id(module.U))
            lowrank_parameter_ids.add(id(module.Y))

            if module.bias is not None:
                other_parameters.append(module.bias)
                lowrank_parameter_ids.add(id(module.bias))

    for parameter in model.parameters():
        if id(parameter) not in lowrank_parameter_ids:
            other_parameters.append(parameter)

    parameter_groups = []

    if U_parameters:
        parameter_groups.append({"params": U_parameters, "role": "U"})

    if Y_parameters:
        parameter_groups.append({"params": Y_parameters, "role": "Y"})

    if other_parameters:
        parameter_groups.append({"params": other_parameters, "role": None})

    return parameter_groups


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    torch.manual_seed(44)

    full_rank_net = FullRankNet(no_hidden_layers=5, hidden_size=512)

    low_rank_net = LowRankNet(
        full_rank_net,
        low_rank=128,
        first_layer_to_low=2,
    )

    convert_to_low_rank(full_rank_net, low_rank_net)

    x = torch.randn(5, 1, 28, 28)

    logits_full_rank = full_rank_net(x)
    logits_low_rank = low_rank_net(x)

    print("Full-rank logits shape:", logits_full_rank.shape)
    print("Low-rank logits shape:", logits_low_rank.shape)
    print("Full-rank parameters:", count_parameters(full_rank_net))
    print("Low-rank parameters:", count_parameters(low_rank_net))
