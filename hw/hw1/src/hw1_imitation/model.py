"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

"""Train and evaluate a Push-T imitation policy."""
def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    curr_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(curr_dim, hidden_dim))
        layers.append(nn.ReLU())
        curr_dim = hidden_dim
    layers.append(nn.Linear(curr_dim, output_dim))
    return nn.Sequential(*layers)


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        output_dim = chunk_size * action_dim    # chunk_size is the number of future steps, action_dim is the dimension of each action
        self.net = build_mlp(state_dim, output_dim, hidden_dims)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_chunk = self.sample_actions(state)  # This sample comes from the forward pass of the network
        return F.mse_loss(pred_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        pred = self.net(state)  # shape (batch_size, chunk_size * action_dim)
        return pred.view(state.shape[0], self.chunk_size, self.action_dim)  # reshape the output to (batch_size, chunk_size, action_dim)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.flat_action_dim = chunk_size * action_dim
        input_dim = state_dim + self.flat_action_dim + 1
        self.net = build_mlp(input_dim, self.flat_action_dim, hidden_dims)

    def _predict_velocity(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        flat_action_chunk = action_chunk.reshape(batch_size, self.flat_action_dim)
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)
        net_input = torch.cat([state, flat_action_chunk, tau], dim=-1)
        velocity = self.net(net_input)
        return velocity.view(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(
            batch_size,
            1,
            1,
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )
        interp_chunk = tau * action_chunk + (1.0 - tau) * noise
        target_velocity = action_chunk - noise
        pred_velocity = self._predict_velocity(state, interp_chunk, tau.view(batch_size, 1))
        return F.mse_loss(pred_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive for flow sampling")

        batch_size = state.shape[0]
        action_chunk = torch.randn(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        dt = 1.0 / num_steps
        for step_idx in range(num_steps):
            # tau_val is the current step index normalized by the total number of steps, giving a value between 0 and 1 that represents the progress through the flow sampling process
            tau_val = step_idx / num_steps
            # tau is the normalized time for the current step, ranging from 0 to 1
            tau = torch.full(
                (batch_size, 1),
                tau_val,
                device=state.device,
                dtype=state.dtype,
            )
            velocity = self._predict_velocity(state, action_chunk, tau)
            action_chunk = action_chunk + dt * velocity
        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
