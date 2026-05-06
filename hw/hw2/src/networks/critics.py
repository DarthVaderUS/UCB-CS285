import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            learning_rate,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass of the critic network
        value = self.network(obs).squeeze() # compute the value for obs and remove extra dimensions
        return value

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Evaluate the critic on numpy observations and return numpy values."""
        self.eval() # set the network to evaluation mode
        with torch.no_grad(): # disable gradient computation for inference
            obs_t = ptu.from_numpy(obs) # convert the input observations from numpy to torch tensors
            vals_t = self.forward(obs_t) # compute the predicted values for the input observations
        return ptu.to_numpy(vals_t) # convert the predicted values back to numpy

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        # TODO: compute the loss using the observations and q_values
        loss = F.mse_loss(self.forward(obs), q_values) # MSE loss between predicted values and target q_values

        # TODO: perform an optimizer step
        self.optimizer.zero_grad() # zero the gradients before backward pass
        loss.backward() # compute gradients
        self.optimizer.step() # update parameters

        return {
            "Baseline Loss": loss.item(),
        }