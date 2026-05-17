from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

from infrastructure import pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],  # function that takes in observation shape and num actions, and returns a critic network 
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection (default epsilon=0 for deterministic/greedy policy).
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(Section 2.4): get the action from the critic using an epsilon-greedy strategy
        q_values = self.critic(observation)
        if np.random.rand() < epsilon: # with probability epsilon, select a random action
            # randomly sample an action with uniform probability and ensure it's a torch tensor
            action = torch.tensor([np.random.randint(self.num_actions)], device=ptu.device)
        else:
            action = q_values.argmax(dim=-1)  # select the action with the highest Q-value according to the critic

        # ENDTODO

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # TODO(Section 2.4): compute target values
            next_qa_values = self.target_critic(next_obs) # compute the Q-values for the next observations using the target critic


            if self.use_double_q:
                # TODO(Section 2.5): implement double-Q target action selection
                next_action = self.critic(next_obs).argmax(dim=-1)
            else:
                next_action = self.target_critic(next_obs).argmax(dim=-1)

            next_q_values = next_qa_values.gather(dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1) # select the Q-values corresponding to the next actions using the target critic's Q-values
            assert next_q_values.shape == (batch_size,), next_q_values.shape # compute the target values using the reward, discount, and next Q-values, and account for the done signal to zero out the next Q-values if the episode has ended

            target_values = reward + (1 - done.float()) * self.discount * next_q_values
            assert target_values.shape == (batch_size,), target_values.shape
            # ENDTODO

        # TODO(Section 2.4): train the critic with the target values
        qa_values = self.critic(obs) # compute the Q-values for the current observations using the critic
        action_q_values = qa_values.gather(dim=-1, index=action.unsqueeze(-1)).squeeze(-1) # select the Q-values corresponding to the taken actions using the critic's Q-values
        assert action_q_values.shape == (batch_size,), action_q_values.shape
        q_values = action_q_values
        loss = self.critic_loss(action_q_values, target_values)
        # ENDTODO

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(Section 2.4): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done) # update the critic and get the critic stats for logging

        # Hint: if step % self.target_update_period == 0: ...
        if step % self.target_update_period == 0:
            self.update_target_critic() # update the target critic if the current step is a multiple of the target update period
        # ENDTODO

        return critic_stats
