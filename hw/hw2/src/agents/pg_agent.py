from typing import Optional, Sequence
import numpy as np
import torch

from networks.critics import ValueCritic
from networks.policies import MLPPolicyPG
from infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs = np.concatenate(obs)   # np.concatenate concatenates a sequence of arrays along an existing axis. 
                                    # By default, it concatenates along axis 0, which is what we want here.
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info = {} # This dictionary is for logging any information returned by the actor and critic updates, such as loss values.
        actor_info = self.actor.update(obs, actions, advantages) 
        # The actor's update method should take in the observations, actions, and advantages, and perform a policy gradient update to the actor network. It may also return some information about the update, such as the loss value, which we can store in the `info` dictionary for logging purposes.
        if actor_info is not None:
            info.update(actor_info)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            critic_info = {}
            for _ in range(self.baseline_gradient_steps):
                out = self.critic.update(obs, q_values)
                if out is not None:
                    critic_info.update(out)

            info.update(critic_info)

        return info

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        # `rewards` is a sequence of reward arrays, one per trajectory.
        # For each trajectory, compute the total discounted return and
        # return an array of that value repeated for the trajectory length.
        q_values = []
        for traj_rews in rewards:
            traj_rews = np.array(traj_rews, dtype=np.float32) # convert rewards to numpy array for easier manipulation
            # If the trajectory is empty, return an empty array for that trajectory
            if traj_rews.size == 0: 
                q_values.append(np.array([]))
                continue
            # To compute the discount of each timestep
            discounts = np.array([
                self.gamma ** t for t in range(len(traj_rews))
            ], dtype=np.float32)
            # The discounted return is the sum of discounted rewards for the entire trajectory
            total = np.sum(discounts * traj_rews)
            q_values.append(np.full(len(traj_rews), total, dtype=np.float32))
        return q_values # an array of reward-to-go values for each trajectory

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        # `rewards` is a sequence of reward arrays, one per trajectory.
        # For each trajectory, compute reward-to-go for each timestep.
        q_values = []
        for traj_rews in rewards:
            traj_rews = np.array(traj_rews, dtype=np.float32)
            T = len(traj_rews)
            if T == 0:
                q_values.append(np.array([]))
                continue
            rtg = np.zeros(T, dtype=np.float32)
            for t in range(T):
                discounts = np.array([self.gamma ** k for k in range(T - t)], dtype=np.float32)
                rtg[t] = np.sum(discounts * traj_rews[t:])
            q_values.append(rtg)
        return q_values

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = self._discounted_return(rewards)
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = self._discounted_reward_to_go(rewards)

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            advantages = q_values.copy() # If there is no baseline, the advantage is just the Q-value itself. 
        else:
            # TODO: run the critic and use it as a baseline
            values = self.critic.predict(obs).flatten()
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                # if not using GAE, the advantage is just the Q-value minus the value estimate from the critic.
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    delta = rewards[i] + self.gamma * values[i + 1] * (1 - terminals[i]) - values[i]
                    advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i + 1] * (1 - terminals[i])


                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages
