# Q&A: CartPole experiment analysis

Q: Which value estimator has better performance without advantage normalization: the trajectory-centric one, or the one using reward-to-go?

A: In these runs the trajectory-centric estimator (no reward-to-go) performed slightly better — the `cartpole` run reached higher average returns than `cartpole_rtg` for the same seed.

Q: Between the two value estimators, why do you think one is generally preferred over the other?

A: Reward-to-go is generally preferred because it reduces variance by summing only future rewards for each timestep, improving credit assignment and producing lower-variance gradient estimates (especially for longer episodes). Trajectory-centric can still work for short-horizon tasks like CartPole, which may explain the similar or slightly better results here.

Q: Did advantage normalization help?

A: Yes — advantage normalization improved stability and often sped up convergence. Normalized runs (e.g., `*_na`) reached high performance more reliably than corresponding non-normalized runs.

Q: Did the batch size make an impact?

A: Both small-batch and large-batch experiments reached near-optimal performance here. Large-batch (`*_lb`) runs have fewer, larger updates (smoother learning per update) while small-batch runs update more frequently and can be noisier; final performance differences were minor in these trials.

## 4.2 HalfCheetah

Q: How did reducing the number of baseline gradient steps affect the baseline curve and policy performance?

A: Reducing `-bgs` from `5` to `1` made the baseline much less effective. The baseline loss still moved, but the policy performance degraded sharply and the final eval return fell to about `-12.39` instead of staying strongly positive. In this case, the critic was no longer accurate enough to provide a useful variance-reduction signal.

## 6 Hyperparameters and Sample Efficiency

Q: What was the best InvertedPendulum configuration?

A: The best local run was:

```bash
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 500 -eb 500 -rtg --discount 0.99 --use_baseline -blr 0.01 -bgs 5 -na --gae_lambda 0.95 -lr 0.01 --exp_name pendulum_tuned_b500 --seed 1
```

It reached an average return of `1000` within about `81.5K` environment steps, which satisfies the assignment target.

Q: Which hyperparameters mattered most?

A: Batch size mattered the most because the assignment measures sample efficiency in environment steps. Dropping the batch size from `5000` to `500` let the policy improve much sooner in terms of steps. Advantage normalization and the baseline were also important for stability, and `reward-to-go` plus `GAE` helped keep the gradient signal usable.

Q: How did the default Pendulum setting compare to the tuned one?

A: The default setting only reached an eval return around `116` after roughly `500K` steps, while the tuned configuration hit `1000` well before `100K` steps. The tuned run was substantially more sample efficient.

Q: What exact command did you use for the default comparison run?

A: I used:

```bash
uv run src/scripts/run.py --env_name InvertedPendulum-v4 -n 100 -b 5000 -eb 1000 --exp_name pendulum --seed 1
```

## 5 GAE Lambda Sweep on LunarLander

Q: Which lambda values did you try, and which one looked best?

A: I tried `λ ∈ {0, 0.95, 0.98, 0.99, 1}`. If I judge by the final eval return, `λ=0.99` looked best at about `102.51` by the end of training. If I judge by the highest transient spike, `λ=1` was the biggest at about `209.05`, but the curve was quite noisy. The lower-lambda run (`λ=0`) was also noisy and ended badly despite occasional spikes.

Q: How did λ affect performance?

A: Small λ values put more weight on short-horizon bootstrapping, which reduced variance but seemed to underperform here. Larger λ values moved closer to Monte Carlo style returns and gave better task performance in this environment. In practice, the middle-to-high values (`0.95` to `0.99`) were the most useful for LunarLander.

Q: What does λ=0 correspond to, and what about λ=1?

A: `λ=0` corresponds to using the one-step TD-style advantage estimate, relying heavily on the critic’s bootstrap. `λ=1` corresponds to the full Monte Carlo return / reward-to-go style estimate. In these runs, the higher-λ settings worked better on LunarLander because they preserved more of the long-horizon return signal.

Q: What exact commands did you use for the LunarLander sweep?

A: I used these local commands:

```bash
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0 --exp_name lunar_lander_lambda0 --seed 1
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name lunar_lander_lambda095 --seed 1
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.98 --exp_name lunar_lander_lambda098 --seed 1
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.99 --exp_name lunar_lander_lambda099 --seed 1
uv run src/scripts/run.py --env_name LunarLander-v2 --ep_len 1000 --discount 0.99 -n 200 -b 2000 -eb 2000 -l 3 -s 128 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 1 --exp_name lunar_lander_lambda1 --seed 1
```
