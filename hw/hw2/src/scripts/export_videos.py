import argparse
import json
from pathlib import Path

import cv2
import gym
import numpy as np
import torch

from agents.pg_agent import PGAgent
from infrastructure import pytorch_util as ptu
from infrastructure import utils


def load_agent(exp_dir: Path) -> tuple[PGAgent, dict]:
    flags_path = exp_dir / "flags.json"
    if not flags_path.is_file():
        raise FileNotFoundError(f"Missing flags.json in {exp_dir}")

    flags = json.loads(flags_path.read_text(encoding="utf-8"))

    ptu.init_gpu(use_gpu=not flags.get("no_gpu", False), gpu_id=flags.get("which_gpu", 0))

    env = gym.make(flags["env_name"], render_mode=None)
    try:
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

        agent = PGAgent(
            ob_dim,
            ac_dim,
            discrete,
            n_layers=flags["n_layers"],
            layer_size=flags["layer_size"],
            gamma=flags["discount"],
            learning_rate=flags["learning_rate"],
            use_baseline=flags["use_baseline"],
            use_reward_to_go=flags["use_reward_to_go"],
            baseline_learning_rate=flags["baseline_learning_rate"],
            baseline_gradient_steps=flags["baseline_gradient_steps"],
            gae_lambda=flags["gae_lambda"],
            normalize_advantages=flags["normalize_advantages"],
        )
        state_dict = torch.load(exp_dir / "agent.pt", map_location=ptu.device)
        agent.load_state_dict(state_dict)
        agent.eval()
    finally:
        env.close()

    return agent, flags


def save_video(frames: np.ndarray, out_path: Path, fps: float) -> None:
    if frames.size == 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def export_videos(exp_dir: Path, output_dir: Path, num_trajectories: int) -> list[Path]:
    agent, flags = load_agent(exp_dir)

    env = gym.make(flags["env_name"], render_mode=None)
    try:
        max_ep_len = flags.get("ep_len") or env.spec.max_episode_steps
        if hasattr(env, "model"):
            fps = 1 / env.dt
        else:
            fps = env.env.metadata["render_fps"]

        trajs = utils.sample_n_trajectories(
            env,
            agent.actor,
            num_trajectories,
            max_ep_len,
            render=True,
        )
    finally:
        env.close()

    saved_paths: list[Path] = []
    run_output_dir = output_dir / exp_dir.name
    for idx, traj in enumerate(trajs):
        out_path = run_output_dir / f"rollout_{idx}.mp4"
        save_video(traj["image_obs"], out_path, fps)
        saved_paths.append(out_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        nargs="+",
        required=True,
        help="One or more experiment directories under exp/",
    )
    parser.add_argument(
        "--output_dir",
        default="exp_video",
        help="Directory to store exported mp4 files",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=2,
        help="Number of rollouts to export per experiment",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for exp_name in args.exp_dir:
        exp_dir = Path(exp_name)
        if not exp_dir.is_absolute():
            exp_dir = root / exp_dir
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        saved_paths = export_videos(exp_dir, output_dir, args.num_trajectories)
        for saved_path in saved_paths:
            print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()