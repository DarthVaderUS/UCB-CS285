from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_runs() -> list[tuple[Path, dict, pd.DataFrame]]:
    runs: list[tuple[Path, dict, pd.DataFrame]] = []
    exp_root = ROOT / "exp"
    if not exp_root.is_dir():
        return runs

    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        flags_path = exp_dir / "flags.json"
        log_path = exp_dir / "log.csv"
        if not flags_path.is_file() or not log_path.is_file():
            continue
        flags = json.loads(flags_path.read_text(encoding="utf-8"))
        df = pd.read_csv(log_path)
        if df.empty:
            continue
        runs.append((exp_dir, flags, df))
    return runs


def prefer_best_final_run(runs: list[tuple[Path, dict, pd.DataFrame]]) -> list[tuple[Path, dict, pd.DataFrame]]:
    best_by_name: dict[str, tuple[Path, dict, pd.DataFrame]] = {}
    best_scores: dict[str, tuple[float, float]] = {}
    for item in runs:
        _, flags, df = item
        exp_name = flags.get("exp_name")
        if exp_name is None or "Eval_AverageReturn" not in df.columns:
            continue
        final_return = float(df["Eval_AverageReturn"].iloc[-1])
        final_step = float(df["Train_EnvstepsSoFar"].iloc[-1]) if "Train_EnvstepsSoFar" in df.columns else 0.0
        score = (final_return, final_step)
        if exp_name not in best_scores or score > best_scores[exp_name]:
            best_by_name[exp_name] = item
            best_scores[exp_name] = score
    return list(best_by_name.values())


def select_runs_by_name(
    runs: list[tuple[Path, dict, pd.DataFrame]],
    ordered_names: list[str],
) -> list[tuple[Path, dict, pd.DataFrame]]:
    by_name = {flags.get("exp_name"): item for item in runs for _, flags, _df in [item]}
    return [by_name[name] for name in ordered_names if name in by_name]


def label_from_flags(flags: dict) -> str:
    label = flags.get("exp_name", "run")
    tags = []
    if flags.get("use_reward_to_go"):
        tags.append("rtg")
    else:
        tags.append("traj")
    if flags.get("use_baseline"):
        tags.append("baseline")
    if flags.get("normalize_advantages"):
        tags.append("na")
    if flags.get("gae_lambda") is not None:
        tags.append(f"gae={flags['gae_lambda']}")
    if flags.get("baseline_gradient_steps") not in (None, 5):
        tags.append(f"bgs={flags['baseline_gradient_steps']}")
    if flags.get("baseline_learning_rate") not in (None, 0.005):
        tags.append(f"blr={flags['baseline_learning_rate']}")
    return f"{label} ({', '.join(tags)})" if tags else label


def plot_curves(
    runs: list[tuple[Path, dict, pd.DataFrame]],
    title: str,
    metric: str,
    out_path: Path,
    *,
    filter_fn=None,
    legend_loc: str = "best",
) -> None:
    plt.figure(figsize=(10, 6))
    plotted = False
    for _, flags, df in runs:
        if filter_fn is not None and not filter_fn(flags, df):
            continue
        if metric not in df.columns or "Train_EnvstepsSoFar" not in df.columns:
            continue
        x = df["Train_EnvstepsSoFar"]
        y = df[metric]
        plt.plot(x, y, linewidth=2, label=label_from_flags(flags))
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Environment steps")
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc=legend_loc, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    runs = prefer_best_final_run(load_runs())
    out_dir = ROOT / "exp_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    halfcheetah = select_runs_by_name(
        [item for item in runs if item[1].get("env_name") == "HalfCheetah-v4"],
        ["cheetah", "cheetah_baseline", "cheetah_baseline_bgs1"],
    )
    pendulum = select_runs_by_name(
        [item for item in runs if item[1].get("env_name") == "InvertedPendulum-v4"],
        ["pendulum", "pendulum_tuned_b500"],
    )
    lunar_lander = select_runs_by_name(
        [item for item in runs if item[1].get("env_name") == "LunarLander-v2"],
        [
            "lunar_lander_lambda0",
            "lunar_lander_lambda095",
            "lunar_lander_lambda098",
            "lunar_lander_lambda099",
            "lunar_lander_lambda1",
        ],
    )

    plot_curves(
        halfcheetah,
        "HalfCheetah-v4 eval return",
        "Eval_AverageReturn",
        out_dir / "halfcheetah_eval_return.png",
    )
    plot_curves(
        [item for item in halfcheetah if item[1].get("use_baseline")],
        "HalfCheetah-v4 baseline loss",
        "Baseline Loss",
        out_dir / "halfcheetah_baseline_loss.png",
    )

    plot_curves(
        lunar_lander,
        "LunarLander-v2 eval return by lambda",
        "Eval_AverageReturn",
        out_dir / "lunarlander_lambda_eval_return.png",
        filter_fn=lambda flags, _df: flags.get("exp_name", "").startswith("lunar_lander_lambda"),
        legend_loc="lower right",
    )

    plot_curves(
        pendulum,
        "InvertedPendulum-v4 eval return",
        "Eval_AverageReturn",
        out_dir / "invertedpendulum_eval_return.png",
        legend_loc="lower right",
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()