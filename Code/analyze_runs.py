import argparse
import json
import os

import numpy as np
import pandas as pd


DEFAULT_AVG_STEPS = {
    "drawer-open-v2-goal-hidden": 150000,
    "door-open-v2-goal-hidden": 200000,
    "window-open-v2-goal-hidden": 100000,
    "window-close-v2-goal-hidden": 100000,
    "push-v2-goal-hidden": 497500,
    "button-press-topdown-v2-goal-hidden": 200000,
}


def iter_run_files(runs_dir: str):
    for root, _, files in os.walk(runs_dir):
        for filename in files:
            yield os.path.join(root, filename)


def load_runs(runs_dir: str):
    run_dicts = []
    for path in iter_run_files(runs_dir):
        try:
            with open(path) as f:
                run_dicts.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return run_dicts


def running_avg_at(run_dict, step: int, metric="train_success", num_logs=30):
    data = pd.DataFrame(run_dict["data"])
    if data.empty or metric not in data:
        return None

    if data["step"].iloc[-1] < step:
        values = data[metric].to_numpy()[-num_logs:]
    else:
        step_index = np.where(data["step"].to_numpy() >= step)[0][0]
        values = data[metric].to_numpy()[max(0, step_index - num_logs):step_index]

    if len(values) == 0:
        return None
    return float(np.nanmean(values))


def summarize_release_runs(runs_dir: str, metric="train_success"):
    runs = load_runs(runs_dir)
    rows = []

    for env, step in DEFAULT_AVG_STEPS.items():
        env_runs = [run for run in runs if run.get("env") == env]
        for palg, dalg in sorted({(run.get("palg"), run.get("dalg")) for run in env_runs}):
            scores = [
                running_avg_at(run, step, metric)
                for run in env_runs
                if run.get("palg") == palg and run.get("dalg") == dalg
            ]
            scores = [score for score in scores if score is not None]
            if scores:
                rows.append({
                    "env": env,
                    "pos_alg": palg,
                    "dir_alg": dalg,
                    "runs": len(scores),
                    metric: float(np.mean(scores)),
                })

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize saved run dictionaries.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing saved run dictionaries.")
    parser.add_argument("--metric", default="train_success", help="Metric to summarize.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize_release_runs(args.runs_dir, args.metric)
    if summary.empty:
        print(f"No saved runs found in {args.runs_dir}.")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
