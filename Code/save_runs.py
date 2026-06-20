from __future__ import annotations

import argparse
import json
import os

import pandas as pd


DEFAULT_GROUPS = [
    "final",
    "stage1",
    "stage2",
    "both_stages",
    "no_positional",
    "tune_rho",
    "sac",
    "combined",
    "release",
]


def wandb_path(project: str, entity: str | None = None) -> str:
    return f"{entity}/{project}" if entity else project


def create_run_dict(run):
    history = run.scan_history()

    step = []
    train_success = []
    eval_success = []
    success_cnt = []

    for log in history:
        step.append(log.get("step"))
        train_success.append(log.get("train_success"))
        eval_success.append(log.get("eval_success"))
        success_cnt.append(log.get("success_cnt", 0))

    data = pd.DataFrame(data={
        "step": step,
        "train_success": train_success,
        "eval_success": eval_success,
        "success_cnt": success_cnt,
    }).iloc[:-1]

    data = data.bfill()
    data = data.ffill()

    return {
        "env": run.config.get("env_name"),
        "palg": run.config.get("pos_alg_version"),
        "dalg": run.config.get("dir_alg_version"),
        "lam": run.config.get("lambda_"),
        "rho": run.config.get("rho"),
        "beta": run.config.get("beta"),
        "group": run.config.get("group"),
        "mult_norms": run.config.get("mult_norms", False),
        "baseline": run.config.get("baseline", ""),
        "data": data.to_dict(),
    }


def create_dict(project: str, run_name: str, entity: str | None = None):
    import wandb

    api = wandb.Api()
    run = api.run(f"{wandb_path(project, entity)}/{run_name}")
    return create_run_dict(run)


def save_run_dict(run_dict, run_name: str, replace_group="standard"):
    palg = run_dict["palg"]
    dalg = run_dict["dalg"]
    env = run_dict["env"]
    lam = run_dict["lam"]
    beta = run_dict["beta"]
    group = run_dict["group"]

    if group is None or replace_group != "standard":
        group = replace_group

    path = f"runs/{env}/p{palg}/d{dalg}/{group}/"
    filename = f"l{lam}_b{beta}_" + run_name.split("-")[-1]

    os.makedirs(path, exist_ok=True)

    with open(path + filename, "w") as f:
        json.dump(run_dict, f, indent=4)

    print(path + filename)


def save_run(project: str, run_name: str, entity: str | None = None, replace_group="standard"):
    run_dict = create_dict(project, run_name, entity)
    save_run_dict(run_dict, run_name, replace_group)


def save_runs(
    project: str,
    entity: str | None = None,
    env: str | None = None,
    palg: int | None = None,
    dalg: int | None = None,
    group: str | list[str] | None = None,
    allowed_groups: list[str] | None = None,
):
    import wandb

    api = wandb.Api()
    runs = api.runs(wandb_path(project, entity))
    allowed_groups = allowed_groups or DEFAULT_GROUPS

    for run in runs:
        run_group = run.config.get("group")

        if env is not None and run.config.get("env_name") != env:
            continue
        if palg is not None and run.config.get("pos_alg_version") != palg:
            continue
        if dalg is not None and run.config.get("dir_alg_version") != dalg:
            continue
        if group is not None:
            if isinstance(group, str) and group != run_group:
                continue
            if isinstance(group, list) and run_group not in group:
                continue

        if run_group in allowed_groups:
            if run.state == "running":
                print(f"Skipping {run.name}, still running.")
                continue
            run_dict = create_run_dict(run)

            if run_group == "final":
                if run.config.get("pos_traj_amount") == 50:
                    save_run_dict(run_dict, run.name, "stage2")
            else:
                save_run_dict(run_dict, run.name, run_group)


def parse_args():
    parser = argparse.ArgumentParser(description="Download W&B run summaries into local runs/ files.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument("--entity", default="", help="Optional W&B entity or team.")
    parser.add_argument("--env", default=None, help="Optional environment filter.")
    parser.add_argument("--palg", type=int, default=None, help="Optional positional algorithm filter.")
    parser.add_argument("--dalg", type=int, default=None, help="Optional directional algorithm filter.")
    parser.add_argument("--group", default=None, help="Optional run group filter.")
    return parser.parse_args()


def main():
    args = parse_args()
    save_runs(
        project=args.project,
        entity=args.entity or None,
        env=args.env,
        palg=args.palg,
        dalg=args.dalg,
        group=args.group,
    )


if __name__ == "__main__":
    main()
