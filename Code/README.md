# Code Overview

Run all commands in this directory unless noted otherwise.

## Environment Setup

From the repository root, create the recommended conda environment:

```shell
conda env create -f environment.yml
conda activate directional-vlm-rl
```

For an existing Python 3.11 environment, run `python -m pip install -r Code/requirements.txt` from the repository root. This installs CPU-compatible JAX by default; full training requires hardware-compatible JAX/PyTorch builds and an NVIDIA GPU.

For headless MuJoCo rendering:

```shell
export MUJOCO_GL=egl
```

## Training

Generate oracle data for a fixed-goal task:

```shell
python main.py --config.env_name=door-open-v2-goal-hidden --config.exp_name=oracle --config.enable_logging=false
```

Run the SAC baseline:

```shell
python main.py --config.env_name=door-open-v2-goal-hidden --config.exp_name=sac --config.enable_logging=false
```

Run the FuRL-style baseline or temporal reward variants:

```shell
python main.py \
  --config.env_name=door-open-v2-goal-hidden \
  --config.exp_name=furl \
  --config.pos_alg_version=0 \
  --config.dir_alg_version=3 \
  --config.lambda_=80 \
  --config.enable_logging=false
```

See the root `README.md` for the reward variant table and public release notes.

## Optional W&B Logging

```shell
python main.py \
  --config.env_name=door-open-v2-goal-hidden \
  --config.exp_name=furl \
  --config.enable_logging=true \
  --config.wandb_project=temporal-vlm-rl \
  --config.wandb_entity=
```

## Third-Party Baseline

This codebase adapts the FuRL implementation:

```txt
@InProceedings{fu2024,
  title = {FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning},
  author = {Yuwei Fu and Haichao Zhang and Di Wu and Wei Xu and Benoit Boulet},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024}
}
```
