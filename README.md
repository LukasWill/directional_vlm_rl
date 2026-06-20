# VLM-RL Temporal Reward Shaping

[![CI](https://github.com/LukasWill/directional_vlm_rl/actions/workflows/ci.yml/badge.svg)](https://github.com/LukasWill/directional_vlm_rl/actions/workflows/ci.yml)

This repository contains code for temporal reward shaping in vision-language-model-guided reinforcement learning. It builds on FuRL-style visual-language rewards and adds reward variants that use temporal information from trajectories instead of scoring each observation independently.

The code is organized around Meta-World robot manipulation tasks and the LIV visual-language representation. The main entry point is `Code/main.py`.

## What Is Included

- A SAC baseline.
- A FuRL-style positional reward baseline.
- Delta-feature positional rewards that compare the current visual embedding to earlier trajectory context.
- Goal-baseline regularization rewards that compare task and baseline text descriptions.
- Directional rewards that score observed changes in representation space.
- Scripts for local experiments and optional Slurm-based batch runs.
- Utilities for downloading W&B run summaries and analyzing saved run data.

## Repository Layout

```text
.
├── Code/
│   ├── configs/              # Run configuration template
│   ├── experiments/          # Training and evaluation loops
│   ├── models/               # SAC, FuRL, LIV, and reward models
│   ├── scripts/              # Local and Slurm example launchers
│   ├── utils/                # Environments, buffers, logging, LIV loading
│   ├── analysis_utils.py     # Shared analysis helpers
│   ├── analyze_runs.py       # Run-summary analysis helper
│   ├── main.py               # Main training entry point
│   ├── plot_runs.py          # Plotting utilities for saved W&B summaries
│   ├── requirements.txt      # Pip-compatible Python dependencies
│   └── save_runs.py          # W&B run-summary export helper
├── tests/                    # Public-release smoke tests
├── CITATION.cff
├── environment.yml          # Recommended conda environment
├── LICENSE
├── NOTICE.md
└── README.md
```

Generated data, checkpoints, videos, W&B logs, run summaries, and trajectory files are ignored by default.

## Environment Setup

Create the recommended conda environment from the repository root:

```shell
conda env create -f environment.yml
conda activate directional-vlm-rl
```

Alternatively, install the Python dependencies into an existing Python 3.11 environment:

```shell
python -m pip install -r Code/requirements.txt
```

The default dependency file installs CPU-compatible JAX. Full training currently moves LIV tensors to CUDA and therefore expects an NVIDIA GPU. Before training, replace the default JAX/PyTorch installations with builds compatible with your CUDA driver and platform.

For headless rendering, set:

```shell
export MUJOCO_GL=egl
```

LIV checkpoints are downloaded by `Code/utils/liv_utils.py` through Hugging Face when first needed. The smoke tests and CI do not download checkpoints or require a GPU.

## Quickstart

Run commands from the `Code/` directory.

SAC baseline:

```shell
python main.py \
  --config.env_name=door-open-v2-goal-hidden \
  --config.exp_name=sac \
  --config.enable_logging=false
```

Generate an oracle trajectory for a task before running FuRL-style methods:

```shell
python main.py \
  --config.env_name=door-open-v2-goal-hidden \
  --config.exp_name=oracle \
  --config.enable_logging=false
```

Run the FuRL-style positional baseline and save positive trajectories:

```shell
python main.py \
  --config.exp_name=furl \
  --config.env_name=door-open-v2-goal-hidden \
  --config.pos_alg_version=0 \
  --config.dir_alg_version=0 \
  --config.seed=0 \
  --config.enable_logging=false \
  --config.save_positive=true
```

Run a directional reward variant with preloaded positive trajectories:

```shell
python main.py \
  --config.exp_name=furl \
  --config.env_name=door-open-v2-goal-hidden \
  --config.pos_alg_version=0 \
  --config.dir_alg_version=3 \
  --config.seed=0 \
  --config.lambda_=80 \
  --config.enable_logging=false \
  --config.checkpoint=true \
  --config.load_positive=/path/to/positive_trajectories
```

Training writes generated artifacts such as `logs/`, `saved_models/`, `pos_trajs/`, `runs/`, `videos/`, and `images/` under `Code/` depending on the selected options.

## Method Configuration

The main configuration is `Code/configs/metaworld.py`. Values can be overridden with `--config.key=value`.

Common settings:

| Setting | Meaning |
| --- | --- |
| `env_name` | Meta-World task name. |
| `exp_name` | Experiment type: `sac`, `oracle`, `furl`, `liv`, or `relay`. |
| `seed` | Random seed and fixed-goal initialization seed. |
| `max_timesteps` | Training horizon. |
| `rho` | Positional VLM reward scale used by FuRL-style rewards. |
| `enable_logging` | Enables W&B logging when `true`. Defaults to `true` in config, but README examples disable it. |
| `wandb_project` | W&B project name. |
| `wandb_entity` | Optional W&B entity/team. Leave empty to use the active W&B default. |
| `baseline` | Optional baseline text description. If empty, the task-specific default is used. |
| `checkpoint` | Save model parameters for later analysis. |
| `save_positive` | Save successful trajectories. |
| `load_positive` | Path to saved positive trajectories to preload. |
| `pos_traj_amount` | Number of positive trajectories to preload. |
| `stage1_eval` | Stop after collecting this many successful trajectories; `0` means normal training. |
| `temp_coef` | Temporal offset `k` for comparisons with earlier states. |
| `lambda_` | Directional reward scale. |
| `beta` | Goal-baseline regularization interpolation coefficient. |
| `no_positional_reward` | Disable positional rewards and train with directional reward only. |

Positional reward variants:

| `pos_alg_version` | Reward |
| --- | --- |
| `0` | Original FuRL-style positional reward. |
| `1` | Delta features, subtraction before projection heads. |
| `2` | Delta features before heads, without subtracting the initial state. |
| `3` | Delta features before heads, without subtracting the baseline text embedding. |
| `4` | Delta features, subtraction after projection heads. |
| `5` | Delta features after heads, without subtracting the initial state. |
| `6` | Delta features after heads, without subtracting the baseline text embedding. |
| `7` | Goal-baseline regularization after projection heads. |
| `8` | Goal-baseline regularization after heads, subtracting the initial state. |
| `9` | Goal-baseline regularization before projection heads. |
| `10` | Goal-baseline regularization before heads, subtracting the initial state. |

Directional reward variants:

| `dir_alg_version` | Reward |
| --- | --- |
| `0` | No directional reward. |
| `1` | Direction-only input, subtraction before projection heads. |
| `2` | Direction-only input, subtraction after projection heads. |
| `3` | Concatenated position and direction input, position is `s_i`. |
| `4` | Concatenated position and direction input, position is `s_i - s_0`. |
| `5` | Concatenated input with goal-baseline-regularized positional component. |

## Logging and Analysis

W&B logging is optional. To enable it:

```shell
python main.py \
  --config.exp_name=furl \
  --config.env_name=door-open-v2-goal-hidden \
  --config.enable_logging=true \
  --config.wandb_project=temporal-vlm-rl \
  --config.wandb_entity=
```

Download summaries from W&B into local `runs/` files:

```shell
python save_runs.py \
  --project temporal-vlm-rl \
  --entity your-wandb-entity \
  --group release
```

Analyze saved run summaries:

```shell
python analyze_runs.py --runs-dir runs
```

`Code/plot_runs.py` contains plotting utilities for saved run summaries. It is import-safe; run it as a script to generate the default example plots.

## Optional Slurm Jobs

`Code/scripts/` includes local and Slurm launch examples. Slurm scripts use generic `#SBATCH` placeholders such as `YOUR_ACCOUNT` and `gpu`; edit those for your own cluster before submitting jobs. W&B project/entity values can be provided through `WANDB_PROJECT` and `WANDB_ENTITY`.

## Tests

Run the public-release smoke tests from the repository root:

```shell
pytest -q
```

The test suite checks the configuration contract, dependency-file syntax, saved-run analysis, and release tree without importing GPU training dependencies. The test dependencies are included in `Code/requirements.txt`.

## Related Paper

This repository accompanies [Rewarding Change Beyond State: Directional VLM Rewards for Sample-Efficient Robot Reinforcement Learning](https://doi.org/10.1109/SII64115.2026.11404492).

## Third-Party Code and Acknowledgements

This repository builds on and adapts components from FuRL:

- Yuwei Fu et al. "FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning". Proceedings of the 41st International Conference on Machine Learning, 2024.
- Upstream code: https://github.com/fuyw/FuRL

It also vendors CLIP code and adapts LIV components. Their MIT license notices are preserved in `Code/clip/LICENSE` and `Code/LIV_LICENSE`. See `NOTICE.md` for provenance and licensing notes.

GitHub renders citation metadata from `CITATION.cff` in the repository's citation panel.

## References

[1] Yuwei Fu et al. "FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning". Proceedings of the 41st International Conference on Machine Learning, 2024.

[2] Yuchen Cui et al. "Can foundation models perform zero-shot task specification for robot manipulation?" Learning for Dynamics and Control Conference. PMLR, 2022.

[3] Juan Rocamonde et al. "Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning". The Twelfth International Conference on Learning Representations, 2024. https://openreview.net/forum?id=N0I2RtD8je
