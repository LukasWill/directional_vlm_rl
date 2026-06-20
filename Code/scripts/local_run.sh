export MUJOCO_GL=egl


TASK=door-open-v2-goal-hidden
TIME=200000
NUM_POS=50
PROJECT=temporal-vlm-rl

GROUP=release

PALG=0
DALG=3
LAM=80
BETA=0.4


POS_TRAJS=pos_trajs/trimmed/door-open-v2-goal-hidden_50-0
python main.py --config.env_name=$TASK --config.pos_alg_version=$PALG --config.dir_alg_version=$DALG --config.seed=0 \
        --config.runner=local --config.lambda_=$LAM --config.max_timesteps=$TIME  --config.beta=$BETA --config.save_cases=true --config.enable_logging=false \
        --config.wandb_project=$PROJECT --config.group=$GROUP --config.checkpoint=true --config.load_positive=$POS_TRAJS &

POS_TRAJS=pos_trajs/trimmed/door-open-v2-goal-hidden_50-1
python main.py --config.env_name=$TASK --config.pos_alg_version=$PALG --config.dir_alg_version=$DALG --config.seed=1 \
        --config.runner=local --config.lambda_=$LAM --config.max_timesteps=$TIME  --config.beta=$BETA --config.save_cases=true --config.enable_logging=false \
        --config.wandb_project=$PROJECT --config.group=$GROUP --config.checkpoint=true --config.load_positive=$POS_TRAJS &

wait


POS_TRAJS=pos_trajs/trimmed/door-open-v2-goal-hidden_50-2
python main.py --config.env_name=$TASK --config.pos_alg_version=$PALG --config.dir_alg_version=$DALG --config.seed=2 \
        --config.runner=local --config.lambda_=$LAM --config.max_timesteps=$TIME  --config.beta=$BETA --config.save_cases=true --config.enable_logging=false \
        --config.wandb_project=$PROJECT --config.group=$GROUP --config.checkpoint=true --config.load_positive=$POS_TRAJS &

POS_TRAJS=pos_trajs/trimmed/door-open-v2-goal-hidden_50-3
python main.py --config.env_name=$TASK --config.pos_alg_version=$PALG --config.dir_alg_version=$DALG --config.seed=3 \
        --config.runner=local --config.lambda_=$LAM --config.max_timesteps=$TIME  --config.beta=$BETA --config.save_cases=true --config.enable_logging=false \
        --config.wandb_project=$PROJECT --config.group=$GROUP --config.checkpoint=true --config.load_positive=$POS_TRAJS &

wait
