export MUJOCO_GL=egl


TIME=500000
TASK=door-open-v2-goal-hidden

python main.py --config.env_name=$TASK --config.seed=1 --config.runner=local --config.max_timesteps=$TIME \
        --config.save_positive=true --config.enable_logging=false &

sleep 2s

python main.py --config.env_name=$TASK --config.seed=1 --config.runner=local --config.max_timesteps=$TIME \
        --config.save_positive=true --config.enable_logging=false &

wait