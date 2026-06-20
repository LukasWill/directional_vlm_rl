export MUJOCO_GL=egl


ENVS=(
    "drawer-open-v2-goal-hidden" 
    "drawer-close-v2-goal-hidden"
    "window-open-v2-goal-hidden"
    "window-close-v2-goal-hidden"
    "push-v2-goal-hidden"
    "button-press-topdown-v2-goal-hidden"
    "door-open-v2-goal-hidden"
)

for env in "${ENVS[@]}"; do
    python main.py --config.exp_name=oracle --config.env_name=$env --config.seed=0
    python main.py --config.exp_name=oracle --config.env_name=$env --config.seed=1
    python main.py --config.exp_name=oracle --config.env_name=$env --config.seed=2
    python main.py --config.exp_name=oracle --config.env_name=$env --config.seed=3
done