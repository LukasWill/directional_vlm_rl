#!/usr/bin/env bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH -t 30:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -J temporal-vlm-rl

# Edit the SBATCH placeholders above for your cluster before submitting.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_PROJECT=${WANDB_PROJECT:-temporal-vlm-rl}
export WANDB_ENTITY=${WANDB_ENTITY:-}

for i in $(seq 1 2)
do
	tasknum=$((i + ${SLURM_ARRAY_TASK_ID} * 2))
	read -r task alg_version lambda seed < <(sed "${tasknum}q;d" scripts/tasks.txt)
	python main.py --config.env_name=${task} --config.alg_version=${alg_version} \
		--config.lambda_=${lambda} --config.seed=${seed} --config.max_timesteps=2000000 \
		--config.wandb_project=${WANDB_PROJECT} --config.wandb_entity=${WANDB_ENTITY} &
done

wait
