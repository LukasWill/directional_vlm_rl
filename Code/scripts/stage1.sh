#!/usr/bin/env bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH -t 7:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -J temporal-vlm-rl

# Edit the SBATCH placeholders above for your cluster before submitting.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export WANDB_PROJECT=${WANDB_PROJECT:-temporal-vlm-rl}
export WANDB_ENTITY=${WANDB_ENTITY:-}

for i in $(seq 1 2)
do
	tasknum=$((i + ${SLURM_ARRAY_TASK_ID} * 2))
	read -r task pos_version dir_version lambda beta seed < <(sed "${tasknum}q;d" scripts/stage1.txt)
	python main.py --config.env_name=${task} --config.pos_alg_version=${pos_version} --config.dir_alg_version=${dir_version} \
		--config.lambda_=${lambda} --config.beta=${beta} --config.wandb_project=${WANDB_PROJECT} --config.wandb_entity=${WANDB_ENTITY} \
		--config.group=stage1 --config.seed=${seed} --config.max_timesteps=500000 \
		--config.stage1_eval=50 --config.baseline="robot arm beside table" &
done

wait
