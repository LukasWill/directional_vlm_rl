import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import time
import gymnasium as gym
import ml_collections
import numpy as np
import pandas as pd

from tqdm import trange
from models import SACAgent
from utils import ReplayBuffer, log_git, get_logger, make_env

import wandb
import pickle
import torch


###################
# Utils Functions #
###################
def eval_policy(agent: SACAgent,
                env: gym.Env,
                eval_episodes: int = 10):
    t1 = time.time()
    eval_reward, eval_success, avg_step = 0, 0, 0
    for i in range(1, eval_episodes + 1):
        obs, _ = env.reset()
        while True:
            avg_step += 1
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            eval_reward += reward
            if terminated or truncated:
                eval_success += info["success"] 
                break

    eval_reward /= eval_episodes
    eval_success /= eval_episodes
    avg_step /= eval_episodes

    return eval_reward, eval_success, avg_step, time.time() - t1


def setup_logging(config):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # logging
    exp_prefix =  f"sac"

    # set random seed
    torch.manual_seed(config.seed)

    run_config = {}
    for key, value in config.items():
        run_config[key] = value

    exp_name = f"{exp_prefix}-{config.env_name}-s{config.seed}_{timestamp}"

    if config.enable_logging:
        wandb_kwargs = {
            "project": config.wandb_project,
            "id": exp_name,
            "name": exp_name,
            "config": run_config,
        }
        if config.wandb_entity:
            wandb_kwargs["entity"] = config.wandb_entity
        wandb_run = wandb.init(**wandb_kwargs)

        return wandb_run, exp_name

    else:
        return None, exp_name


def setup_exp(config):
    # initialize the environment
    env = make_env(config.env_name,
                   image_size=480,
                   seed=config.seed)
    eval_seed = config.seed if "hidden" in config.env_name else config.seed+100
    eval_env = make_env(config.env_name,
                        seed=eval_seed,
                        image_size=480,
                        camera_id=config.camera_id)

    # environment parameter
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # SAC agent
    agent = SACAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     max_action=max_action,
                     seed=config.seed,
                     tau=config.tau,
                     gamma=config.gamma,
                     lr=config.lr,
                     hidden_dims=config.hidden_dims)

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim)

    return env, eval_env, agent, replay_buffer


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()

    # logging
    wandb_run, exp_name = setup_logging(config)

    # experiment setup
    (env,
     eval_env,
     agent,
     replay_buffer) = setup_exp(config)

    # reward for untrained agent
    eval_episodes = 1 if "hidden" in config.env_name else 10
    eval_reward, eval_success, _, _ = eval_policy(agent,    
                                                  eval_env,
                                                  eval_episodes=eval_episodes)
    # start training
    obs, _ = env.reset()
    success, cum_success, ep_step = 0, 0, 0
    train_success_cnt, train_ep_cnt, train_ep_step = 0, 0, 0
    first_success_step = 0
    ep_task_reward, ep_reward = 0, 0
    lst_ep_task_reward, lst_ep_reward = 0, 0
    
    if config.load_positive:
        for i, (obs, action, next_obs, task_reward, terminated, truncated, info, _, _) in \
                enumerate(pickle.load(open(config.load_positive, "rb"))[:config.pos_traj_amount * 500]):
            replay_buffer.add(
                obs,
                action,
                next_obs,
                int(info["success"])-1,
                terminated)

            if i % 500 == 0:
                cum_success += 1

    for t in range(1, config.max_timesteps + 1):
        ep_step += 1
        if t <= config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(obs)
        next_obs, task_reward, terminated, truncated, info = env.step(action)
        cum_success += info["success"]

        replay_buffer.add(obs,
                          action,
                          next_obs,
                          info["success"]-1,
                          terminated)
        obs = next_obs
        ep_reward += info["success"]
        ep_task_reward += task_reward

        if first_success_step == 0 and int(info["success"]):
            first_success_step = ep_step

        # start a new trajectory
        if terminated or truncated:
            obs, _ = env.reset()
            success = info["success"] 
            lst_ep_task_reward = ep_task_reward
            lst_ep_reward = ep_reward
            ep_task_reward = 0
            ep_reward = 0
            ep_step = 0

            train_ep_cnt += 1
            train_ep_step += first_success_step

            if first_success_step != 0:
                train_success_cnt += 1
            
            first_success_step = 0

        # training
        if t > config.start_timesteps:
            batch = replay_buffer.sample(config.batch_size)
            log_info = agent.update(batch)

        # eval
        if t % config.eval_freq == 0:
            eval_reward, eval_success, _, _ = eval_policy(agent,
                                                          eval_env,
                                                          eval_episodes=eval_episodes)

        # logging
        if t % config.log_freq == 0:
            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "success": success,
                    "reward": lst_ep_reward,
                    "task_reward": lst_ep_task_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "batch_reward": batch.rewards.mean(),
                    "batch_reward_max": batch.rewards.max(),
                    "batch_reward_min": batch.rewards.min(),
                    "train_success": train_success_cnt / train_ep_cnt,
                    "train_avg_step": train_ep_step / train_ep_cnt,
                    "time": (time.time() - start_time) / 60
                })
                
                wandb_run.log(log_info)
            else:
                wandb_run.log({
                    "step": t,
                    "reward": lst_ep_reward,
                    "task_reward": lst_ep_task_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "train_success": train_success_cnt / train_ep_cnt,
                    "train_final_step": train_ep_step / train_ep_cnt,
                    "time": (time.time() - start_time) / 60,
                })
            
            train_success_cnt, train_ep_cnt, train_ep_step = 0, 0, 0
        
        if (t - 1) % (config.max_timesteps // 25) == 0:
            print(str(np.round((t-1) / config.max_timesteps * 100, -1)) + "%")


    print("100%")

    wandb_run.finish()

    # close env
    env.close()
    eval_env.close()
