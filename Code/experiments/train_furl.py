import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".4"

import cv2
import time
import clip
import optax
import imageio
import ml_collections
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wandb
import pickle

from tqdm import trange

import torch
import torchvision.transforms as T

from models import SACAgent, FuRLAgent, RewardModel
from utils import (TASKS, DistanceBuffer, EmbeddingBuffer, log_git,
                   get_logger, make_env, load_liv)


###################
# Utils Functions #
###################
def crop_center(config, image):
    x1, x2, y1, y2 = 32, 224, 32, 224
    return image[x1:x2, y1:y2, :]


def eval_policy(agent: SACAgent,
                env: gym.Env,
                eval_episodes: int = 10):
    t1 = time.time()
    eval_reward, eval_success, avg_step = 0, 0, 0
    for i in range(1, eval_episodes + 1):
        obs, _ = env.reset()
        success = False
        while True:
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            eval_reward += reward
            if not success:
                avg_step += 1
                if info["success"]:
                    eval_success += 1
                    success = True
            if terminated or truncated:
                break

    eval_reward /= eval_episodes
    eval_success /= eval_episodes
    avg_step /= eval_episodes

    return eval_reward, eval_success, avg_step, time.time() - t1


def final_eval_video(agent, env, camera_id = 2):
    # run it 20 times to try to get a successful trajectory for video
    # if it never succeeds, then it just saves the last trajectory
    success = False
    frames = []
    for _ in range(20):
        obs, _ = env.reset()
        frames = []
        frame = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=camera_id)
        frames.append(frame[::-1])
        while True:
            action = agent.sample_action(obs, eval_mode=True)
            obs, _, terminated, truncated, info = env.step(action)
            frame = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=camera_id)
            frames.append(frame[::-1])
            if terminated or truncated:
                success = info["success"] == 1
                break
        
        if success:
            break
    
    
    return frames


def failed_traj_video(agent, env, camera_id = 2):
    # run it 20 times to try to get a successful trajectory for video
    # if it never succeeds, then it just saves the last trajectory
    success = False
    frames = []
    for _ in range(20):
        obs, _ = env.reset()
        frames = []
        frame = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=camera_id)
        frames.append(frame[::-1])
        while True:
            action = agent.sample_action(obs, eval_mode=True)
            obs, _, terminated, truncated, info = env.step(action)
            frame = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=camera_id)
            frames.append(frame[::-1])
            if terminated or truncated:
                success = info["success"] == 1
                break
        
        if not success:
            break
    
    
    return frames

        
BASELINES = {
    "drawer-open-v2-goal-hidden" : "closed drawer",
    "drawer-open-v2-goal-observable" : "closed drawer",
    "drawer-close-v2-goal-hidden" : "open drawer",
    "drawer-close-v2-goal-observable" : "open drawer",
    "window-open-v2-goal-hidden" : "closed window",
    "window-open-v2-goal-observable" : "closed window",
    "window-close-v2-goal-hidden" : "open window",
    "window-close-v2-goal-observable" : "open window",
    "button-press-topdown-v2-goal-hidden" : "unpressed button",
    "button-press-topdown-v2-goal-observable" : "unpressed button",
    "door-open-v2-goal-hidden" : "closed door with a revolving joint",
    "door-open-v2-goal-observable" : "closed door with a revolving joint",
    "push-v2-goal-hidden" : "puck on table not at goal",
    "push-v2-goal-observable" : "puck on table not at goal"
}    

def setup_logging(config):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # logging
    exp_prefix =  f"p{config.pos_alg_version}_d{config.dir_alg_version}_l{config.lambda_}_b{config.beta}"

    # set random seed
    torch.manual_seed(config.seed)

    run_config = {}
    for key, value in config.items():
        run_config[key] = value
    
    if config.baseline == "":
        run_config["baseline"] = BASELINES[config.env_name]

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
    # liv
    transform = T.Compose([T.ToTensor()])
    liv = load_liv()

    # task description embedding
    with torch.no_grad():
        token = clip.tokenize([TASKS[config.env_name]])
        text_embedding = liv(input=token, modality="text")
    text_embedding = text_embedding.detach().cpu().numpy()
    data = np.load(f"data/oracle/{config.env_name}/s{config.seed}_c{config.camera_id}.npz")

    print(config.baseline)

    baseline = BASELINES[config.env_name] if config.baseline == "" else config.baseline

    # baseline description embedding
    with torch.no_grad():
        token = clip.tokenize([baseline])
        baseline_embedding = liv(input=token, modality="text")
    baseline_embedding = baseline_embedding.detach().cpu().numpy()

    # goal_embedding / text_embedding
    oracle_images = data["images"]
    oracle_success = data["success"]
    oracle_traj_len = np.where(oracle_success)[0][0] + 1  # 84

    # initialize the environment
    env = make_env(config.env_name,
                   seed=config.seed,
                   camera_id=config.camera_id)
    eval_seed = config.seed if "hidden" in config.env_name else config.seed+100
    eval_env = make_env(config.env_name,
                        seed=eval_seed,
                        image_size=256,
                        camera_id=config.camera_id)

    # environment parameter
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    goal_image = data["images"][oracle_traj_len-1]
    goal_image = crop_center(config, goal_image)
    processed_goal_image = cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR)
    processed_goal_image = transform(processed_goal_image)
    goal_embedding = liv(input=processed_goal_image.to("cuda")[None], modality="vision")
    goal_embedding = goal_embedding.detach().cpu().numpy()

    # fixed LIV representation projection
    vlm_agent = FuRLAgent(obs_dim=obs_dim,
                          act_dim=act_dim,
                          max_action=max_action,
                          seed=config.seed,
                          tau=config.tau,
                          rho=config.rho,
                          margin=config.cosine_margin,
                          gamma=config.gamma,
                          lr=config.lr,
                          text_embedding=text_embedding,
                          goal_embedding=goal_embedding,
                          hidden_dims=config.hidden_dims)

    # SAC agent
    sac_agent = SACAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_action=max_action,
                         seed=config.seed,
                         tau=config.tau,
                         gamma=config.gamma,
                         lr=config.lr,
                         hidden_dims=config.hidden_dims)

    # Initialize the reward model
    reward_model = RewardModel(config,
                               seed=config.seed,
                               emb_dim=1024,
                               ckpt_dir=os.path.abspath("./cpt"),
                               text_embedding=text_embedding,
                               baseline_embedding=baseline_embedding,
                               goal_embedding=goal_embedding)

    # Replay buffer
    replay_buffer = DistanceBuffer(obs_dim=obs_dim,
                                   act_dim=act_dim,
                                   max_size=int(5e5))

    return (
        transform,
        liv,
        env,
        eval_env,
        vlm_agent,
        sac_agent,
        reward_model,
        replay_buffer,
        goal_image,
    )


# def params_is_nan(params):
#     for key, value in params.items():
#         if not isinstance(value, dict):
#             if np.isnan(value).any():
#                 return True
#         elif params_is_nan(value):
#             return True

#     return False

#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()

    # logging setup
    wandb_run, exp_name = setup_logging(config)
 
    # experiment setup
    (transform,
     liv,
     env,
     eval_env,
     vlm_agent,
     sac_agent,
     reward_model,
     replay_buffer,
     goal_image) = setup_exp(config)
    
    # reward for untrained agent
    eval_episodes = 1 if "hidden" in config.env_name else 10
    eval_reward, eval_success, avg_step, _ = eval_policy(vlm_agent,
                                                  eval_env,
                                                  eval_episodes)

    if config.enable_logging:
        wandb_run.log({
            "step": 0,
            "eval_reward": eval_reward,
            "eval_success": eval_success,
            "avg_step": avg_step
        })

    first_success_step = 0 

    # trajectory embedding
    embedding_buffer = EmbeddingBuffer(emb_dim=1024,
                                       gap=config.gap,
                                       max_size=config.embed_buffer_size)
    traj_embeddings = np.zeros((500, 1024))
    traj_success = np.zeros(500)

    # relay freqs
    relay_freqs = [50, 100, 150, 200]
    relay_freq = np.random.choice(relay_freqs)

    # start training
    obs, _ = env.reset()

    if config.save_first_stage:
        first_stage = []
    elif config.load_first_stage:
        first_stage = pickle.load(open(config.load_first_stage, "rb"))

    if config.save_positive or config.save_negative:
        current_trajectory = []
        if config.save_positive:
            pos_trajs = []
            saved_pos_count = 0
        if config.save_negative:
            neg_trajs = []
            saved_neg_count = 0
    
    if config.save_cases:
        case_neg_trajs = []
        case_pos_trajs = []
        case_traj = []
        

    success_cnt, success_ep_cnt, ep_num, ep_step = 0, 0, 0, 0
    
    if config.load_positive:
        after_success = False
        valid = True
        for i, (obs, action, next_obs, task_reward, terminated, truncated, info, image_embedding, l2_distance) in \
                enumerate(pickle.load(open(config.load_positive, "rb"))[:config.pos_traj_amount * 500]):
            replay_buffer.add(
                obs,
                action,
                next_obs,
                int(info["success"])-1,
                terminated,
                image_embedding,
                l2_distance)
            
            if not after_success and task_reward:
                after_success = True
            
            if after_success and not task_reward:
                valid = False

            if i % 500 == 0:
                success_cnt += 1
                after_success = False
                valid = True

            if valid:
                embedding_buffer.add(
                    embedding = image_embedding,
                    success = True,
                    valid = i % 500 >= config.gap)

    reward, ep_task_reward, ep_vlm_reward, ep_positional_reward, ep_directional_reward = 0, 0, 0, 0, 0
    train_success_cnt, train_ep_cnt, train_ep_step = 0, 0, 0
    lst_ep_task_reward, lst_ep_vlm_reward = 0, 0
    sac_step, vlm_step = 0, 0
    policies = ["vlm", "sac"]
    use_relay = True
    for t in range(1, config.max_timesteps + 1):
        if not config.load_first_stage or t > len(first_stage):
            if t <= config.start_timesteps:
                action = env.action_space.sample()
            else:
                if use_relay:
                    if policies[(ep_step//relay_freq)%2] == "vlm":
                        vlm_step += 1
                        action = vlm_agent.sample_action(obs)
                    else:
                        sac_step += 1
                        action = sac_agent.sample_action(obs)
                        action_noise = np.random.normal(
                            0, sac_agent.max_action*config.expl_noise, size=sac_agent.act_dim)
                        action = (action + action_noise).clip(
                            -sac_agent.max_action, sac_agent.max_action)
                else:
                    vlm_step += 1
                    action = vlm_agent.sample_action(obs)

            # try:
            #     print(log_info)
            #     print(proj_log_info)
            # except:
            #     pass
            if np.isnan(action).any():
                print(log_info)
                print(proj_log_info)
                return

            next_obs, task_reward, terminated, truncated, info = env.step(action)

            # vision language model reward
            image = env.mujoco_renderer.render(
                render_mode="rgb_array",
                camera_id=config.camera_id).copy()
            image = image[::-1]

            image = crop_center(config, image)
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            processed_image = transform(processed_image)
            with torch.no_grad():
                image_embedding = liv(input=processed_image.to("cuda")[None], modality="vision")
            image_embedding = image_embedding.detach().cpu().numpy()
            
            l2_distance = np.square(image_embedding - vlm_agent.goal_embedding).sum(-1)**0.5

            if config.save_first_stage:
                first_stage.append((obs, action, next_obs, task_reward, terminated, truncated, info, image_embedding, l2_distance))
            
            if config.save_positive or config.save_negative:
                current_trajectory.append((obs, action, next_obs, task_reward, terminated, truncated, info, image_embedding, l2_distance))

            if config.save_cases and t > config.max_timesteps * 0.75:
                case_traj.append((obs, action, next_obs, task_reward, terminated, truncated, info, image_embedding, l2_distance, image))

            
        elif config.load_first_stage:
            obs, action, next_obs, task_reward, terminated, truncated, info, image_embedding, l2_distance = first_stage[t-1]


        positional_reward, directional_reward = reward_model.get_vlm_reward(
            proj_state = reward_model.proj_state, 
            img_embeddings = image_embedding, 
            init_embedding = traj_embeddings[0], 
            prev_embeddings = traj_embeddings[max(0, ep_step - config.temp_coef)],
            dir_proj_state = reward_model.dir_proj_state)
        
        positional_reward = positional_reward.item()
        directional_reward = directional_reward.item()
        
        vlm_reward = (positional_reward if not config.no_positional_reward else 0) + config.lambda_ * directional_reward
        
        # print(vlm_reward)

        reward = int(info["success"])
        success_cnt += reward

        traj_embeddings[ep_step] = image_embedding
        traj_success[ep_step] = reward
        ep_step += 1

        if first_success_step == 0 and reward:
            first_success_step = ep_step

        # add to buffer
        replay_buffer.add(obs,
                          action,
                          next_obs,
                          reward-1,
                          terminated,
                          image_embedding,
                          l2_distance)
        obs = next_obs
        ep_positional_reward += positional_reward
        ep_directional_reward += directional_reward
        ep_vlm_reward += vlm_reward
        ep_task_reward += task_reward

        # start a new trajectory
        if terminated or truncated:
            obs, _ = env.reset()
            lst_ep_task_reward = ep_task_reward
            lst_ep_vlm_reward = ep_vlm_reward
            lst_ep_pos_reward = ep_positional_reward
            lst_ep_dir_reward = ep_directional_reward
            ep_vlm_reward = 0
            ep_task_reward = 0
            ep_positional_reward = 0
            ep_directional_reward = 0
            sac_step = 0
            vlm_step = 0
            policies = policies[::-1]
            relay_freq = np.random.choice(relay_freqs)

            train_ep_cnt += 1
            train_ep_step += first_success_step

            if config.save_positive and first_success_step > 0 and len(pos_trajs) // 500 < 500:
                pos_trajs.extend(current_trajectory)
                saved_pos_count += 1
            elif config.save_negative and first_success_step == 0 and len(neg_trajs) // 500 < 500 and (saved_neg_count < t // 3000 or t > config.max_timesteps * 0.6):
                neg_trajs.extend(current_trajectory)
                saved_neg_count += 1

            current_trajectory = []

            # save embedding
            if first_success_step == 0:
                if config.save_cases and t > config.max_timesteps * 0.75 and len(case_neg_trajs) < 10:
                    case_neg_trajs.append(case_traj)
                    case_traj = []


                for j in range(ep_step):
                    embedding_buffer.add(embedding=traj_embeddings[j],
                                         success=False)

            else:
                success_ep_cnt += 1
                if config.save_first_stage:
                    os.makedirs("first_stage", exist_ok=True)
                    pickle.dump(first_stage, open(f"first_stage/{exp_name}", "wb"))
                    break

                if config.save_cases and t > config.max_timesteps * 0.75 and len(case_pos_trajs) < 10:
                    case_pos_trajs.append(case_traj)
                    case_traj = []

                train_success_cnt += 1
                for j in range(first_success_step):
                    embedding_buffer.add(embedding=traj_embeddings[j],
                                         success=True,
                                         valid=j>=config.gap)

                for j in range(first_success_step, ep_step):
                    if traj_success[j]:
                        embedding_buffer.add(embedding=traj_embeddings[j],
                                             success=True,
                                             valid=j>=config.gap)
                    else:
                        break
                

            ep_step = 0
            ep_num += 1
            first_success_step = 0

            if use_relay and embedding_buffer.pos_size >= config.relay_threshold:
                use_relay = False

        # training
        if  t > config.start_timesteps:
            if (success_cnt > 0) and (embedding_buffer.valid_size > 0):
                batch = replay_buffer.sample(config.batch_size, config.temp_coef)
                embedding_batch = embedding_buffer.sample(config.batch_size, config.temp_coef)
                batch_positional_rewards, batch_directional_rewards = reward_model.get_vlm_reward(
                    proj_state = reward_model.proj_state,
                    img_embeddings = batch.embeddings,
                    init_embedding = traj_embeddings[0],
                    prev_embeddings = batch.prev_embeddings,
                    dir_proj_state = reward_model.dir_proj_state)
                
                batch_vlm_rewards = (batch_positional_rewards if not config.no_positional_reward else 0) + config.lambda_ * batch_directional_rewards

                proj_log_info = reward_model.update_pos(embedding_batch, traj_embeddings[0])
                log_info = vlm_agent.update(batch, batch_vlm_rewards)

            # collected zero successful trajectory
            else:
                batch = replay_buffer.sample_with_mask(config.batch_size, config.temp_coef, config.l2_margin)

                proj_log_info = reward_model.update_neg(batch, traj_embeddings[0])
                batch_vlm_rewards = proj_log_info.get("vlm_rewards")
                log_info = vlm_agent.update(batch, batch_vlm_rewards)

            # update SAC agent
            if use_relay: _ = sac_agent.update(batch)

        # eval
        if t % config.eval_freq == 0:
            eval_reward, eval_success, avg_step, _ = eval_policy(vlm_agent,
                                                          eval_env,
                                                          eval_episodes)

        # logging
        if config.enable_logging and t % config.log_freq == 0:
            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "success": reward,
                    "task_reward": lst_ep_task_reward,
                    "vlm_reward": lst_ep_vlm_reward,
                    "positional_reward": lst_ep_pos_reward,
                    "directional_reward": lst_ep_dir_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "batch_reward": batch.rewards.mean(),
                    "batch_reward_max": batch.rewards.max(),
                    "batch_reward_min": batch.rewards.min(),
                    "batch_vlm_reward": batch_vlm_rewards.mean(),
                    "batch_vlm_reward_max": batch_vlm_rewards.max(),
                    "batch_vlm_reward_min": batch_vlm_rewards.min(),
                    "time": (time.time() - start_time) / 60,
                    "train_success": train_success_cnt / train_ep_cnt,
                    "train_avg_step": train_ep_step / train_ep_cnt,
                    "avg_step": avg_step,
                    "success_cnt" : success_ep_cnt
                })
                log_info.update(proj_log_info)
                wandb_run.log(log_info)
            else:
                wandb_run.log({
                    "step": t,
                    "task_reward": lst_ep_task_reward,
                    "vlm_reward": lst_ep_vlm_reward,
                    "positional_reward": lst_ep_pos_reward,
                    "directional_reward": lst_ep_dir_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "time": (time.time() - start_time) / 60,
                    "train_success": train_success_cnt / train_ep_cnt,
                    "train_final_step": train_ep_step / train_ep_cnt,
                    "avg_step": avg_step,
                    "success_cnt" : success_ep_cnt
                })
            
            train_success_cnt, train_ep_cnt, train_ep_step = 0, 0, 0
        
        if (t - 1) % (config.max_timesteps // 25) == 0:
            print(str(np.round((t-1) / config.max_timesteps * 100, 0)) + "%")

        # stage 1 evaluation
        if config.stage1_eval > 0 and success_ep_cnt >= config.stage1_eval:
            break
        


    print("100%")

    if config.save_cases:
        os.makedirs(f"case_analysis/{exp_name}/positive")
        os.makedirs(f"case_analysis/{exp_name}/negative")
        for i, pos_traj in enumerate(case_pos_trajs):
            pickle.dump(pos_traj, open(f"case_analysis/{exp_name}/positive/pos_{i}", "wb"))
        for i, neg_traj in enumerate(case_neg_trajs):
            pickle.dump(neg_traj, open(f"case_analysis/{exp_name}/negative/neg_{i}", "wb"))
        

    if config.save_positive:
        os.makedirs("pos_trajs", exist_ok=True)
        pickle.dump(pos_trajs, open(f"pos_trajs/{exp_name}", "wb"))

    if config.save_negative:
        os.makedirs("neg_trajs", exist_ok=True)
        pickle.dump(neg_trajs, open(f"neg_trajs/{exp_name}", "wb"))

    if config.checkpoint:
        if not os.path.isdir("cpt"):
            os.mkdir("cpt")
        
        reward_model.save(exp_name)
    
    frames = final_eval_video(vlm_agent, eval_env)


    final_video = np.array(frames, dtype=np.uint8).transpose((0, 3, 1, 2))

    if config.enable_logging:
        wandb_run.log({ "video" : wandb.Video(final_video, fps=60, format="mp4") })
        wandb_run.finish()


    # close env
    env.close()
    eval_env.close()
