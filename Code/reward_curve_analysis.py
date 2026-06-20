import pickle
import numpy as np
from models.projection import RewardModel
from analysis_utils import Config, get_text_and_goal_embeddings, get_img_from_fig
import os
import cv2
import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt



def get_rewards(model_name, success = True, traj_num = 0, seed = 0):
    folder, file = ("positive" if success else "negative"), ("pos" if success else "neg")
    traj = pickle.load(open(f"case_analysis/{model_name}/{folder}/{file}_{traj_num}", "rb"))

    embeddings = np.array([item[-3] for item in traj])
    prev_embeddings = np.array([embeddings[max(0, i-5)] for i in range(len(embeddings))])

    config = Config(0, 3, 80, 0.4, False)


    text_embedding, baseline_embedding, goal_embedding = get_text_and_goal_embeddings("door-open-v2-goal-hidden")
    model = RewardModel(
        config = config,
        seed = seed,
        ckpt_dir=os.path.abspath("./cpt"),
        text_embedding = text_embedding,
        baseline_embedding = baseline_embedding,
        goal_embedding = goal_embedding)
    
    model.load(os.path.abspath("./cpt"), model_name)

    pos_rewards, dir_rewards = [], []

    for embedding, prev_embedding in zip(embeddings, prev_embeddings):
        positional, directional = model.get_vlm_reward(model.proj_state, embedding, prev_embeddings=prev_embedding, dir_proj_state = model.dir_proj_state)
        pos_rewards.append(positional)
        dir_rewards.append(directional)
    
    return pos_rewards, dir_rewards


def get_video(model_name, success = True, traj_num = 0, seed = 0):

    folder, file = ("positive" if success else "negative"), ("pos" if success else "neg")
    traj = pickle.load(open(f"case_analysis/{model_name}/{folder}/{file}_{traj_num}", "rb"))

    pos_rewards, dir_rewards = get_rewards(model_name, success, traj_num, seed)
    

    frames = [item[-1] for item in traj]

    min_pos_reward = min(pos_rewards)
    max_pos_reward = max(pos_rewards)

    min_dir_reward = min(dir_rewards)
    max_dir_reward = max(dir_rewards)

    pos_margin = (max_pos_reward - min_pos_reward) * 0.05
    dir_margin = (max_dir_reward - min_dir_reward) * 0.05

    out = cv2.VideoWriter(f"videos/{folder}_s{seed}_{traj_num}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (2000, 1000), True)

    pb = tqdm.tqdm(total=len(frames))

    for i, frame in enumerate(frames):
        frame = frame[:,:,[2, 1, 0]]
        frame = cv2.resize(frame, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)

        fig = plt.figure(figsize=(8, 3))
        fig.tight_layout(pad=0)
        plt.xlim((-10, 510))
        plt.ylim((min_pos_reward - pos_margin, max_pos_reward + pos_margin))
        plt.plot(np.arange(len(pos_rewards)), pos_rewards, linewidth=1, linestyle="--", alpha=0.4)
        plt.plot(np.arange(i+1), pos_rewards[:i+1], linewidth=1)
        plt.title("positional reward")
        fig.canvas.draw()
        pos_reward_image = get_img_from_fig(fig, (1000, 500))
        plt.close(fig)

        fig = plt.figure(figsize=(8, 3))
        fig.tight_layout(pad=0)
        plt.xlim((-10, 510))
        plt.ylim((min_dir_reward - dir_margin, max_dir_reward + dir_margin))
        plt.plot(np.arange(len(dir_rewards)), dir_rewards, linewidth=1, linestyle="--", alpha=0.4)
        plt.plot(np.arange(i+1), dir_rewards[:i+1], linewidth=1)
        plt.title("directional reward")
        fig.canvas.draw()
        dir_reward_image = get_img_from_fig(fig, (1000, 500))
        plt.close(fig)

        reward_plot_image = np.concatenate((pos_reward_image, dir_reward_image), axis=0)[:,:,[2,1,0]]

        frame = np.concatenate((frame, reward_plot_image), axis=1)
        out.write(frame)
        pb.update(1)

    pb.close()
    out.release()


# get_video("p0_d3_l80.0_b0.4-door-open-v2-goal-hidden-s1_20250828_220720", success=False, traj_num=1, seed=1)
# get_video("p0_d3_l80.0_b0.4-door-open-v2-goal-hidden-s1_20250828_220720", success=False, traj_num=2, seed=1)
# get_video("p0_d3_l80.0_b0.4-door-open-v2-goal-hidden-s1_20250828_220720", success=False, traj_num=3, seed=1)
# get_video("p0_d3_l80.0_b0.4-door-open-v2-goal-hidden-s1_20250828_220720", success=False, traj_num=4, seed=1)
# get_video("p0_d3_l80.0_b0.4-door-open-v2-goal-hidden-s1_20250828_220720", success=False, traj_num=5, seed=1)