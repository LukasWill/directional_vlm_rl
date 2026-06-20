import numpy as np
import cv2
import torchvision.transforms as T
import torch
from utils.liv_utils import load_liv
import os
import clip
from experiments.train_furl import crop_center
from models.projection import RewardModel
from utils import TASKS
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
import io
import pickle
from collections import namedtuple


transform = T.Compose([T.ToTensor()])
liv = load_liv()

def get_embeddings(frames):
    transform = T.Compose([T.ToTensor()])
    liv = load_liv()
    processed_images = [
        transform(cv2.cvtColor(crop_center(None, image), cv2.COLOR_RGB2BGR))
        for image in frames]
    
    with torch.no_grad():
        traj_embeddings = [
            liv(input=processed_image.to("cuda")[None], modality="vision")
            for processed_image in processed_images]
    
    return [embedding.detach().cpu().numpy() for embedding in traj_embeddings]


def get_text_and_goal_embeddings(env_name):
    with torch.no_grad():
        text_token = clip.tokenize([TASKS[env_name]])
        baseline_token = clip.tokenize(["robot arm beside table"])
        text_embedding = liv(input=text_token, modality="text")
        baseline_embedding = liv(input=baseline_token, modality="text")

    text_embedding = text_embedding.detach().cpu().numpy()
    baseline_embedding = baseline_embedding.detach().cpu().numpy()

    data = np.load(f"data/oracle/{env_name}/s0_c2.npz")
    oracle_success = data["success"]
    oracle_traj_len = np.where(oracle_success)[0][0] + 1
    goal_image = data["images"][oracle_traj_len-1]
    goal_image = crop_center(None, goal_image)
    processed_goal_image = cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR)
    processed_goal_image = transform(processed_goal_image)
    goal_embedding = liv(input=processed_goal_image.to("cuda")[None], modality="vision")
    goal_embedding = goal_embedding.detach().cpu().numpy()

    return text_embedding, baseline_embedding, goal_embedding


def get_img_from_fig(fig, dsize):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)


def get_params_new(model_name):
    seed = model_name.split("s")[-1].split("_")[0]
    left, right = model_name.index("b"), model_name.index("hidden") if "hidden" in model_name else model_name.index("observable")
    while model_name[left] != "-":
        left += 1
    left += 1
    while model_name[right] != "-":
        right += 1
    
    env = model_name[left:right]

    palg = int(model_name.split("_")[0][1:])
    dalg = int(model_name.split("_")[1][1:])
    lam = float(model_name.split("_")[2][1:])
    beta = float(model_name.split("_")[3].split("-")[0][1:])

    return palg, dalg, lam, beta, env, seed


def get_params_old(model_name):
    model_name_split = model_name.split("_")
    alg_version = int(model_name_split[0][1:])
    lam = float(model_name_split[1][1:])
    env_name = model_name_split[3][1:]
    for i in range(1, len(env_name)):
        if env_name[-i] == "s":
            seed=int(env_name[1-i:])
            env_name = env_name[:len(env_name)-i-1]
            break

    return alg_version, lam, env_name, seed

def get_params(model_name):
    if model_name[0] == "a":
        return get_params_old(model_name)
    else:
        return get_params_new(model_name)


Config = namedtuple("Config", ["pos_alg_version", "dir_alg_version", "lambda_", "beta", "mult_norms"])

def get_rewards(model_name, traj_embeddings = None, traj_num = 0, successful = True):
    if not traj_embeddings:
        traj_embeddings = get_embeddings(pickle.load(open(f"trajs/{model_name}/" + ("successful" if successful else "failed") + f"/traj{traj_num}", "rb"))["frames"])

    if model_name[0] == "a":
        alg_version, lam, env_name, seed = get_params(model_name)
        text_embedding, baseline_embedding, goal_embedding = get_text_and_goal_embeddings(env_name)
        config = Config(0, 3, lam, 0.4, False)
        reward_model = RewardModel(
            seed=seed,
            config=config,
            emb_dim=1024,
            ckpt_dir=os.path.abspath("./cpt"),
            text_embedding=text_embedding,
            baseline_embedding=baseline_embedding,
            goal_embedding=goal_embedding)
        reward_model.load(os.path.abspath("./cpt"), model_name)
    else:
        palg, dalg, lam, beta, env_name, seed = get_params(model_name)
        config = Config(palg, dalg, lam, beta, False)
        text_embedding, baseline_embedding, goal_embedding = get_text_and_goal_embeddings(env_name)
        reward_model = RewardModel(
            config=config,
            seed=int(model_name[-1]),
            emb_dim=1024,
            ckpt_dir=os.path.abspath("./cpt"),
            text_embedding=text_embedding,
            baseline_embedding=baseline_embedding,
            goal_embedding=goal_embedding)
        reward_model.load(os.path.abspath("./cpt"), model_name)


    pos_rewards = []
    dir_rewards = []
    for i in range(len(traj_embeddings)):
        embedding = traj_embeddings[i]
        prev_embedding = traj_embeddings[max(0, i-5)]
        pos_reward, dir_reward = reward_model.get_vlm_reward(reward_model.proj_state, embedding, traj_embeddings[0], prev_embedding, reward_model.dir_proj_state)
        pos_rewards.append(pos_reward)
        dir_rewards.append(dir_reward * lam)

    return pos_rewards, dir_rewards


def create_video(model_names, frames = None, run_name = None, traj_num = 0, successful = True, filepath = ""):
    # get frames
    if frames is None and run_name is None:
        raise Exception("Frames or run name must be specified.")
    elif frames is None:
        frames = pickle.load(open(f"trajs/{run_name}/" + ("successful" if successful else "failed") + f"/traj{traj_num}", "rb"))["frames"]

    traj_embeddings = get_embeddings(frames)
    norms = [np.linalg.norm(traj_embeddings[max(0, i-5)] - traj_embeddings[i]) for i in range(len(traj_embeddings))]

    pos_reward_lists, dir_reward_lists, model_params = [], [], []
    for model_name in model_names:
        pos_rewards, dir_rewards = get_rewards(model_name, traj_embeddings)
        pos_reward_lists.append(pos_rewards)
        dir_reward_lists.append(dir_rewards)
        model_params.append(get_params(model_name))

    min_pos_reward, max_pos_reward = np.min(pos_reward_lists), np.max(pos_reward_lists)
    min_dir_reward, max_dir_reward = np.min(dir_reward_lists), np.max(dir_reward_lists)
    min_norm, max_norm = np.min(norms), np.max(norms)

    pos_margin = (max_pos_reward - min_pos_reward) * 0.05
    dir_margin = (max_dir_reward - min_dir_reward) * 0.05
    norm_margin = (max_norm - min_norm) * 0.05


    if len(filepath) > 0 and filepath[-1] != "/":
        filepath = filepath + "/"
    filepath = "videos/" + filepath
    os.makedirs(filepath, exist_ok=True)

    video_name = model_params[0][2] + "_a"
    for alg_version, _, _ in model_params:
        video_name = video_name + f"{alg_version}-"
    video_name = video_name[:-1] + "_l"
    for _, lam, _ in model_params:
        video_name = video_name + f"{lam}-"
    video_num = sum([1 if video_name[:-1] in filename else 0 for filename in os.listdir(filepath)])
    video_name = video_name[:-1] + f"_{video_num}.mp4"
    
    out = cv2.VideoWriter(filepath + video_name, cv2.VideoWriter_fourcc(*"mp4v"), 24, (2000, 1000), True)
    
    pb = tqdm.tqdm(total=len(frames))
    colors = list(mpl.colors.TABLEAU_COLORS.values())

    for i, frame in enumerate(frames):
        frame = frame[:,:,[2, 1, 0]]
        frame = cv2.resize(frame, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)

        fig = plt.figure(figsize=(8, 3))
        fig.tight_layout(pad=0)
        plt.xlim((-10, 510))
        plt.ylim((min_pos_reward - pos_margin, max_pos_reward + pos_margin))
        for model_i, pos_rewards, (alg_version, lam, _) in zip(np.arange(len(model_names)), pos_reward_lists, model_params):
            plt.plot(np.arange(len(pos_rewards)), pos_rewards, linewidth=1, linestyle="--", color=colors[model_i], alpha=0.4)
            plt.plot(np.arange(i+1), pos_rewards[:i+1], linewidth=1, label=f"a{alg_version} l{lam}", color=colors[model_i])
        plt.title("positional reward")
        plt.legend()
        fig.canvas.draw()
        pos_reward_image = get_img_from_fig(fig, (1000, 333))
        plt.close(fig)

        fig = plt.figure(figsize=(8, 3))
        fig.tight_layout(pad=0)
        plt.xlim((-10, 510))
        plt.ylim((min_dir_reward - dir_margin, max_dir_reward + dir_margin))
        for model_i, dir_rewards, (alg_version, lam, _) in zip(np.arange(len(model_names)), dir_reward_lists, model_params):
            plt.plot(np.arange(len(dir_rewards)), dir_rewards, linewidth=1, linestyle="--", color=colors[model_i], alpha=0.4)
            plt.plot(np.arange(i+1), dir_rewards[:i+1], linewidth=1, label=f"a{alg_version} l{lam}", color=colors[model_i])
        plt.title("directional reward")
        plt.legend()
        fig.canvas.draw()
        dir_reward_image = get_img_from_fig(fig, (1000, 333))
        plt.close(fig)

        fig = plt.figure(figsize=(8, 3))
        fig.tight_layout(pad=0)
        plt.xlim((-10, 510))
        plt.ylim((min_norm - norm_margin, max_norm + norm_margin))
        plt.plot(np.arange(len(norms)), norms, linewidth=1, linestyle="--", color="gray")
        plt.plot(np.arange(i+1), norms[:i+1], linewidth=1)
        plt.title("2 norms (k=5)")
        fig.canvas.draw()
        norm_image = get_img_from_fig(fig, (1000, 334))
        plt.close(fig)

        reward_plot_image = np.concatenate((pos_reward_image, dir_reward_image, norm_image), axis=0)[:,:,[2,1,0]]

        frame = np.concatenate((frame, reward_plot_image), axis=1)
        out.write(frame)
        pb.update(1)
    
    pb.close()
    out.release()