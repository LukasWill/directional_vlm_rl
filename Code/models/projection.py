import jax
import jax.numpy as jnp
import functools
import optax
import orbax.checkpoint as ocp

from flax import linen as nn
from flax.training import train_state
from models import MLP

import numpy as np


class Projection(nn.Module):
    def setup(self):
        self.text_encoder = MLP(hidden_dims=(256, 64), activate_final=False)
        self.image_encoder = MLP(hidden_dims=(256, 64), activate_final=False)

    def __call__(self, text_embedding, image_embedding):
        proj_text_embedding = self.text_encoder(text_embedding)
        proj_image_embedding = self.image_encoder(image_embedding)
        return proj_text_embedding, proj_image_embedding

    def encode_image(self, image_embeddings):
        return self.image_encoder(image_embeddings)

    def encode_text(self, text_embedding):
        return self.text_encoder(text_embedding)


class RewardModel:
    def __init__(self,
                 config,
                 seed: int = 42,
                 lr: float = 1e-4,
                 margin: float = 0.1,
                 emb_dim: int = 1024,
                 ckpt_dir: str = None,
                 text_embedding: jnp.ndarray = None,
                 baseline_embedding: jnp.ndarray = None,
                 goal_embedding: jnp.ndarray = None):
        
        self.config = config
        self.pos_reward = config.pos_alg_version
        self.dir_reward = config.dir_alg_version
        self.mult_norms = config.mult_norms
        self.lr = lr
        self.margin = margin
        self.lambda_ = config.lambda_
        self.beta = config.beta
        self.text_embedding = text_embedding
        self.baseline_embedding = baseline_embedding
        self.goal_embedding = goal_embedding
        self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng, 2)
        dummy_emb = jnp.ones([1, emb_dim], dtype=jnp.float32)

        self.proj = Projection()
        proj_params = self.proj.init(key,
                                     jnp.ones([1, 1024], dtype=jnp.float32),
                                     dummy_emb)["params"]
        self.proj_state = train_state.TrainState.create(
            apply_fn=self.proj.apply,
            params=proj_params,
            tx=optax.adam(lr))

        # directional input is 1024
        if self.dir_reward in [1, 2]:
            self.dir_proj = Projection()
            dir_proj_params = self.dir_proj.init(key,
                                        jnp.ones([1, 1024], dtype=jnp.float32),
                                        dummy_emb)["params"]
            self.dir_proj_state = train_state.TrainState.create(
                apply_fn=self.dir_proj.apply,
                params=dir_proj_params,
                tx=optax.adam(lr))
        
        # directional input is 2048
        elif self.dir_reward in [3, 4, 5]:
            self.dir_proj = Projection()
            dir_proj_params = self.dir_proj.init(key,
                                        jnp.ones([1, 1024], dtype=jnp.float32),
                                        jnp.ones([1, 2048], dtype=jnp.float32))["params"]
            self.dir_proj_state = train_state.TrainState.create(
                apply_fn=self.dir_proj.apply,
                params=dir_proj_params,
                tx=optax.adam(lr))
            
        else:
            self.dir_proj = None
            self.dir_proj_state = None

        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
            self.checkpointer = ocp.StandardCheckpointer()

    def normalize_embeddings(self, embeddings):
        return embeddings / jnp.maximum(jnp.linalg.norm(embeddings, axis=1)[:, jnp.newaxis], 1e-7)
    

    # EMBEDDING PROJECTION FUNCTIONS

    def img_projection(self, proj, params, embeddings):
        return proj.apply(
            {"params": params}, embeddings,
            method=proj.encode_image)

    def text_projection(self, proj, params, embeddings):
        return proj.apply(
            {"params": params}, embeddings,
            method=proj.encode_text)

    # ALG_VERSION LOOKUP (deprecated)
    # - 0  : Original FuRL
    # - 1  : Delta features, after heads
    # - 2  : Delta features, before heads
    # - 3  : Double network, directional input before heads
    # - 4  : Double network, directional input before heads, multiplied by norms
    # - 5  : Unused
    # - 6  : Unused
    # - 7  : Double network, directional input after heads
    # - 8  : Double network, directional input after heads, multiplied by norms
    # - 9  : Double network, concatenated input
    # - 10 : Double network, concatenated input, multiplied by norms
    # - 11 : Goal-baseline regularization
    # - 12 : Goal-baseline regularization, subtract initial state


    # POSITIONAL REWARD FUNCTIONS

    # 0 : original FuRL
    # 1 : delta features, before heads
    # 2 : delta features, before heads, without subtracting init state
    # 3 : delta features, before heads, without subtracting baseline
    # 4 : delta features, after heads
    # 5 : delta features, after heads, without subtracting init state
    # 6 : delta features, after heads, without subtracting baseline
    # 7 : gb regularization, reg after heads (standard)
    # 8 : gb regularization, reg after heads, subtract initial state
    # 9 : gb regularization, reg before heads
    # 10 : gb regularization, reg before heads, subtract initial state

    # original FuRL
    def furl_reward(self, proj_params, embeddings):
        proj_embeddings = self.img_projection(self.proj, proj_params, embeddings)
        proj_text_embedding = self.text_projection(self.proj, proj_params, self.text_embedding)

        return optax.cosine_similarity(proj_embeddings, proj_text_embedding)


    # delta features
    
    def df_reward_before(self, proj_params, embeddings, init_embedding):
        # image embeddings, with or without initial state subtracted
        embeddings = embeddings - (init_embedding if self.pos_reward in [1, 3] else 0)
        proj_embeddings = self.img_projection(self.proj, proj_params, embeddings)

        # text embedding, goal with or without baseline subtracted
        text_embedding = self.text_embedding - (self.baseline_embedding if self.pos_reward in [1, 2] else 0)
        proj_text_embedding = self.text_projection(self.proj, proj_params, text_embedding)

        return optax.cosine_similarity(proj_embeddings, proj_text_embedding, 1e-7)


    def df_reward_after(self, proj_params, embeddings, init_embedding):
        # image embeddings, with or without initial state subtracted
        proj_embeddings = self.img_projection(self.proj, proj_params, embeddings)
        if self.pos_reward in [4, 6]:
            proj_embeddings = proj_embeddings - self.img_projection(self.proj, proj_params, init_embedding)
        
        # text embedding, with or without baseline subtracted
        proj_text_embedding = self.text_projection(self.proj, proj_params, self.text_embedding)
        if self.pos_reward in [4, 5]:
            proj_text_embedding = proj_text_embedding - self.text_projection(self.proj, proj_params, self.baseline_embedding)

        return optax.cosine_similarity(proj_embeddings, proj_text_embedding, 1e-7)


    # goal baseline regularization
    # before: run embeddings through heads and then compute gb regularized reward
    # after: compute weighted sum of embeddings as in gb with goal and baseline in VLM embedding space,
    #        run this sum through image head, compute cosine similarity between projected goal embedding and output from image head

    def gb_reward_after(self, proj_params, embeddings, init_embedding):
        # image embeddings
        if self.pos_reward == 7:
            proj_embeddings = self.normalize_embeddings(self.img_projection(self.proj, proj_params, embeddings))
        else:
            proj_embeddings = self.img_projection(self.proj, proj_params, embeddings) - self.img_projection(self.proj, proj_params, init_embedding)
            proj_embeddings = self.normalize_embeddings(proj_embeddings)
            
        # text embeddings
        proj_text_embedding = self.normalize_embeddings(self.text_projection(self.proj, proj_params, self.text_embedding))
        proj_base_embedding = self.normalize_embeddings(self.text_projection(self.proj, proj_params, self.baseline_embedding))
        
        # g-b line, which the image embeddings will be projected onto for the regularization
        regularizer = proj_text_embedding - proj_base_embedding

        # projection onto g-b line
        projected_embeddings = (jnp.inner(proj_embeddings, regularizer) / jnp.inner(regularizer, regularizer)).reshape(-1, 1) * regularizer

        # weighted image embeddings by projection, creating regularizing effect
        regularized_embeddings = self.beta * projected_embeddings + (1 - self.beta) * proj_embeddings

        # distance to goal, shorter distance means higher reward
        diff_to_goal = regularized_embeddings - proj_text_embedding

        return 1 - (1 / 2) * (diff_to_goal * diff_to_goal).sum(axis=1)

    def gb_reward_before(self, proj_params, embeddings, init_embedding):
        if self.pos_reward == 10:
            embeddings = embeddings - init_embedding

        # g-b line in VLM embedding space
        regularizer = self.text_embedding - self.baseline_embedding

        # project VLM image embedding onto g-b before running through heads
        projected_embeddings = (jnp.inner(embeddings, regularizer) / jnp.inner(regularizer, regularizer)).reshape(-1, 1) * regularizer

        # weighted sum, regularized embeddings by projection term
        regularized_embeddings = self.beta * projected_embeddings + (1 - self.beta) * embeddings

        # run through image head, normalized because required by this formulation of cosine sim
        proj_reg_embeddings = self.normalize_embeddings(self.img_projection(self.proj, proj_params, regularized_embeddings))

        # projected text embedding, normalized for the same reason as before
        proj_text_embedding = self.normalize_embeddings(self.text_projection(self.proj, proj_params, self.text_embedding))

        # distance to goal, reward is higher with shorter distance
        diff_to_goal = proj_reg_embeddings - proj_text_embedding

        return 1 - (1 / 2) * (diff_to_goal * diff_to_goal).sum(axis=1)


    def get_positional_reward(self, proj_params, embeddings, init_embedding):
        if self.pos_reward == 0:
            return self.furl_reward(proj_params, embeddings)
        elif self.pos_reward in [1, 2, 3]:
            return self.df_reward_before(proj_params, embeddings, init_embedding)
        elif self.pos_reward in [4, 5, 6]:
            return self.df_reward_after(proj_params, embeddings, init_embedding)
        elif self.pos_reward in [7, 8]:
            return self.gb_reward_after(proj_params, embeddings, init_embedding)
        elif self.pos_reward in [9, 10]:
            return self.gb_reward_before(proj_params, embeddings, init_embedding)



    # DIRECTIONAL REWARD FUNCTIONS

    # 0 : no directional reward
    # 1 : only directional input, before heads
    # 2 : only directional input, after heads
    # 3 : concatenated input, s_i positional
    # 4 : concatenated input, s_i-s_0 positional
    # 5 : concatenated input, gb regularization positional (before head)

    def directional_input_reward_before(self, dir_proj_params, embeddings, prev_embeddings):
        directions = self.normalize_embeddings(embeddings - prev_embeddings)
        proj_embeddings = self.img_projection(self.dir_proj, dir_proj_params, directions)
        proj_text_embedding = self.text_projection(self.dir_proj, dir_proj_params, self.text_embedding - self.baseline_embedding)

        directional_reward = optax.cosine_similarity(proj_embeddings, proj_text_embedding, 1e-7)
        
        return directional_reward


    def directional_input_reward_after(self, dir_proj_params, embeddings, prev_embeddings):
        proj_embeddings = self.img_projection(self.dir_proj, dir_proj_params, embeddings)
        proj_prev_embeddings = self.img_projection(self.dir_proj, dir_proj_params, prev_embeddings)

        proj_text_embedding = self.text_projection(self.dir_proj, dir_proj_params, self.text_embedding)
        proj_base_embedding = self.text_projection(self.dir_proj, dir_proj_params, self.baseline_embedding)

        directional_reward = optax.cosine_similarity(
            proj_embeddings - proj_prev_embeddings, 
            proj_text_embedding - proj_base_embedding,
            1e-7)
        
        return directional_reward
        
    def concatenated_input_reward(self, dir_proj_params, positions, directions):
        positions = self.normalize_embeddings(positions)
        directions = self.normalize_embeddings(directions)
        proj_embeddings = self.img_projection(self.dir_proj, dir_proj_params, jnp.concatenate((positions, directions), axis=1))
        proj_text_embeddings = self.text_projection(self.dir_proj, dir_proj_params, self.text_embedding - self.baseline_embedding)

        directional_reward = optax.cosine_similarity(proj_embeddings, proj_text_embeddings, 1e-7)
        
        return directional_reward

    def concatenated_gb_reward(self, dir_proj_params, embeddings, directions):
        directions = self.normalize_embeddings(directions)
        # g-b line in VLM embedding space
        regularizer = self.text_embedding - self.baseline_embedding

        # project VLM image embedding before running through heads
        projected_embeddings = (jnp.inner(embeddings, regularizer) / jnp.inner(regularizer, regularizer)).reshape(-1, 1) * regularizer

        # weighted sum, regularized embeddings by projection term
        positions = self.normalize_embeddings(self.beta * projected_embeddings + (1 - self.beta) * embeddings)

        proj_embeddings = self.img_projection(self.dir_proj, dir_proj_params, jnp.concatenate((positions, directions), axis=1))
        proj_text_embedding = self.text_projection(self.dir_proj, dir_proj_params, self.text_embedding - self.baseline_embedding)

        return optax.cosine_similarity(proj_embeddings, proj_text_embedding)
    
    def concatenated_reward(self, dir_proj_params, embeddings, directions, init_embedding):
        if self.dir_reward == 3:
            return self.concatenated_input_reward(dir_proj_params, embeddings, directions)
        elif self.dir_reward == 4:
            return self.concatenated_input_reward(dir_proj_params, embeddings - init_embedding, directions)
        elif self.dir_reward == 5:
            return self.concatenated_gb_reward(dir_proj_params, embeddings, directions)
        
    def get_directional_reward(self, dir_proj_params, embeddings, prev_embeddings, init_embedding):
        if self.dir_reward == 1:
            directional_reward = self.directional_input_reward_before(dir_proj_params, embeddings, prev_embeddings)
        elif self.dir_reward == 2:
            directional_reward = self.directional_input_reward_after(dir_proj_params, embeddings, prev_embeddings)
        elif self.dir_reward in [3, 4, 5]:
            directional_reward = self.concatenated_reward(dir_proj_params, embeddings, embeddings - prev_embeddings, init_embedding)
        else:
            return 0
        
        return directional_reward


    @functools.partial(jax.jit, static_argnames=("self"))
    def get_vlm_reward(self, proj_state, img_embeddings, init_embedding=None, prev_embeddings=None, dir_proj_state=None):

        is_directional = self.dir_reward > 0

        positional_reward = self.get_positional_reward(proj_state.params, img_embeddings, init_embedding)
        directional_reward = self.get_directional_reward(dir_proj_state.params if is_directional else None, img_embeddings, prev_embeddings, init_embedding)
        if self.mult_norms:
            directional_reward *= jnp.linalg.norm(img_embeddings - prev_embeddings, axis=1)

        return positional_reward, directional_reward
    

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_pos_dir_step(self,
                       pos_embeddings,
                       neg_embeddings,
                       dir_proj_state,
                       prev_pos_embeddings,
                       prev_neg_embeddings,
                       init_embedding):

        def loss_fn(params):
            if self.dir_reward in [3, 4, 5]:
                # from positive trajectories
                pos_cosine = self.concatenated_reward(params, pos_embeddings, pos_embeddings - prev_pos_embeddings, init_embedding)
                # from negative trajectories
                neg_cosine = self.concatenated_reward(params, neg_embeddings, neg_embeddings - prev_neg_embeddings, init_embedding)
                # same position, opposite directions
                oppo_pos_cosine = self.concatenated_input_reward(params, pos_embeddings, prev_pos_embeddings - pos_embeddings)
                # same position, direction towards goal state
                pref_neg_cosine = self.concatenated_input_reward(params, neg_embeddings, self.goal_embedding - neg_embeddings)

            else:
                pos_cosine = self.get_directional_reward(params, pos_embeddings, prev_pos_embeddings, init_embedding)
                neg_cosine = self.get_directional_reward(params, neg_embeddings, prev_neg_embeddings, init_embedding)
                # opposite direction
                oppo_pos_cosine = self.get_directional_reward(params, prev_pos_embeddings, pos_embeddings, init_embedding)
                # towards goal state
                pref_neg_cosine = self.get_directional_reward(params, self.goal_embedding, neg_embeddings, init_embedding)

            if self.mult_norms:
                pos_norms = jnp.linalg.norm(pos_embeddings - prev_pos_embeddings, axis=1)
                pos_cosine *= pos_norms
                oppo_pos_cosine *= pos_norms

                neg_norms = jnp.linalg.norm(neg_embeddings - prev_neg_embeddings, axis=1)
                neg_cosine *= neg_norms
                pref_neg_cosine *= neg_norms

            # maximize reward in actual pos direction, minimize in opposite direction
            pos_loss = (oppo_pos_cosine - pos_cosine).mean()

            # maximize reward in direction towards goal, minimize in actual neg direction
            neg_loss = (neg_cosine - pref_neg_cosine).mean()

            total_loss = pos_loss + neg_loss

            return total_loss, {
                "d_loss" : total_loss,
                "d_pos_loss" : pos_loss,
                "d_neg_loss" : neg_loss,
                "d_pos_cosine" : pos_cosine,
                "d_neg_cosine" : neg_cosine,
                "d_oppo_pos_cosine" : oppo_pos_cosine,
                "d_pref_neg_cosine" : pref_neg_cosine,
            }

        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)        
        (_, log_info), grad = grad_fn(dir_proj_state.params)
        new_dir_proj_state = dir_proj_state.apply_gradients(grads=grad)
        return new_dir_proj_state, log_info


    @functools.partial(jax.jit, static_argnames=("self"))
    def train_pos_step(self,
                       pos_embeddings,
                       neg_embeddings,
                       lag_embeddings,
                       proj_state,
                       init_embedding = None):

        def loss_fn(params):

            pos_cosine = self.get_positional_reward(params, pos_embeddings, init_embedding)
            neg_cosine = self.get_positional_reward(params, neg_embeddings, init_embedding)
            lag_cosine = self.get_positional_reward(params, lag_embeddings, init_embedding)

            # pos-neg: pos_cosine > lag_cosine > negative_cosine
            neg_mask = (neg_cosine - pos_cosine + self.margin) > 0
            neg_loss = neg_mask * (neg_cosine - pos_cosine)

            # pos-pos: pos_cosine > lag_cosine
            pos_mask = (lag_cosine - pos_cosine + self.margin) > 0
            pos_loss = pos_mask * (lag_cosine - pos_cosine)

            total_loss = pos_loss.mean() + neg_loss.mean()

            log_info = {
                "p_pos_cosine_mean" : pos_cosine.mean(),
                "p_pos_cosine" : pos_cosine,

                "p_neg_cosine_mean" : neg_cosine.mean(),
                "p_neg_cosine" : neg_cosine,

                "p_lag_cosine_mean" : lag_cosine.mean(),
                "p_lag_cosine" : lag_cosine,

                "p_neg_num": neg_mask.sum(),
                "p_neg_loss_mean": neg_loss.mean(),
                "p_neg_loss": neg_loss,

                "p_pos_num": pos_mask.sum(),
                "p_pos_loss_mean": pos_loss.mean(),
                "p_pos_loss": pos_loss,

                "p_loss" : total_loss
            }

            return total_loss, log_info
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)        
        (_, log_info), grad = grad_fn(proj_state.params)
        new_proj_state = proj_state.apply_gradients(grads=grad)
        return new_proj_state, log_info

    @functools.partial(jax.jit, static_argnames=("self"))
    def train_neg_dir_step(self,
                       embeddings,
                       dir_proj_state,
                       prev_embeddings = None,
                       init_embedding = None):

        def loss_fn(params):
            if self.dir_reward in [3, 4, 5]:
                # from negative trajectories
                neg_cosine = self.concatenated_reward(params, embeddings, embeddings - prev_embeddings, init_embedding)
                # same position, direction towards goal state
                pref_neg_cosine = self.concatenated_input_reward(params, embeddings, self.goal_embedding - embeddings)

            else:
                neg_cosine = self.get_directional_reward(params, embeddings, prev_embeddings, init_embedding)
                # towards goal state
                pref_neg_cosine = self.get_directional_reward(params, self.goal_embedding, embeddings, init_embedding)
            
            if self.mult_norms:
                neg_norms = jnp.linalg.norm(embeddings - prev_embeddings, axis=1)
                neg_cosine *= neg_norms
                pref_neg_cosine *= neg_norms
            
            # maximize reward in direction towards goal, minimize in actual neg direction
            neg_loss = (neg_cosine - pref_neg_cosine).mean()

            return neg_loss, {
                "d_loss" : neg_loss,
                "d_neg_loss" : neg_loss,
                "d_neg_cosine" : neg_cosine,
                "d_pref_neg_cosine" : pref_neg_cosine,
            }
    
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grad = grad_fn(dir_proj_state.params)
        new_dir_proj_state = dir_proj_state.apply_gradients(grads=grad)
        return new_dir_proj_state, log_info


    @functools.partial(jax.jit, static_argnames=("self"))
    def train_neg_step(self,
                       embeddings,
                       masks,
                       proj_state,
                       init_embedding = None):

        def loss_fn(params):
            neg_cosine = self.get_positional_reward(params, embeddings, init_embedding)

            cosine_delta = neg_cosine.reshape(-1, 1) - neg_cosine.reshape(1, -1)
  
            loss = (nn.relu(-cosine_delta + self.margin) * masks).sum(-1).mean()

            log_info = {
                "p_loss": loss, 
                "vlm_rewards": neg_cosine
            }
            return loss, log_info
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_info), grad = grad_fn(proj_state.params)
        new_proj_state = proj_state.apply_gradients(grads=grad)
        return new_proj_state, log_info


    def update_neg(self, batch, init_embedding = None):
        self.proj_state, log_info = self.train_neg_step(
            embeddings = batch.embeddings,
            masks = batch.masks,
            proj_state = self.proj_state,
            init_embedding = init_embedding)
        
        if self.dir_reward > 0:
            self.dir_proj_state, dir_log_info = self.train_neg_dir_step(
                embeddings = batch.embeddings,
                dir_proj_state = self.dir_proj_state,
                prev_embeddings = batch.prev_embeddings,
                init_embedding = init_embedding)

            if self.config.no_positional_reward:
                log_info["vlm_rewards"] = self.lambda_ * dir_log_info["d_neg_cosine"]
            else:
                log_info["vlm_rewards"] += self.lambda_ * dir_log_info["d_neg_cosine"]
            log_info.update(dir_log_info)

        return log_info

    def update_pos(self, batch, init_embedding = None):  
        
        self.proj_state, log_info = self.train_pos_step(
            pos_embeddings = batch.pos_embeddings,
            neg_embeddings = batch.neg_embeddings,
            lag_embeddings = batch.lag_embeddings,
            proj_state = self.proj_state,
            init_embedding = init_embedding) 
        
        if self.dir_reward > 0:
            self.dir_proj_state, dir_log_info = self.train_pos_dir_step(
                pos_embeddings = batch.pos_embeddings,
                neg_embeddings = batch.neg_embeddings,
                dir_proj_state = self.dir_proj_state,
                prev_pos_embeddings = batch.prev_pos_embeddings,
                prev_neg_embeddings = batch.prev_neg_embeddings,
                init_embedding = init_embedding)
            
            log_info.update(dir_log_info)
        
        return log_info

    def save(self, name):
        params = {"proj": self.proj_state.params}
        if self.dir_reward > 0:
            params["dir_proj"] = self.dir_proj_state.params

        self.checkpointer.save(f"{self.ckpt_dir}/{name}",
                               params,
                               force=True)

    def load(self, ckpt_dir: str, name):
        raw_restored = self.checkpointer.restore(f"{ckpt_dir}/{name}")
        proj_params = raw_restored["proj"]
        self.proj_state = train_state.TrainState.create(
            apply_fn=self.proj.apply,
            params=proj_params,
            tx=optax.adam(self.lr))
        if self.dir_reward > 0:
            dir_proj_params = raw_restored["dir_proj"]
            self.dir_proj_state = train_state.TrainState.create(
                apply_fn=self.proj.apply,
                params=dir_proj_params,
                tx=optax.adam(self.lr))