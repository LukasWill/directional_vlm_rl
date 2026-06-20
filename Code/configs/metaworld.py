import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.env_name = "peg-insert-side-v2-goal-observable"
    config.camera_id = 2
    config.residual = False
    config.eval_episodes = 100
    config.start_timesteps = 10000
    config.max_timesteps = int(1e6)
    config.decay_timesteps = int(7.5e5)
    config.eval_freq = config.max_timesteps // 100
    config.log_freq = config.max_timesteps // 400
    config.ckpt_freq = config.max_timesteps // 10
    config.lr = 1e-4
    config.seed = 0
    config.tau = 0.01
    config.gamma = 0.99
    config.batch_size = 256
    config.hidden_dims = (256, 256)
    config.initializer = "orthogonal"
    config.exp_name = "furl"

    # relay
    config.relay_threshold = 2500
    config.expl_noise = 0.2

    # fine-tune
    config.rho = 0.05
    config.gap = 10
    config.crop = False
    config.l2_margin = 0.2 # 0.25
    config.cosine_margin = 0.1 # 0.25
    config.embed_buffer_size = 20000

    config.baseline = ""

    config.enable_logging = True
    config.wandb_project = "test-project"
    config.wandb_entity = ""

    config.runner = "remote"  # whether experiment ran locally or in remote cluster
    config.checkpoint = False

    config.save_first_stage = False
    config.load_first_stage = ""

    config.save_trajs = False

    config.save_cases = False
    
    config.save_negative = False
    config.save_positive = False
    config.load_positive = ""
    config.pos_traj_amount = 50


    config.alg_version = 0  # deprecated

    # positional reward function
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
    config.pos_alg_version = 0

    # directional reward function
    # 0 : no directional reward
    # 1 : only directional input, before heads
    # 2 : only directional input, after heads
    # 3 : concatenated input, s_i positional
    # 4 : concatenated input, s_i-s_0 positional
    # 5 : concatenated input, gb regularization positional (before head)
    config.dir_alg_version = 0

    # multiply by norms
    config.mult_norms = False

    # end run after collecting `config.stage1_eval` successful trajectories. 0 is normal run.
    config.stage1_eval = 0

    # delta features
    config.temp_coef = 5
    config.lambda_ = 1.0

    # goal-baseline regularization
    config.beta = 0.2

    config.group = "standard"

    config.no_positional_reward = False

    return config
