params = {
    'type': 'GLearn',
    'universe': 'gym',
    'domain': 'antours', ## mbpo/env/ant.py
    'task': 'v2',

    'log_dir': './logs/ant/',
    'exp_name': 'glearn_ant_ext9_d1e-2_random',
    'kwargs_env': {
        'enable_qlimit': True,
        'need_jac': True,
        'random_init': 1,
    },

    'kwargs': {
        'n_epochs': 100,
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'num_ext': 9,
        'maxdelta': 1e-2,
        'n_initial_exploration_steps':5000,
        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -4,
        'max_model_t': None,
        'rollout_schedule': [20, 100, 1, 25],
    }
}
