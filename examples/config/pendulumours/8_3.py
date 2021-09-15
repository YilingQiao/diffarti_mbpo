params = {
    'type': 'GLearnOld',
    'universe': 'gym',
    'domain': 'pendulumours', ## mbpo/env/ant.py
    'task': 'v2',

    'log_dir': './logs/pendulum/',
    'exp_name': 'glearn_link_3',

    'kwargs_env': {
        'n_links': 3,
        "use_term": 0
    },

    'kwargs': {
        'n_epochs': 300, ## 20k steps
        'epoch_length': 100,
        'train_every_n_steps': 1,
        'n_train_repeat': 5,
        'eval_render_mode': None,
        'eval_n_episodes': 10,
        'eval_deterministic': True,

        'discount': 1,
        'tau': 5e-3,
        'reward_scale': 1.0,
        'use_new': False,

        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -1e-2,
        'max_model_t': None,
        'rollout_schedule': [1, 15, 1, 10],
        'n_initial_exploration_steps': 500
    }
}
