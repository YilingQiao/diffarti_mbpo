import gym

MBPO_ENVIRONMENT_SPECS = (
    {
        'id': 'AntTruncatedObs-v2',
        'entry_point': (f'mbpo.env.ant:AntTruncatedObsEnv'),
    },
    {
        'id': 'pendulumours-v2',
        'entry_point': (f'mbpo.env.pendulumours:PendulumOursEnv'),
    },
    {
        'id': 'laikago-v2',
        'entry_point': (f'mbpo.env.laikago:LaikagoEnv'),
    },
    {
        'id': 'antours-v2',
        'entry_point': (f'mbpo.env.antours:AntOursEnv'),
    },
    {
        'id': 'HumanoidTruncatedObs-v2',
        'entry_point': (f'mbpo.env.humanoid:HumanoidTruncatedObsEnv'),
    },
    {
        'id': 'cartpoleours-v2',
        'entry_point': (f'mbpo.env.cartpoleours:CartPoleOursEnv'),
    },
)

def register_mbpo_environments():
    for mbpo_environment in MBPO_ENVIRONMENT_SPECS:
        gym.register(**mbpo_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MBPO_ENVIRONMENT_SPECS)

    return gym_ids
