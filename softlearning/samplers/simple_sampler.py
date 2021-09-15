from collections import defaultdict

import numpy as np

from .base_sampler import BaseSampler


class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              terminal,
                              next_observation,
                              latent,
                              info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': np.array([reward]),
            'terminals': np.array([terminal]),
            'latents': latent,
            'next_observations': next_observation
        }
        for k, v in info.items():
            processed_observation[k] = v

        return processed_observation

    def sample(self, num_ext=0, maxdelta=0.01):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action, latent = self.policy.actions_np_ours([
            self.env.convert_to_active_observation(
                self._current_observation)[None]
        ])
        action = action[0]
        latent = latent[0]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            latent=latent,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        current_samples = {field_name: [values]
            for field_name, values in processed_sample.items()}
        for extra in range(num_ext):
            dx = np.random.uniform(low=-maxdelta, high=maxdelta, size=processed_sample['actions'].shape)
            new_sample = {key: value + 0 for key, value in processed_sample.items()}
            new_sample['actions'] += dx
            new_sample['next_observations'][processed_sample['jacind']] += np.matmul(processed_sample['jac'], dx)
            new_sample['rewards'] += np.matmul(processed_sample['jacr'], dx)
            for key, value in new_sample.items():
                current_samples[key].append(value)
        current_samples_np = {
            field_name: np.array(values)
            for field_name, values in current_samples.items()
        }
        self.pool.add_samples(current_samples_np)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            # print('obsshape=',last_path['infos'].shape)
            # print('obsshape=',len(self._current_path['observations']))
            # print('obsshape=',np.array(self._current_path['observations']).shape)
            # print('obsshape=',self._current_path['observations'])
            # import sys;sys.exit(0)
            # self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch( #
            batch_size, observation_keys=observation_keys, **kwargs)

    def last_n_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.last_n_batch( 
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics
