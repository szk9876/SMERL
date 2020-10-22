import numpy as np
from gym.spaces import Dict

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import flatten_n, flatten_dict
from gym.spaces import Box, Discrete, Tuple


class ObsDictPathReplayBuffer(ReplayBuffer):
    """
    Replay buffer for
        - keeping track of information about paths/trajectories
        - environments where observations are dictionaries
    max_replay_buffer_size: maximum number of paths
    """

    def __init__(
            self,
            max_size,
            max_path_length,
            env,
            subopt_return_threshold,
            environment_reward_weight,
            unsupervised_reward_weight,
            observation_key='observation',
            context_key='context',
            state_observation_key='state_observation', 
            state_desired_goal_key='state_desired_goal',
            achieved_goal_key='achieved_goal',
            desired_goal_key='desired_goal'
    ):
        assert isinstance(env.observation_space, Dict)
        self.max_size = max_size
        self.max_path_length = max_path_length
        self.env = env
        self.ob_keys_to_save = [
            observation_key,
            context_key,
            state_observation_key,
        ]
        self.subopt_return_threshold = subopt_return_threshold
        self.environment_reward_weight = environment_reward_weight
        self.unsupervised_reward_weight = unsupervised_reward_weight
        self.observation_key = observation_key
        self.context_key = context_key

        self._action_dim = env.action_space.low.size

        self._actions = np.zeros((max_size, max_path_length, self._action_dim))
        self._environment_rewards = np.zeros((max_size, max_path_length, 1))
        self._terminals = np.ones((max_size, max_path_length, 1), dtype='uint8')
        self._episode_lengths = np.zeros((max_size))
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        for key in self.ob_keys_to_save:
            assert key in self.ob_spaces, "Key not found in the observation space: {}".format(key)
            dtype = np.float64
            if type(self.ob_spaces[key]) is Box:
                dsize = self.ob_spaces[key].low.size
            elif type(self.ob_spaces[key]) is Discrete:
                dsize = 1
            else:
                raise NotImplementedError

            self._obs[key] = np.zeros((max_size, max_path_length, dsize), dtype=dtype)
            self._next_obs[key] = np.zeros((max_size, max_path_length, dsize), dtype=dtype)

        self._top = 0
        self._size = 0

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        next_obs = path["next_observations"]
        rewards = path["rewards"]
        terminals = path["terminals"]
        path_len = len(terminals)

        # assert path_len == self.max_path_length

        actions = flatten_n(actions)
        obs = flatten_dict(obs, self.ob_keys_to_save)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save)

        self._actions[self._top][:path_len] = actions

        rewards = np.squeeze(rewards)
        rewards = np.expand_dims(rewards, -1)
        self._environment_rewards[self._top][:path_len] = rewards

        self._terminals[self._top][:path_len] = terminals
        if len(np.argwhere(terminals)):
            episode_length = np.argwhere(terminals)[:, 0][0] + 1
            assert episode_length == path_len
        else:
            episode_length = self.max_path_length
        self._episode_lengths[self._top] = episode_length

        for key in self.ob_keys_to_save:
            self._obs[key][self._top][:path_len] = obs[key]
            self._next_obs[key][self._top][:path_len] = next_obs[key]

        self._actions[self._top][:path_len] = actions

        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self.max_size
        if self._size < self.max_size:
            self._size += 1

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size * self.max_path_length

    def random_batch(self, batch_size):
        path_indices = np.random.randint(low=0, high=self._size, size=batch_size)
        time_indices = np.random.randint(low=0, high=self.max_path_length, size=batch_size)
        return self._get_batch(path_indices, time_indices)

    def random_paths(self, batch_size, path_indices=None):
        if path_indices is None:
            path_indices = np.random.randint(low=0, high=self._size, size=batch_size)
        obs_dict = self._path_obs_dict(path_indices)
        next_obs_dict = self._path_next_obs_dict(path_indices)

        time_indices = np.zeros(shape=batch_size, dtype=int)
        # Labels for the discriminator.
        contexts = self._batch_obs_dict(path_indices, time_indices)['context']
        terminals = self._terminals[path_indices]
        episode_lengths = np.expand_dims(self._episode_lengths[path_indices], axis=-1)       

        batch = {
            'observations': obs_dict[self.observation_key],
            'terminals': self._terminals[path_indices],
            'next_observations': next_obs_dict[self.observation_key],
            'context': contexts, 
            'path_terminals': terminals,
            'episode_lengths': episode_lengths
        }
        return batch

    def recent_batch(self, batch_size, window):
        if self._size < window:
            path_index_low = 0
            path_index_high = self._size
        else:
            path_index_low = self._top - window
            path_index_high = self._top
        path_indices = np.random.randint(low=path_index_low, high=path_index_high, size=batch_size)
        time_indices = np.random.randint(low=0, high=self.max_path_length, size=batch_size)
        return self._get_batch(path_indices, time_indices)

    def _get_batch(self, path_indices, time_indices):
        # Determine the indices of terminal states for each path.
        # Select the time indices in the range [0, episode_length).
        terminal_indices = np.zeros_like(time_indices)
        for i in range(len(path_indices)):
            episode_length = self._episode_lengths[path_indices[i]]
            time_indices[i] = np.random.randint(low=0, high=episode_length)
            terminal_indices[i] = episode_length - 1

        obs_dict = self._batch_obs_dict(path_indices, time_indices)
        next_obs_dict = self._batch_next_obs_dict(path_indices, time_indices)
        actions = self._actions[path_indices, time_indices]

        # Get environment rewards for the selected paths and time indices. 
        # Get environment returns for the selected paths. 
        environment_rewards = self._environment_rewards[path_indices, time_indices]
        environment_returns = np.sum(self._environment_rewards[path_indices], axis=-2)
        assert environment_returns.shape == (len(path_indices), 1)
        
        path_batch = None
        unsupervised_rewards = self.env.compute_discriminator_rewards_from_paths(next_obs_dict, path_batch)

        if self.unsupervised_reward_weight == 0.:
            rewards = environment_rewards
        elif self.environment_reward_weight == 0.:
            rewards = unsupervised_rewards
        else:
            # Masked version of adding the unsupervised rewards.
            rewards = environment_rewards + (environment_returns >= self.subopt_return_threshold) * unsupervised_rewards

        batch = {
            'observations': obs_dict[self.observation_key],
            'actions': actions,
            'rewards': rewards,
            'terminals': self._terminals[path_indices, time_indices],
            'next_observations': next_obs_dict[self.observation_key],
            'context': obs_dict[self.context_key]
        }

        return batch

    def _path_obs_dict(self, path_indices):
        return {
            key: self._obs[key][path_indices]
            for key in self.ob_keys_to_save
        }

    def _path_next_obs_dict(self, path_indices):
        return {
            key: self._obs[key][path_indices]
            for key in self.ob_keys_to_save
        }
    
    def _batch_obs_dict(self, path_indices, time_indices):
        return {
            key: self._obs[key][path_indices, time_indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, path_indices, time_indices):
        return {
            key: self._next_obs[key][path_indices, time_indices]
            for key in self.ob_keys_to_save
        }

    def get_trajectories(self):
        """Returns 3D data: (num_paths, episode_length, num_features)."""
        indices = np.s_[self._top-self._size : self._top]
        obs_dict = {key: self._obs[key][indices] for key in self.ob_keys_to_save}
        actions = self._actions[indices]
        return {'observations': obs_dict[self.observation_key],
                'actions': actions}

    def resample_tasks(self, env):
        num_tasks = self._size
        task_dict = env.sample_goals(num_tasks)
        context = task_dict['context']
        context = np.broadcast_to(context, [num_tasks, self.max_path_length, env.context_dim])
        indices = np.s_[self._top - self._size: self._top]
        self._obs[self.context_key][indices] = context

