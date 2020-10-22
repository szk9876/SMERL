import sys
sys.path.append('/iris/u/szk/ICML_FINAL_CODE/multiworld')

from multiworld.core.multitask_env import MultitaskEnv
from rlkit.envs.wrappers import ProxyEnv
from rlkit.util.io import load_local_or_remote_file
import numpy as np
from rlkit.torch.torch_rl_algorithm import np_to_pytorch_batch
from rlkit.torch import pytorch_util as ptu
import copy
from gym.spaces import Discrete, Dict

class DiscriminatorWrappedEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
            disc,
            mode='train',
            reward_params=None,
            unsupervised_reward_weight=0.,
            reward_weight=0.,
            noise_scale=0.,
    ):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        if type(disc) is str:
            self.disc = load_local_or_remote_file(disc)
        else:
            self.disc = disc
        self._num_skills = self.disc.num_skills
        self._p_z = np.full(self._num_skills, 1.0 / self._num_skills)
        self.task = {'context': -1}
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get('type', 'diayn')
        # TODO: Check that TIAYN reward is set properly.
        spaces = copy.deepcopy(self.wrapped_env.observation_space.spaces)
        spaces['context'] = Discrete(n=self._num_skills)
        self.observation_space = Dict(spaces)
        self.unsupervised_reward_weight = unsupervised_reward_weight
        self.reward_weight = reward_weight
        self.noise_scale = noise_scale

        assert self.reward_type == 'wrapped_env' or self.reward_type == 'diayn' or self.reward_type == 'wrapped_env + diayn' or self.reward_type == 'tiayn' or self.reward_type == 'wrapped_env + tiayn'
        # Commented out June 1
        # if 'wrapped_env' not in self.reward_type:
        #    assert self.reward_weight == 0.

    def reset(self):
        # import pdb; pdb.set_trace()
        obs = self.wrapped_env.reset()
        z = self._sample_z(1)
        task = {}
        task['context'] = z
        self.task = task
        return self._update_obs(obs)

    def step(self, action):
        if self.noise_scale > 0:
            action += np.random.normal(loc=0.0, scale=self.noise_scale)
        obs, env_reward, done, info = self.wrapped_env.step(action)
        # Commented out June 1
        # if 'wrapped_env' not in self.reward_type:
        #    env_reward = 0.
        new_obs = self._update_obs(obs)
        # Commented out June 1
        # if self.reward_type == 'tiayn' or self.reward_type == 'diayn':
        #     done = False
        env_reward = self.reward_weight * env_reward
        
        return new_obs, env_reward, done, info

    def _sample_z(self, batch_size):
        """Samples z from p(z), using probabilities in self._p_z."""
        return np.random.choice(self._num_skills, size=batch_size, replace=True, p=self._p_z)

    def _update_obs(self, obs):
        obs = {**obs, **self.task}
        return obs

    def get_goal(self):
        raise NotImplementedError()

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def update_rewards(self, path):
        observations = path['full_observations'][:len(path['observations'])]

        path['environment_rewards'] = np.copy(path['rewards'])
        path['unsupervised_rewards'] = np.zeros_like(path['rewards'])

        # Compute TIAYN rewards.
        if 'tiayn' in self.reward_type:
            raise NotImplementedError
            # trajectory_reward = 0.
            # for i in range(len(observations) - 1):
                # trajectory_reward += self.compute_discriminator_rewards(
                #    np.array([observations[i+1]['observation']]), np.array([observations[i]['context']]), np.array([observations[i]['observation']]))[0]
            for i in range(len(path['rewards'])):
                path['unsupervised_rewards'][i] = trajectory_reward
        
        # Compute DIAYN rewards.
        elif 'diayn' in self.reward_type:
            for i in range(len(path['rewards'])):
                path['unsupervised_rewards'][i] = self.compute_discriminator_rewards(
                    np.array([observations[i]['observation']]), np.array([observations[i]['context']]))

    def compute_discriminator_rewards(self, next_ob, skill, ob=None):
        if 'diayn' in self.reward_type:
            inputs = next_ob
        elif 'tiayn' in self.reward_type and ob is not None:
            inputs = np.concatenate((ob, next_ob), axis=-1)
        else:
            return 0.

        cross_entropy = self.disc.evaluate_cross_entropy(inputs=inputs, labels=skill)
        p_z = self._p_z[skill.astype(np.int)]
        log_p_z = np.log(p_z)

        assert cross_entropy.shape == log_p_z.shape
        reward = -1 * cross_entropy - log_p_z
        return self.unsupervised_reward_weight * reward

    def compute_discriminator_rewards_from_paths(self, obs, path):
        if 'diayn' in self.reward_type:
            return self.compute_discriminator_rewards(obs['observation'], obs['context'])
        elif 'tiayn' in self.reward_type:
            # batch_size = len(next_obs_paths)
            # trajectory_rewards = np.zeros((batch_size, 1))
            # for i in range(len(next_obs_paths[0])):
            #    trajectory_rewards += (1. - terminal_paths[:, i]) * self.compute_discriminator_rewards(
            #        next_obs_paths[:, i], context_paths[:, i], obs_paths[:, i])
            # return trajectory_rewards
            raise NotImplementedError
        return None

    def compute_rewards(self, actions, obs):
        return 0.

    def sample_goals(self, batch_size):
        return {
            'context': self._sample_z(batch_size)
        }

    def sample_goal(self):
        goals = self.sample_goals(1)
        return self.unbatchify_dict(goals, 0)

    def fit(self, replay_buffer):
        return self.disc.fit(replay_buffer)

    @property
    def context_dim(self):
        return self._num_skills


