from multiworld.core.multitask_env import MultitaskEnv
from rlkit.envs.wrappers import ProxyEnv
from rlkit.util.io import load_local_or_remote_file
import numpy as np
from rlkit.torch.torch_rl_algorithm import np_to_pytorch_batch
from rlkit.torch import pytorch_util as ptu
import copy
from gym.spaces import Discrete, Dict

class ClustererWrappedEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
            clusterer,
            mode='train',
            reward_params=None,
    ):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        if type(clusterer) is str:
            self.clusterer = load_local_or_remote_file(clusterer)
        else:
            self.clusterer = clusterer
        self._num_clusters = self.clusterer.num_clusters
        self.task = {}
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get('type', 's_given_z')

        spaces = copy.deepcopy(self.wrapped_env.observation_space.spaces)
        spaces['context'] = Discrete(n=self._num_clusters)
        self.observation_space = Dict(spaces)

        assert self.reward_type == 'wrapped_env' or self.reward_type == 's_given_z'

        self._set_clusterer_attributes()

    def reset(self):
        obs = self.wrapped_env.reset()

        task_ind, context = self._sample_task(1)
        task = {'task_ind': task_ind,
                'context': context}
        self.task = task
        return self._update_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        reward = self.compute_reward(action,
                                     {'observation': new_obs['observation'],
                                      'context': new_obs['context']})
        return new_obs, reward, done, info

    def _sample_task(self, batch_size):
        task_ind = np.random.choice(a=self._num_clusters, size=batch_size, replace=True, p=self._p_task)
        context = np.concatenate([self._means[task_ind], self._evecs_[task_ind]])
        return task_ind, context

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

    def compute_rewards(self, actions, obs):
        if self.reward_type == 's_given_z':
            ob = obs['observation']





            skill = obs['context']
            next_ob = obs['observation']

            cross_entropy = self.disc.evaluate_cross_entropy(inputs=next_ob, labels=skill)
            p_z = self._p_z[skill.astype(np.int)]
            log_p_z = np.log(p_z)

            assert cross_entropy.shape == log_p_z.shape
            reward = -1 * cross_entropy - log_p_z
            return reward
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError()

    def sample_goals(self, batch_size):
        task_ind, context = self._sample_task(batch_size)
        return {
            'task_ind': task_ind,
            'context': context
        }

    def sample_goal(self):
        goals = self.sample_goals(1)
        return self.unbatchify_dict(goals, 0)

    def fit(self, replay_buffer):
        loss = self.clusterer.fit(replay_buffer)
        self._set_clusterer_attributes()
        return loss

    @property
    def context_dim(self):
        return self._means[0].size + self._evecs[0].size

    def _set_clusterer_attributes(self):
        self._weights = self.clusterer.wrapped_clusterer.weights_
        self._means = self.clusterer.wrapped_clusterer.means_
        self._covs = self.clusterer.wrapped_clusterer.covariances_
        evals, self._evecs = np.linalg.eigh(self._covs)
        self._p_task = self._weights / self._weights.sum()
