import numpy as np
import torch

from rlkit.torch.url.path_replay_buffer import ObsDictPathReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch.sac.twin_sac import TwinSAC
from rlkit.core.rl_algorithm import set_to_eval_mode, set_to_train_mode
import gtimer as gt
from gym.spaces import Discrete
from rlkit.torch import pytorch_util as ptu

class URL(TorchRLAlgorithm):
    """
        Note: this assumes the env will sample the goal when reset() is called,
        i.e. use a "silent" env.

        Unsupervised Reinforcement Learning

        This is a template class that should be the first sub-class, i.e.[

        ```
        class UrlSac(URL, SAC):
        ```

        and not

        ```
        class UrlSac(SAC, URL):
        ```

        Or if you really want to make SAC the first subclass, do alternatively:
        ```
        class UrlSac(SAC, URL):
            def get_batch(self):
                return URL.get_batch(self)
        ```
        for each function defined below.
        """

    def __init__(
            self,
            observation_key=None,
            context_key=None,
            fitting_period=1,
            env_loss_key='discriminator loss'
    ):
        self.observation_key = observation_key
        self.context_key = context_key
        self.fitting_period = fitting_period
        self.env_loss = float('nan')
        self.env_loss_key = env_loss_key

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )

    def get_batch(self):
        batch = super().get_batch()
        obs = batch['observations']
        next_obs = batch['next_observations']

        if type(self.replay_buffer.ob_spaces[self.context_key]) is Discrete:
            context = ptu.zeros((obs.shape[0], self.replay_buffer.ob_spaces[self.context_key].n))
            context.scatter_(dim=1, index=batch['context'].long(), src=torch.tensor(1))
        else:
            context = batch['context']

        batch['observations'] = torch.cat((
            obs,
            context
        ), dim=1)
        batch['next_observations'] = torch.cat((
            next_obs,
            context
        ), dim=1)
        return batch

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            # self.env.update_rewards(path)
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)  # unneeded, wastes memory
            self._current_path_builder = PathBuilder()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)

        if type(self.replay_buffer.ob_spaces[self.context_key]) is Discrete:
            context = np.zeros(self.replay_buffer.ob_spaces[self.context_key].n)
            context[observation[self.context_key]] = 1
        else:
            context = observation[self.context_key]

        new_obs = np.hstack((
            observation[self.observation_key],
            context
        ))
        return self.exploration_policy.get_action(new_obs)

    def get_eval_paths(self):
        paths = []
        n_steps_total = 0
        while n_steps_total <= self.num_steps_per_eval:
            path = self.eval_multitask_rollout()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def eval_multitask_rollout(self):
        return multitask_rollout(
            self.env,
            self.policy,
            self.max_path_length,
            observation_key=self.observation_key,
            desired_goal_key=self.context_key,
        )

    def train_online(self, start_epoch=0):
        self._current_path_builder = PathBuilder()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            observation = self._start_new_rollout()
            for _ in range(self.num_env_steps_per_epoch):
                observation = self._take_step_in_env(observation)
                gt.stamp('sample')

                self._try_to_fit(epoch)
                gt.stamp('env_fit')

                self._try_to_train()
                gt.stamp('train')

            self.logger.record_tabular(self.env_loss_key, self.env_loss)

            set_to_eval_mode(self.env)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            self._end_epoch(epoch)

            self.logger.dump_tabular(with_prefix=False, with_timestamp=False)
    
    def train_batch(self, start_epoch):
        self._current_path_builder = PathBuilder()
        for epoch in gt.timed_for(
                range(start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._start_epoch(epoch)
            set_to_train_mode(self.training_env)
            observation = self._start_new_rollout()
            # This implementation is rather naive. If you want to (e.g.)
            # parallelize data collection, this would be the place to do it.
            for _ in range(self.num_env_steps_per_epoch):
                observation = self._take_step_in_env(observation)
            gt.stamp('sample')

            self._try_to_train()
            gt.stamp('train')

            set_to_eval_mode(self.env)
            self._try_to_eval(epoch)
            gt.stamp('eval')
            
            self._try_to_fit(epoch)
            gt.stamp('env_fit')
            self.logger.record_tabular(self.env_loss_key, self.env_loss)
            
            self._end_epoch(epoch)
            self.logger.dump_tabular(with_prefix=False, with_timestamp=False)
            
    def _try_to_fit(self, epoch):
        if epoch % self.fitting_period == 0 and self._can_train():
            self.env_loss = self.env.fit(self.replay_buffer)

    def _try_to_eval(self, epoch, eval_paths=None):
        self.logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch, eval_paths=eval_paths)

            params = self.get_epoch_snapshot(epoch)
            self.logger.save_itr_params(epoch, params)
            table_keys = self.logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            self.logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            self.logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            self.logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            env_fit_time = times_itrs['env_fit'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time + env_fit_time
            total_time = gt.get_times().total

            self.logger.record_tabular('Train Time (s)', train_time)
            self.logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            self.logger.record_tabular('Sample Time (s)', sample_time)
            self.logger.record_tabular('Epoch Time (s)', epoch_time)
            self.logger.record_tabular('Env Fit Time (s)', env_fit_time)
            self.logger.record_tabular('Total Train Time (s)', total_time)
            self.logger.record_tabular("Epoch", epoch)
        else:
            self.logger.log("Skipping eval for now.")


class UrlTwinSac(URL, TwinSAC):
    def __init__(
            self,
            *args,
            url_kwargs,
            tsac_kwargs,
            **kwargs
    ):
        URL.__init__(self, **url_kwargs)
        TwinSAC.__init__(self, *args, **kwargs, **tsac_kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictPathReplayBuffer
        )

    def get_eval_action(self, observation, goal):
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.context_key:
            goal = goal[self.context_key]
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)
