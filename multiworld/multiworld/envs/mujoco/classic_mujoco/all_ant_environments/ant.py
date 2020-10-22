from collections import OrderedDict
import numpy as np
from gym.spaces import Dict, Box
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from multiworld.envs.mujoco.mujoco_env import MujocoEnv


class AntEnv(MujocoEnv, MultitaskEnv, Serializable):
    def __init__(self, action_scale=1, frame_skip=5):
        # self.init_serialization(locals())
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        self.action_scale = action_scale
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low=low, high=high)
        obs_size = self._get_env_obs().shape[0]
        high = np.inf * np.ones(obs_size)
        low = -high
        self.obs_space = Box(low=low, high=high)
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('state_observation', self.obs_space),
        ])
        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('classic_mujoco/ant.xml')


    def step(self, a):
        a = a * self.action_scale
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0]/self.dt
        # ctrl_cost = 0. # .5 * np.square(a).sum()
        
        # Saurabh 5/28
        ctrl_cost = .5 * np.square(a).sum()

        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # survive_reward = 0. # 1.0
        
        # Saurabh 5/28
        survive_reward = 1.0

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def _get_env_obs(self):
        # this is gym ant obs, should use rllab?
        # if position is needed, override this in subclasses
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def _get_obs(self):
        state_obs = self._get_env_obs()
        return dict(observation=state_obs,
                    state_observation=state_obs,
        )

    def compute_rewards(self):
        pass

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reset(self):
        self.reset_model()
        return self._get_obs()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def get_goal(self):
        return None
    
    def sample_goals(self):
        return None
