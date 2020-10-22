from collections import OrderedDict
import numpy as np
from gym.spaces import Dict, Box
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

class HalfCheetahEnv(MujocoEnv, MultitaskEnv, Serializable):
    def __init__(self, action_scale=1, frame_skip=5):
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
        self.obs_space = Box(low, high)
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('state_observation', self.obs_space),
        ])
        self.reset()


    @property
    def model_name(self):
        return get_asset_full_path('classic_mujoco/half_cheetah.xml')

    def step(self, action):
        action = action * self.action_scale
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        info = self._get_info()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, info

    def _get_env_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])
    
    def _get_obs(self):
        state_obs = self._get_env_obs()
        return dict(
            observation=state_obs,
            state_observation=state_obs,
        )

    def _get_info(self, ):
        info = dict()
        return info
    
    def compute_rewards(self, actions, obs):
        pass

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
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
