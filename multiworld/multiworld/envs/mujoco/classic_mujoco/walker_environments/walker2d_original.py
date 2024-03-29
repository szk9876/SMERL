from collections import OrderedDict
import numpy as np
from gym.spaces import Dict, Box
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

class Walker2dEnv(MujocoEnv, MultitaskEnv, Serializable):

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
        return get_asset_full_path('classic_mujoco/walker2d.xml')

    def step(self, action):
        action = action * self.action_scale
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        infor = self._get_info()
        return ob, reward, done, {}

    def _get_env_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

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
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def reset(self):
        self.reset_model()
        return self._get_obs()

    def get_diagnostics(Self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def get_goal(self):
        return None

    def sample_goals(self):
        return None
