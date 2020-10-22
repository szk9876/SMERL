from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.half_cheetah_original import HalfCheetahEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class HalfCheetahGoalEnv(HalfCheetahEnv):
    def __init__(self, action_scale=1, frame_skip=5, goal_position=4.):
        self.quick_init(locals())
        self.goal_position = goal_position
        HalfCheetahEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip)

    def step(self, action):
        # goal_marker_idx = self.sim.model.site_name2id('goal')
        # self.data.site_xpos[goal_marker_idx,0] = self.goal_position

        action = action * self.action_scale
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        
        # goal_marker_idx = self.sim.model.site_name2id('goal')
        # self.data.site_xpos[goal_marker_idx,0] = self.goal_position
        
        xposafter = self.sim.data.qpos[0]

        reward = -1.0 * np.linalg.norm(xposafter - self.goal_position)
        # print(xposafter, self.goal_position, reward)
        ob = self._get_obs()
        info = self._get_info()
        done = (reward > -0.5)
        return ob, reward, done, info
