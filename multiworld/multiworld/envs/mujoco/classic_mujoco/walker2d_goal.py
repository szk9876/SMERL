from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.walker2d_original import Walker2dEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class Walker2dGoalEnv(Walker2dEnv):

    def __init__(self, action_scale=1, frame_skip=5, goal_position=5.):
        self.quick_init(locals())
        self.goal_position = goal_position
        Walker2dEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip)

    def step(self, action):
        action = action * self.action_scale
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 0.0
        # alive_bonus = 1.0
        ######### Never used: reward = ((posafter - posbefore) / self.dt)
        reward = alive_bonus
        # print(posafter, self.goal_position)
        control_reward = -1e-3 * np.square(action).sum()
        goal_reward = -1.0 * np.linalg.norm(posafter - self.goal_position)
        reward = control_reward + goal_reward + alive_bonus
        
        bad_done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        
        done = bad_done or (goal_reward > -0.3)
        done = goal_reward > -0.3

        if bad_done:
            reward += -1000

        # if goal_reward > -0.3:
        #    reward += 1000

        ob = self._get_obs()
        infor = self._get_info()
        return ob, reward, done, {}
