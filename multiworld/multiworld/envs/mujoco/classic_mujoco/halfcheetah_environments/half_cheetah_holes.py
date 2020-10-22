from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.half_cheetah_original import HalfCheetahEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class HalfCheetahObstaclesEnv(HalfCheetahEnv):
    def __init__(self, action_scale=1, frame_skip=5, obstacle_region=None):
        self.quick_init(locals())
        self.obstacle_region = obstacle_region
        HalfCheetahEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip)

    def step(self, action):
        action = action * self.action_scale
        xposbefore = self.sim.data.qpos[0]
        
        if self.obstacle_region is not None:
            # Compute x position of HalfCheetah's front foot.
            ffoot_key = self.sim.model.body_name2id('ffoot')
            ffoot_x_pos = self.sim.data.body_xpos[ffoot_key][0]            
          
            if not(ffoot_x_pos >= self.obstacle_region[0] and ffoot_x_pos < self.obstacle_region[1]):
                self.do_simulation(action, self.frame_skip)
        else:
            self.do_simulation(action, self.frame_skip)
    
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        info = self._get_info()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, info


    # def reset_model(self):
    #    qpos = self.init_qpos
    #    qvel = self.init_qvel + .1
    #    self.set_state(qpos, qvel)
    #    return self._get_obs()
