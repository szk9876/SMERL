from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.walker2d_original import Walker2dEnv

class Walker2dObstaclesEnv(Walker2dEnv):

    def __init__(self, action_scale=1, frame_skip=5, obstacle_region=None):
        self.quick_init(locals())
        self.obstacle_region = obstacle_region
        Walker2dEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip)

    def step(self, action):
        action = action * self.action_scale
        posbefore = self.sim.data.qpos[0]

        if self.obstacle_region is not None:
            # Compute x position of Walker's foot.
            ffoot_key = self.sim.model.body_name2id('foot')
            ffoot_x_pos = self.sim.data.body_xpos[ffoot_key][0]
            # import pdb; pdb.set_trace()
            if not(ffoot_x_pos >= self.obstacle_region[0] and ffoot_x_pos < self.obstacle_region[1]):
                self.do_simulation(action, self.frame_skip)
        else:
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
    
    # def reset_model(self):
    #    self.set_state(
    #        self.init_qpos,
    #        self.init_qvel
    #    )
    #    return self._get_obs()
