from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal import HalfCheetahGoalEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class HalfCheetahGoalDisabledJointsEnv(HalfCheetahGoalEnv):
    def __init__(self, action_scale=1, frame_skip=5, goal_position=3.0, force=None, timestep_start=100000, timestep_end=100000):
        self.quick_init(locals())
        self.force = force
        self.timestep_start = timestep_start
        self.timestep_end = timestep_end
        HalfCheetahGoalEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip, goal_position=goal_position)


    def step(self, action):
        action = action * self.action_scale
        self.step_count += 1
        # print(self.sim.model.actuator_acc0)
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        
        if self.step_count >= self.timestep_start and self.step_count <= self.timestep_end:
            # for i in range(9):
            self.sim.data.qfrc_applied[:] = self.force
            # import pdb; pdb.set_trace()
            # self.sim.model.actuator_acc0[i] = -1000.
            # self.sim.data.qacc[i] = -1000. #self.force

        xposafter = self.sim.data.qpos[0]

        reward = -1.0 * np.linalg.norm(xposafter - self.goal_position)

        ob = self._get_obs()
        info = self._get_info()
        done = (reward > -0.5)
        if done:
            print(self.step_count)
        return ob, reward, done, info

    def reset(self):
        self.step_count = 0
        return super().reset()
