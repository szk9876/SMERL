from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.half_cheetah_goal import HalfCheetahGoalEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class HalfCheetahGoalMassEnv(HalfCheetahGoalEnv):
    def __init__(self, action_scale=1, frame_skip=5, goal_position=3.0, mass_multiplier=1., timestep_start=100000, timestep_end=100000):
        self.quick_init(locals())
        self.mass_multiplier = mass_multiplier
        self.timestep_start = timestep_start
        self.timestep_end = timestep_end
        # self.original_mass = self.sim.model.body_mass[1]
        HalfCheetahGoalEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip, goal_position=goal_position)
        self.mass_index = 1
        # self.original_mass = np.array(self.sim.model.body_mass)[self.mass_index]
        self.original_mass = np.array(self.sim.model.body_mass)[:]

    def step(self, action):
        action = action * self.action_scale
        self.step_count += 1

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        # print(self.sim.model.body_mass)
        # if self.step_count >= self.timestep_start and self.step_count <= self.timestep_end:
        if xposafter >= self.timestep_start and xposafter <= self.timestep_end:
            body_mass = self.sim.model.body_mass
            body_mass = np.array(body_mass)
            # body_mass[self.mass_index] = self.original_mass * self.mass_multiplier
            body_mass[:] = self.original_mass * self.mass_multiplier
        else:
            body_mass = self.sim.model.body_mass
            body_mass = np.array(body_mass)
            # body_mass[self.mass_index] = self.original_mass
            body_mass[:] = self.original_mass     
        self.sim.model.body_mass[:] = body_mass

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
