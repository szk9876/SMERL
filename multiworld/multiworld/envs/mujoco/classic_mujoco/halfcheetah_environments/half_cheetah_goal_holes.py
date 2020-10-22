from collections import OrderedDict
import numpy as np 
from multiworld.envs.mujoco.classic_mujoco.half_cheetah_goal import HalfCheetahGoalEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path


class HalfCheetahGoalHolesEnv(HalfCheetahGoalEnv):

    def __init__(self, action_scale=1, frame_skip=5, goal_position=4.):
        self.quick_init(locals())

        self.num_panels = 0

        HalfCheetahGoalEnv.__init__(
            self, action_scale=action_scale, frame_skip=frame_skip, goal_position=goal_position)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        # self.dt = self.model.opt.timestep
        damping = self.model.dof_damping.copy()
        damping[:self.num_panels] = np.zeros(self.num_panels)
        self.model.dof_damping[:] = damping
        # print(self.model.dof_damping)

    @property
    def model_name(self):
        return get_asset_full_path('classic_mujoco/half_cheetah_blocks.xml')
    
    def step(self, action):
        action = action * self.action_scale
        xposbefore = self.sim.data.qpos[self.num_panels]

        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[self.num_panels]

        reward = -1.0 * np.linalg.norm(xposafter - self.goal_position)

        ob = self._get_obs()
        info = self._get_info()
        done = (reward > -0.5)
        return ob, reward, done, info

    # @property
    # def dt(self):
    #     return self.model.opt.timestep * self.frame_skip

    def _get_env_obs(self):
        # print(self.sim.data.qpos.flat[:])
        return np.concatenate([
            self.sim.data.qpos.flat[1 + self.num_panels:],
            self.sim.data.qvel.flat[self.num_panels:],
        ]) 
