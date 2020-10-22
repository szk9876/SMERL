from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.walker2d_original import Walker2dEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class Walker2dObstaclesEnv(Walker2dEnv):

    def __init__(self, action_scale=1, frame_skip=5, obstacle_position=None, obstacle_height=None):
        assert obstacle_position is not None and obstacle_height is not None
        self.quick_init(locals())
        self.obstacle_position = obstacle_position
        self.obstacle_height = obstacle_height
        Walker2dEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip)

    @property
    def model_name(self):
        if self.obstacle_height == 'multiple':
            return get_asset_full_path('classic_mujoco/walker_obstacles/walker_multiple_obstacles.xml')

        return get_asset_full_path(
            'classic_mujoco/walker_obstacles/walker_obstacle_position={}_height={}.xml'.format(
                self.obstacle_position, self.obstacle_height))
    
    #def reset_model(self):
    #    self.set_state(
    #        self.init_qpos,
    #        self.init_qvel
    #    )
    #    return self._get_obs()
