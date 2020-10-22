from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.ant_goal import AntGoalEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class AntGoalObstaclesEnv(AntGoalEnv):
    def __init__(self, action_scale=1, frame_skip=5, goal_position=4., obstacle_position=None, obstacle_height=None):
        assert obstacle_position is not None and obstacle_height is not None
        self.quick_init(locals())
        self.obstacle_position = obstacle_position
        self.obstacle_height = obstacle_height
        AntGoalEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip, goal_position=goal_position)

    @property
    def model_name(self):
        return get_asset_full_path(
            'classic_mujoco/ant_obstacles/ant_obstacle_position={}_height={}.xml'.format(
                self.obstacle_position, self.obstacle_height))


