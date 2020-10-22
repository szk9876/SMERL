from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_goal import AntGoalEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class AntGoalDisabledJointsEnv(AntGoalEnv):
    def __init__(self, action_scale=1, frame_skip=5, goal_position=4., force=None, timestep_start=100000, timestep_end=100000):
        self.quick_init(locals())
        self.force = force
        self.timestep_start = timestep_start
        self.timestep_end = timestep_end
        AntGoalEnv.__init__(self, action_scale=action_scale, frame_skip=frame_skip, goal_position=goal_position)


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

        goal_reward = -1.0 * np.linalg.norm(xposafter - self.goal_position) # make it happy, not suicidal
        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        survive_reward = 0

        # state = self.state_vector()
        # notdone = np.isfinite(state).all() \
        #    and state[2] >= 0.2 and state[2] <= 1.0

        reward = goal_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()

        done = False

        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def reset(self):
        self.step_count = 0
        return super().reset()
