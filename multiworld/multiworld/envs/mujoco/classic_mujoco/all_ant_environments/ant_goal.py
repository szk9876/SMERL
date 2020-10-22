import numpy as np
from multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_multitask_base import MultitaskAntEnv

# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, n_tasks=1, randomize_tasks=False, action_scale=1, frame_skip=5, goal_position=5):
        self.quick_init(locals())
        self.randomize_tasks = randomize_tasks
        self.goal_position = goal_position
        # self.goal_position = 3
        super(AntGoalEnv, self).__init__(n_tasks=n_tasks, action_scale=action_scale, frame_skip=frame_skip)

    def step(self, action):
        # print('Goal: ', self._goal)
        action = action * self.action_scale
        self.do_simulation(action, self.frame_skip)
        
        # Set the goal in the xml file.
        goal_marker_idx = self.sim.model.site_name2id('goal')
        self.data.site_xpos[goal_marker_idx,:2] = self._goal
        self.data.site_xpos[goal_marker_idx,-1] = 1  

        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

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

    def sample_tasks(self, num_tasks):
        if num_tasks > 1 and not self.randomize_tasks:
            raise NotImplementedError
        if not self.randomize_tasks and num_tasks == 1:
            a = np.array([0.5]) * 2 * np.pi  # 0.12
            r = self.goal_position * np.array([1.]) ** 0.5    # 3 * np.array...    

        if self.randomize_tasks:        
            a = np.random.random(num_tasks) * 2 * np.pi
            r = 5 * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_env_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
