from multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant import AntEnv

class MultitaskAntEnv(AntEnv):
    def __init__(self, n_tasks=1, action_scale=1, frame_skip=5):
        self.quick_init(locals())
        self._task = {}
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']
        super(MultitaskAntEnv, self).__init__(action_scale=action_scale, frame_skip=frame_skip)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self.reset()
