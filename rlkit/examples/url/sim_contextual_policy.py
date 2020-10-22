import argparse
import pickle

from rlkit.core import logger
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.torch import pytorch_util as ptu
# from rlkit.envs.vae_wrapper import VAEWrappedEnv
from collections import defaultdict
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import numpy as np
import os

def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    data = pickle.load(open(args.file, "rb"))
    policy = data['policy']
    env = data['env']
    print("Policy and environment loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    # if isinstance(env, VAEWrappedEnv):
    #     env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    policy.train(False)
    paths = []
    for i in range(1000):
        paths.append(multitask_rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
            observation_key='observation',
            desired_goal_key='context',
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()

    if 'point2d' in type(env.wrapped_env).__name__.lower():
        point2d(paths, args)


def point2d(paths, args):
    def add_time(trajectories):
        assert trajectories.ndim == 3
        time = np.arange(trajectories.shape[1])
        time = np.tile(time, [trajectories.shape[0], 1])
        time = np.expand_dims(time, axis=2)
        trajectories = np.concatenate([trajectories, time], axis=2)
        return trajectories

    skill_to_obs = defaultdict(list)
    for path in paths:
        skill = path['goals'][0].argmax()
        skill_to_obs[skill].append(path['observations'])

    num_skills = len(skill_to_obs.keys())
    ncols = 5
    nrows = int(np.ceil(num_skills / 5))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all',
                             figsize=[5 * ncols, 5 * nrows])
    axes = axes.reshape([-1])

    # ipdb.set_trace()

    for key, ax in zip(skill_to_obs.keys(), axes):
        trajectories = np.stack(skill_to_obs[key], axis=0)
        trajectories = add_time(trajectories)
        states = trajectories.reshape([-1, trajectories.shape[-1]])
        ax.scatter(states[:, 0], states[:, 1], s=1 ** 2, c=states[:, 2], marker='o')

        ax.set_title('{}'.format(key))

    filename = os.path.join(os.path.dirname(args.file), 'skills.png')
    plt.savefig(filename)

    plt.close('all')

    x = 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=50,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    # parser.add_argument('--mode', default='video_env', type=str,
    #                     help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
