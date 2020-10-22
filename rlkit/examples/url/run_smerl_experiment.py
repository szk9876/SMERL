import argparse
import gym
from rlkit.torch.url.path_replay_buffer import ObsDictPathReplayBuffer
from rlkit.torch.url.url import UrlTwinSac
from rlkit.torch.url.envs.disc_wrapper import DiscriminatorWrappedEnv
from rlkit.torch.url.discriminator import Discriminator
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.launchers.launcher_util import setup_logger


from multiworld.envs.pygame.point2d import Point2DEnv  # trigger env registration
from multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal import HalfCheetahGoalEnv
from multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal_obstacles import HalfCheetahGoalObstaclesEnv

from multiworld.envs.mujoco.classic_mujoco.walker_environments.walker2d_velocity import Walker2dVelocityEnv
from multiworld.envs.mujoco.classic_mujoco.walker_environments.walker2d_velocity_obstacles import Walker2dVelocityObstaclesEnv
from multiworld.envs.mujoco.classic_mujoco.walker_environments.walker2d_velocity_joints import Walker2dVelocityJointsEnv

from multiworld.envs.mujoco.classic_mujoco.all_hopper_environments.hopper_velocity import HopperVelocityEnv
from multiworld.envs.mujoco.classic_mujoco.all_hopper_environments.hopper_velocity_obstacles import HopperVelocityObstaclesEnv

import random
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--unsupervised-reward-weight", type=float, default=1.0, 
                    help="weight on the unsupervised reward")
parser.add_argument("--environment-reward-weight", type=float, default=1.0,
                    help="weight on the environment reward")
parser.add_argument("--algo", type=str, default="diayn",
                    help="diayn or tiayn")
parser.add_argument("--replay-buffer-size", type=int, default=1000)
parser.add_argument("--tiayn-training", type=str, default="transitions",
                    help="the manner in which the TIAYN discriminator is trained.")
parser.add_argument("--num_skills", type=int, default=10, help="number of skills.")
parser.add_argument("--env", type=str, default="HalfCheetah-v1", help="Environment.")
parser.add_argument("--subopt-return-threshold", type=float, default=0.0,
                    help="threshold at which to add the unsupervised reward.")
parser.add_argument("--target-entropy-multiplier", type=float, default=1.0,
                    help="target entropy multiplier.")
parser.add_argument("--seed", type=int, default=0, required=True)
parser.add_argument("--noise_scale", type=float, default=0.0, required=False)

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def experiment(variant):
    wrapped_env = gym.make(variant['env_name'])
    obs_dim = wrapped_env.observation_space.spaces['observation'].low.size
    action_dim = wrapped_env.action_space.low.size
    net_size = variant['net_size']
    subopt_return_threshold = variant['subopt_return_threshold']    
    environment_reward_weight = variant['environment_reward_weight']
    unsupervised_reward_weight = variant['unsupervised_reward_weight']   
    noise_scale = variant['noise_scale'] 

    assert variant['experiment'] == 'diayn' or variant['experiment'] == 'tiayn'

    if variant['experiment'] == 'diayn':
        disc = Discriminator(
            discriminator_type='diayn',
            input_size=obs_dim,
            output_size=variant['disc_kwargs']['num_skills'],
            hidden_sizes=[net_size, net_size],
            **variant['disc_kwargs'])
    elif variant['experiment'] == 'tiayn':
        disc = Discriminator(
            discriminator_type='tiayn',
            input_size=obs_dim + obs_dim,
            output_size=variant['disc_kwargs']['num_skills'],
            hidden_sizes=[net_size, net_size],
            **variant['disc_kwargs'])
    
    # disc = disc.to('cuda:1')

    env = DiscriminatorWrappedEnv(wrapped_env=wrapped_env,
                                  noise_scale=noise_scale,
                                  disc=disc,
                                  **variant['env_kwargs']
                                  )

    context_dim = env.context_dim

    qf1 = FlattenMlp(
        input_size=obs_dim + context_dim + action_dim,
        output_size=1,
        hidden_sizes=[net_size, net_size],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + context_dim + action_dim,
        output_size=1,
        hidden_sizes=[net_size, net_size],
    )
    vf = FlattenMlp(
        input_size=obs_dim + context_dim,
        hidden_sizes=[net_size, net_size],
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + context_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size],
        num_skills=variant['num_skills'],
    )
    replay_buffer = ObsDictPathReplayBuffer(
        env=env,
        max_path_length=variant['algo_kwargs']['max_path_length'],
        observation_key='observation',
        context_key='context',
        subopt_return_threshold=subopt_return_threshold,
        environment_reward_weight=environment_reward_weight,
        unsupervised_reward_weight=unsupervised_reward_weight,
        **variant['replay_buffer_kwargs']
    )
    algorithm = UrlTwinSac(
        replay_buffer=replay_buffer,
        url_kwargs=dict(
            observation_key='observation',
            context_key='context',
            fitting_period=1,
            env_loss_key='discriminator loss'
        ),
        tsac_kwargs=dict(
            env=env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            vf=vf,
        ),
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":

    args = parser.parse_args()

    # Set the seed for the experiment.
    # set_seed(args.seed)   

    if args.unsupervised_reward_weight == 0.:
        algo = 'wrapped_env'
    elif args.environment_reward_weight == 0.:
        algo = args.algo
    else:
        algo = 'wrapped_env + {}'.format(args.algo)

    sampling_strategy = 'random'
    if 'tiayn' in algo:
        if args.tiayn_training == 'paths':
            sampling_strategy = 'random_paths'

    if args.env == 'Point2DWalls-corner-v2' or args.env == 'Point2D-center-v2':
    # point2d
        variant = dict(
            env_name=args.env,
            num_skills=args.num_skills,
            subopt_return_threshold=args.subopt_return_threshold,
            environment_reward_weight=args.environment_reward_weight,
            unsupervised_reward_weight=args.unsupervised_reward_weight,
            noise_scale=args.noise_scale,
            algo_kwargs=dict(
                num_epochs=50,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                # num_updates_per_epoch=1000,
                num_updates_per_env_step=1,
                max_path_length=100,
                batch_size=128,
                discount=0.99,
   
                soft_target_tau=0.01,
                policy_lr=3e-4,
                qf_lr=3e-4,
                vf_lr=3e-4,
    
                collection_mode='online',
                save_replay_buffer=False,
                save_environment=True,
    
             	use_automatic_entropy_tuning=True,
                target_entropy_multiplier=args.target_entropy_multiplier,
                fixed_entropy=0.1
            ),
            replay_buffer_kwargs=dict(
                max_size=args.replay_buffer_size,
            ),
            disc_kwargs=dict(
                batch_size=128,
                num_batches_per_fit=1,
                num_skills=args.num_skills,
                sampling_strategy=sampling_strategy,
                sampling_window=10,
            ),
            env_kwargs=dict(
                reward_params=dict(type=algo),
                unsupervised_reward_weight=args.unsupervised_reward_weight,
                reward_weight=args.environment_reward_weight
            ),
            net_size=32, 
            experiment=args.algo,
        )
    else:
        if 'HalfCheetahGoal' in args.env:
            max_path_length = 500
            num_epochs = 200
        else:
            max_path_length = 200
            num_epochs = 1000

        variant = dict(
            env_name=args.env,
            num_skills=args.num_skills,
            subopt_return_threshold=args.subopt_return_threshold,
            environment_reward_weight=args.environment_reward_weight,
            unsupervised_reward_weight=args.unsupervised_reward_weight,
            noise_scale=args.noise_scale,
            algo_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                # num_updates_per_epoch=1000,
                num_updates_per_env_step=1,
                max_path_length=max_path_length,
                batch_size=256,
                discount=0.99,

                soft_target_tau=0.005,
                policy_lr=3e-4,
                qf_lr=3e-4,
                vf_lr=3e-4,

                collection_mode='online',
                save_replay_buffer=False,
                save_environment=True,

                use_automatic_entropy_tuning=True,
                target_entropy_multiplier=args.target_entropy_multiplier,
                fixed_entropy=0.1
            ),
            replay_buffer_kwargs=dict(
                max_size=args.replay_buffer_size,
            ),
            disc_kwargs=dict(
                batch_size=256,
                num_batches_per_fit=1,
                num_skills=args.num_skills,

                sampling_strategy=sampling_strategy,
                sampling_window=10,
            ),
            env_kwargs=dict(
                reward_params=dict(type=algo),
                unsupervised_reward_weight=args.unsupervised_reward_weight,
                reward_weight=args.environment_reward_weight
            ),
            net_size=300, 
            experiment=args.algo,
        )

    ptu.set_gpu_mode(True, 0)  # optionally set the GPU (default=False)
    if algo == 'wrapped_env':
        setup_logger('CAMERA_READY_EXPERIMENTS/{}/env_weight_{}/seed{}/replay_buffer_size_{}/num_skills_{}/target_entropy_multiplier_{}/action_noise_{}'.format(args.env, args.environment_reward_weight, args.seed, args.replay_buffer_size, args.num_skills, args.target_entropy_multiplier, args.noise_scale), variant=variant)
    elif algo == 'diayn':
        setup_logger('CAMERA_READY_EXPERIMENTS/{}/unsupervised_weight_{}/seed{}/replay_buffer_size_{}/num_skills_{}/target_entropy_multiplier_{}/action_noise_{}'.format(args.env, args.unsupervised_reward_weight, args.seed, args.replay_buffer_size, args.num_skills, args.target_entropy_multiplier, args.noise_scale), variant=variant)
    else:
        setup_logger('CAMERA_READY_EXPERIMENTS/{}/env_weight_{}_{}_weight_{}/subopt_return_threshold_{}/seed_{}/replay_buffer_size_{}/num_skills_{}/target_entropy_multiplier_{}/action_noise_{}'.format(args.env, args.environment_reward_weight, algo, args.unsupervised_reward_weight, args.subopt_return_threshold, args.seed, args.replay_buffer_size, args.num_skills, args.target_entropy_multiplier, args.noise_scale), variant=variant)

    experiment(variant)
