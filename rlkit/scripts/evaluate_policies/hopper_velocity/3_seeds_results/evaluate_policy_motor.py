from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger

from collections import defaultdict
import numpy as np
import rlkit.torch.pytorch_util as ptu
import pickle as pkl


import gym
from rlkit.torch.url.envs.disc_wrapper import DiscriminatorWrappedEnv

from multiworld.envs.mujoco.classic_mujoco.all_hopper_environments.hopper_velocity_motor import HopperVelocityMotorFailureEnv



import numpy as np
import torch

filename = str(uuid.uuid4())


def simulate_policy(args):
    assert args.env_name == 'HopperVelocity'

    if args.env_name == 'HopperVelocity':
        if args.model == 'sac_1_skill_seed0':
             files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0/seed0/replay-buffer-size-1000/num-skills-1/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0/seed0/replay_buffer_size_1000/num_skills_1/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_12_21_25_23_0000--s-0/params.pkl']
        elif args.model == 'sac_1_skill_seed1':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0/seed1/replay-buffer-size-1000/num-skills-1/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0/seed1/replay_buffer_size_1000/num_skills_1/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_12_22_28_15_0000--s-0/params.pkl']
        elif args.model == 'sac_1_skill_seed2':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0/seed2/replay-buffer-size-1000/num-skills-1/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0/seed2/replay_buffer_size_1000/num_skills_1/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_13_00_49_17_0000--s-0/params.pkl']

        elif args.model == 'sac_5_skills_seed0':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0/seed0/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0/seed0/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_12_19_33_22_0000--s-0/params.pkl']
        elif args.model == 'sac_5_skills_seed1':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0/seed1/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0/seed1/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_12_20_37_51_0000--s-0/params.pkl']
        elif args.model == 'sac_5_skills_seed2':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0/seed2/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0/seed2/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_12_21_28_45_0000--s-0/params.pkl']

        elif args.model == 'diayn_seed0':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/unsupervised-weight-1.0/seed0/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/unsupervised_weight_1.0/seed0/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_13_02_28_44_0000--s-0/params.pkl']
        elif args.model == 'diayn_seed1':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/unsupervised-weight-1.0/seed1/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/unsupervised_weight_1.0/seed1/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_13_02_42_23_0000--s-0/params.pkl']
        elif args.model == 'diayn_seed2':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/unsupervised-weight-1.0/seed2/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/unsupervised_weight_1.0/seed2/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_13_02_46_00_0000--s-0/params.pkl']

        elif args.model == 'smerl_seed0':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0-wrapped-env + diayn-weight-10.0/subopt-return-threshold-600.0/seed-0/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0_wrapped_env + diayn_weight_10.0/subopt_return_threshold_600.0/seed_0/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_14_22_58_16_0000--s-0/params.pkl']
        elif args.model == 'smerl_seed1':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0-wrapped-env + diayn-weight-10.0/subopt-return-threshold-600.0/seed-1/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0_wrapped_env + diayn_weight_10.0/subopt_return_threshold_600.0/seed_1/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_14_22_58_16_0000--s-0/params.pkl']
        elif args.model == 'smerl_seed2':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0-wrapped-env + diayn-weight-10.0/subopt-return-threshold-600.0/seed-2/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0_wrapped_env + diayn_weight_10.0/subopt_return_threshold_600.0/seed_2/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_14_22_58_16_0000--s-0/params.pkl']

        elif args.model == 'sac+diayn_seed0':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0-wrapped-env + diayn-weight-0.5/subopt-return-threshold--10000000.0/seed-0/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0_wrapped_env + diayn_weight_0.5/subopt_return_threshold_-10000000.0/seed_0/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_13_08_07_52_0000--s-0/params.pkl']
        elif args.model == 'sac+diayn_seed1':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0-wrapped-env + diayn-weight-0.5/subopt-return-threshold--10000000.0/seed-1/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0_wrapped_env + diayn_weight_0.5/subopt_return_threshold_-10000000.0/seed_1/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_13_08_36_05_0000--s-0/params.pkl']
        elif args.model == 'sac+diayn_seed2':
            files = ['rlkit/data/CAMERA-READY-EXPERIMENTS/HopperVelocityEnv-v1/env-weight-1.0-wrapped-env + diayn-weight-0.5/subopt-return-threshold--10000000.0/seed-2/replay-buffer-size-1000/num-skills-5/target-entropy-multiplier-1.0/action-noise-0.0/CAMERA_READY_EXPERIMENTS/HopperVelocityEnv-v1/env_weight_1.0_wrapped_env + diayn_weight_0.5/subopt_return_threshold_-10000000.0/seed_2/replay_buffer_size_1000/num_skills_5/target_entropy_multiplier_1.0/action_noise_0.0_2020_10_13_08_49_34_0000--s-0/params.pkl']

    data = joblib.load(files[0])
    policy = data['policy']             

    # Create test environments.
    envs = []
    
    # for action in range(5, 35, 5):  # (3, 9)
    for action in range(10, 110, 10):
        if args.env_name == 'HopperVelocity':
            wrapped_env = gym.make('HopperVelocityMotorFailure-v{}'.format(action))
            disc = data['env'].disc
            reward_params = data['env'].reward_params
            unsupervised_reward_weight = 0.0
            reward_weight = 1.0
            env = DiscriminatorWrappedEnv(wrapped_env=wrapped_env,
                                          disc=disc,
                                          reward_params=reward_params,
                                          unsupervised_reward_weight=unsupervised_reward_weight,
                                          reward_weight=reward_weight)
            envs.append(env)

    performance_data = np.zeros((len(envs), args.num_skills))

    for env_index in range(len(envs)):
        env = envs[env_index]

        skill_returns = defaultdict(int)
        for skill in range(args.num_skills):

            for f in files:

                data = joblib.load(f)
                policy = data['policy']

                print("Policy loaded")
                if args.gpu:
                    set_gpu_mode(True)
                    policy.cuda()
                if isinstance(policy, PyTorchModule):
                    policy.train(False)
                
                for trial in range(5):
                    print('Skill: {}'.format(skill))
                    path = rollout(
                        env,
                        policy,
                        max_path_length=args.H,
                        animated=False,
                        skill=skill,
                        deterministic=True
                    )
                    if skill not in skill_returns:
                        skill_returns[skill] = []
                    skill_returns[skill].append(np.sum(path['rewards']))

                    if hasattr(env, "log_diagnostics"):
                        env.log_diagnostics([path])
                    logger.dump_tabular()
            print(skill_returns[skill])
        skill_returns_averaged = defaultdict(int)
        for skill in range(args.num_skills):
            skill_returns_averaged[skill] = np.mean(np.array(skill_returns[skill]))
            performance_data[env_index, skill] = skill_returns_averaged[skill]
        print(skill_returns_averaged)

    # Compute the best skills and then evaluate the best ones over multiple seeds and trials.

    # Compute best skills. 
    best_skills = np.argmax(performance_data, -1)

    num_trials = 5
    performance_data_final = np.zeros((len(envs), num_trials * len(files)))
    skill_returns = []

    for env_index in range(len(envs)):
        env = envs[env_index]
        skill = best_skills[env_index]
        skill_returns = []

        for f in files:
            data = joblib.load(f)
            policy = data['policy']

            print("Policy loaded")
            if args.gpu:
                set_gpu_mode(True)
                policy.cuda()
            if isinstance(policy, PyTorchModule):
                policy.train(False)
            
            for trial in range(num_trials):
                print('Skill: {}'.format(skill))
                path = rollout(
                    env,
                    policy,
                    max_path_length=args.H,
                    animated=False,
                    skill=skill,
                    deterministic=True
                )
                skill_returns.append(np.sum(path['rewards']))

                if hasattr(env, "log_diagnostics"):
                    env.log_diagnostics([path])
                logger.dump_tabular()
       
        performance_data_final[env_index] = np.array(skill_returns)

    fileName = 'MotorResults/{}_{}_motor'.format(args.env_name, args.model)
    np.save(fileName, performance_data_final)

    fileName = 'MotorResults/{}_{}_motor_all_data'.format(args.env_name, args.model)
    np.save(fileName, performance_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=200,
                        help='Max length of rollout', required=False)
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--num_skills', type=int, default=5)
    parser.add_argument('--env_name', type=str, default='HopperVelocity', required=False)
    parser.add_argument('--model', type=str, default='sac', required=True, help='sac or prod')
    args = parser.parse_args()
    ptu.set_gpu_mode(True, 0)
    simulate_policy(args)

