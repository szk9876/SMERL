import gym
from gym.envs.registration import register
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld mujoco gym environments")

    register(
        id='HalfCheetahEnv-v0',
        entry_point='multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah:HalfCheetahEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
            'reward_type': 'vel_distance',
            'fix_goal': False,
            'max_speed': 6
        }
    )
    
    # author: Saurabh
    register(
        id='HalfCheetahEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_original:HalfCheetahEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5
        }
    )

    # author: Saurabh
    register(
        id='HalfCheetahGoalEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal:HalfCheetahGoalEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
            'goal_position': 3.0
        }
    )

    register(
        id='HalfCheetahGoalHolesEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal_holes:HalfCheetahGoalHolesEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
            'goal_position': 3.0
        }
    )

    ##############################################
    # HalfCheetahGoal environments with obstacles. 
    # author: Saurabh
    ##############################################
    obstacle_heights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    obstacle_positions = [0.0]
    for height_index in range(len(obstacle_heights)):
        for position_index in range(len(obstacle_positions)):
            height = obstacle_heights[height_index]
            position = obstacle_positions[position_index] 
            register(
                id='HalfCheetahGoalObstaclesEnv-v{}'.format(height_index),
                entry_point='multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal_obstacles:HalfCheetahGoalObstaclesEnv',
                kwargs={
                    'action_scale': 1,
                    'frame_skip': 5,
                    'obstacle_position': position,
                    'obstacle_height': height,
                    'goal_position': 3.0
                }
            )

    ####################################################
    # HalfCheetahGoal environments with adversarial force applied to joints.
    # author: Saurabh
    ####################################################
        
    for force_index in range(20):
        force = np.zeros(9)
        force[4] = -100.0 * force_index
        timestep_start = 10
        timestep_end = 15
        register(
            id='HalfCheetahGoalDisabledJoints-v{}'.format(force_index),
            entry_point='multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal_joints:HalfCheetahGoalDisabledJointsEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'goal_position': 3.0,
                'force': force,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )
    
    ####################################################
    # HalfCheetahGoal environments with motor failure at one of the actions.
    ####################################################
    for length in range(10, 110, 10):
        timestep_start = 10
        timestep_end = 10 + length
        register(
            id='HalfCheetahGoalMotorFailure-v{}'.format(length),
            entry_point='multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal_motor:HalfCheetahGoalMotorFailureEnv',
            kwargs={
                'action_scale': 1, 
                'frame_skip': 5,
                'goal_position': 3.0,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )


    ####################################################
    # HalfCheetahGoal environments with heavier body mass.
    # author: Saurabh
    ####################################################
    mass_multipliers = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]) * 50.
    for mass_multiplier_index in range(len(mass_multipliers)):
        timestep_start = 1.0
        timestep_end = 1.5
        register(
            id='HalfCheetahGoalMass-v{}'.format(mass_multiplier_index),
            entry_point='multiworld.envs.mujoco.classic_mujoco.halfcheetah_environments.half_cheetah_goal_mass:HalfCheetahGoalMassEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'goal_position': 3.0,
                'mass_multiplier': mass_multipliers[mass_multiplier_index],
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )


    register(
        id='Walker2dEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.walker_environments.walker2d_original:Walker2dEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5
        }
    )
    
    register(
        id='Walker2dVelocityEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.walker_environments.walker2d_velocity:Walker2dVelocityEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5
        }
    )

    ####################################################
    # Walker Velocity environments with negative force applied to one of the joints.
    ####################################################
    for force_index in range(1, 20):
        force = np.zeros(9)
        force[4] = -10.0 * force_index
        timestep_start = 10
        timestep_end = 15
        register(
            id='WalkerVelocityDisabledJoints-v{}'.format(force_index),
            entry_point='multiworld.envs.mujoco.classic_mujoco.walker_environments.walker2d_velocity_joints:Walker2dVelocityJointsEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'force': force,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )

    ##############################################
    # WalkerVelocity environments with obstacles. 
    ##############################################
    obstacle_heights = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575]
    obstacle_positions = [2.5]
    for height_index in range(len(obstacle_heights)):
        for position_index in range(len(obstacle_positions)):
            height = obstacle_heights[height_index]
            position = obstacle_positions[position_index]
            register(
                id='WalkerVelocityObstaclesEnv-v{}'.format(height_index),
                entry_point='multiworld.envs.mujoco.classic_mujoco.walker_environments.walker2d_velocity_obstacles:Walker2dVelocityObstaclesEnv',
                kwargs={
                    'action_scale': 1,
                    'frame_skip': 5,
                    'obstacle_position': position,
                    'obstacle_height': height
                }
            ) 
   
    ####################################################
    # WalkerVelocity environments with motor failure at one of the actions.
    ####################################################
    for length in range(10, 110, 10):  # (5, 35, 5)
        timestep_start = 10
        timestep_end = timestep_start + length
        register(
            id='WalkerVelocityMotorFailure-v{}'.format(length),
            entry_point='multiworld.envs.mujoco.classic_mujoco.walker_environments.walker2d_velocity_motor:WalkerVelocityMotorFailureEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )
    
    # author: Saurabh
    register(
       id='HopperEnv-v1',
       entry_point='multiworld.envs.mujoco.classic_mujoco.all_hopper_environments.hopper:HopperEnv',
       kwargs={
           'action_scale': 1,
           'frame_skip': 5,
       }
    )

    # author: Saurabh
    register(
       id='HopperVelocityEnv-v1',
       entry_point='multiworld.envs.mujoco.classic_mujoco.all_hopper_environments.hopper_velocity:HopperVelocityEnv',
       kwargs={
           'action_scale': 1,
           'frame_skip': 5,
       }
    )

    ##############################################
    # HopperVelocity environments with obstacles. 
    ##############################################
    obstacle_heights = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575]
    obstacle_positions = [2.5]
    for height_index in range(len(obstacle_heights)):
        for position_index in range(len(obstacle_positions)):
            height = obstacle_heights[height_index]
            position = obstacle_positions[position_index]
            register(
                id='HopperVelocityObstaclesEnv-v{}'.format(height_index),
                entry_point='multiworld.envs.mujoco.classic_mujoco.all_hopper_environments.hopper_velocity_obstacles:HopperVelocityObstaclesEnv',
                kwargs={
                    'action_scale': 1,
                    'frame_skip': 5,
                    'obstacle_position': position,
                    'obstacle_height': height
                }
            )
  

    ####################################################
    # Hopper Velocity environments with negative force applied to one of the joints.
    ####################################################
    for force_index in range(1, 20): 
        force = np.zeros(6)
        force[4] = -10.0 * force_index
        timestep_start = 10
        timestep_end = 15
        register(
            id='HopperVelocityDisabledJoints-v{}'.format(force_index),
            entry_point='multiworld.envs.mujoco.classic_mujoco.all_hopper_environments.hopper_velocity_joints:HopperVelocityDisabledJointsEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'force': force,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )

    ####################################################
    # HopperVelocity environments with motor failure at one of the actions.
    ####################################################
    for length in range(10, 110, 10):
        timestep_start = 10
        timestep_end = 10 + length
        register(
            id='HopperVelocityMotorFailure-v{}'.format(length),
            entry_point='multiworld.envs.mujoco.classic_mujoco.all_hopper_environments.hopper_velocity_motor:HopperVelocityMotorFailureEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )

    # author: Saurabh
    register(
        id='AntEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant:AntEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
        }
    )

    # author: Saurabh
    register(
        id='AntVelocityEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_velocity:AntVelocityEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
        }
    )

    ####################################################
    # Ant Velocity environments with negative force applied to one of the joints.
    ####################################################
    for joint in range(1, 20): # (3, 9) Ant has 6 joints.
        force = np.zeros(9)
        force[4] = -100.0 * joint  # force[joint]
        timestep_start = 10
        timestep_end = 15
        register(
            id='AntVelocityDisabledJoints-v{}'.format(joint),
            entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_velocity_joints:AntVelocityJointsEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'force': force,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )
    
    ##############################################
    # AntVelocity environments with obstacles. 
    ##############################################
    obstacle_heights = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575]
    obstacle_positions = [3.0]
    for height_index in range(len(obstacle_heights)):
        for position_index in range(len(obstacle_positions)):
            height = obstacle_heights[height_index]
            position = obstacle_positions[position_index]
            register(
                id='AntVelocityObstaclesEnv-v{}'.format(height_index),
                entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_velocity_obstacles:AntVelocityObstaclesEnv',
                kwargs={
                    'action_scale': 1,
                    'frame_skip': 5,
                    'obstacle_position': position,
                    'obstacle_height': height
                }
            )


    # author: Saurabh
    register(
        id='AntGoalEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_goal:AntGoalEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
            'n_tasks': 1,
            'goal_position': 5,
        }
    )

    ##############################################
    # AntGoal environments with obstacles. 
    ##############################################
    obstacle_heights = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575]
    obstacle_positions = [2.0]
    for height_index in range(len(obstacle_heights)):
        for position_index in range(len(obstacle_positions)):
            height = obstacle_heights[height_index]
            position = obstacle_positions[position_index]
            register(
                id='AntGoalObstaclesEnv-v{}'.format(height_index),
                entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_goal_obstacles:AntGoalObstaclesEnv',
                kwargs={
                    'action_scale': 1,
                    'frame_skip': 5,
                    'obstacle_position': position,
                    'obstacle_height': height,
                    'n_tasks': 1,
                    'goal_position': 5.0
                }
            )



    ####################################################
    # AntGoal environments with adversarial force applied to joints.
    ####################################################
    for multiplier in range(1, 20): # (3, 9) HalfCheetah has 6 joints.
        force = np.zeros(14)
        force[2] = 0. * -100.0 * multiplier  # force[joint]
        timestep_start = 10
        timestep_end = 11
        register(
            id='AntGoalDisabledJoints-v{}'.format(multiplier),
            entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_goal_joints:AntGoalDisabledJointsEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'goal_position': 3.0,
                'force': force,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )
    
    ####################################################
    # AntGoal environments with motor failure at one of the actions.
    ####################################################
    for action in range(1):
        timestep_start = 10
        timestep_stop = 11
        register(
            id='AntGoalMotorFailure-v{}'.format(action),
            entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_goal_motor:AntGoalMotorFailureEnv',
            kwargs={
                'action_scale': 1,
                'frame_skip': 5,
                'goal_position': 3.0,
                'action_disabled': action,
                'timestep_start': timestep_start,
                'timestep_end': timestep_end
            }
        )

    register(
        id='AntGoalTerminalEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_goal_terminal:AntGoalTerminalEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
            'n_tasks': 1,
            'goal_position': 5,
        }
    )

    register(
        id='AntGoalFallingEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_goal_falling:AntGoalFallingEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
            'n_tasks': 1,
            'goal_position': 5,
        }
    )

    register(
        id='AntGoalAliveBonusEnv-v1',
        entry_point='multiworld.envs.mujoco.classic_mujoco.all_ant_environments.ant_goal_alive_bonus:AntGoalAliveBonusEnv',
        kwargs={
            'action_scale': 1,
            'frame_skip': 5,
            'n_tasks': 1,
            'goal_position': 5,
        }
    )
        

    """
    Reaching tasks
    """

    register(
        id='SawyerReachXYEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )

    register(
        id='SawyerReachXYZEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': '7b3113b',
            'author': 'vitchyr'
        },
        kwargs={
            'hide_goal_markers': False,
            'norm_order': 2,
        },
    )

    register(
        id='SawyerReachXYZEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYZEnv',
        tags={
            'git-commit-hash': 'bea5de',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )

    register(
        id='Image48SawyerReachXYEnv-v1',
        entry_point=create_image_48_sawyer_reach_xy_env_v1,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
    )
    register(
        id='Image84SawyerReachXYEnv-v1',
        entry_point=create_image_84_sawyer_reach_xy_env_v1,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
    )


    """
    Pushing Tasks, XY
    """

    register(
        id='SawyerPushAndReachEnvEasy-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .45),
            goal_high=(0.15, 0.7, 0.02, .1, .65),
            puck_low=(-.1, .45),
            puck_high=(.1, .65),
            hand_low=(-0.15, 0.4, 0.02),
            hand_high=(0.15, .7, 0.02),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    register(
        id='SawyerPushAndReachEnvMedium-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.2, 0.35, 0.02, -.15, .4),
            goal_high=(0.2, 0.75, 0.02, .15, .7),
            puck_low=(-.15, .4),
            puck_high=(.15, .7),
            hand_low=(-0.2, 0.35, 0.05),
            hand_high=(0.2, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    register(
        id='SawyerPushAndReachEnvHard-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'ddd73dc',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .35),
            goal_high=(0.25, 0.8, 0.02, .2, .75),
            puck_low=(-.2, .35),
            puck_high=(.2, .75),
            hand_low=(-0.25, 0.3, 0.02),
            hand_high=(0.25, .8, 0.02),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=True,
        )
    )

    """
    Pushing tasks, XY, Arena
    """
    register(
        id='SawyerPushAndReachArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'dea1627',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_arena.xml',
            reward_type='state_distance',
            reset_free=False,
        )
    )

    register(
        id='SawyerPushAndReachArenaResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'dea1627',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_arena.xml',
            reward_type='state_distance',
            reset_free=True,
        )
    )

    register(
        id='SawyerPushAndReachSmallArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '7256aaf',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .5),
            goal_high=(0.15, 0.75, 0.02, .1, .7),
            puck_low=(-.3, .25),
            puck_high=(.3, .9),
            hand_low=(-0.15, 0.4, 0.05),
            hand_high=(0.15, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_small_arena.xml',
            reward_type='state_distance',
            reset_free=False,
            clamp_puck_on_step=False,
        )
    )

    register(
        id='SawyerPushAndReachSmallArenaResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '7256aaf',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.15, 0.4, 0.02, -.1, .5),
            goal_high=(0.15, 0.75, 0.02, .1, .7),
            puck_low=(-.3, .25),
            puck_high=(.3, .9),
            hand_low=(-0.15, 0.4, 0.05),
            hand_high=(0.15, .75, 0.3),
            norm_order=2,
            xml_path='sawyer_xyz/sawyer_push_puck_small_arena.xml',
            reward_type='state_distance',
            reset_free=True,
            clamp_puck_on_step=False,
        )
    )

    """
    NIPS submission pusher environment
    """
    register(
        id='SawyerPushNIPS-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYEasyEnv',
        tags={
            'git-commit-hash': 'bede25d',
            'author': 'ashvin',
        },
        kwargs=dict(
            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        )

    )

    register(
        id='SawyerPushNIPSHarder-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_nips:SawyerPushAndReachXYHarderEnv',
        tags={
            'git-commit-hash': 'b5cac93',
            'author': 'murtaza',
        },
        kwargs=dict(
            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        )

    )

    """
    Door Hook Env
    """

    register(
        id='SawyerDoorHookEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '15b48d5',
            'author': 'murtaza',
        },
        kwargs = dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=False,
        )
    )

    register(
        id='SawyerDoorHookResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '15b48d5',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=True,
        )
    )

    """
    Pick and Place
    """
    register(
        id='SawyerPickupEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
            num_goals_presampled=1000,
        )

    )


def create_image_48_sawyer_reach_xy_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v1')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_push_and_reach_arena_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )

def create_image_48_sawyer_push_and_reach_arena_env_reset_free_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachArenaResetFreeEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )

register_custom_envs()
