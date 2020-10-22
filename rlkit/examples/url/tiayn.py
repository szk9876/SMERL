import gym
from rlkit.torch.url.path_replay_buffer import ObsDictPathReplayBuffer
from rlkit.torch.url.url import UrlTwinSac
from rlkit.torch.url.envs.disc_wrapper import DiscriminatorWrappedEnv
from rlkit.torch.url.discriminator import Discriminator
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.launchers.launcher_util import setup_logger
from multiworld.envs.pygame.point2d import Point2DEnv  # trigger env registration
from multiworld.envs.mujoco.classic_mujoco.half_cheetah import HalfCheetahEnv
import rlkit.torch.pytorch_util as ptu


def experiment(variant):
    wrapped_env = gym.make(variant['env_name'])
    obs_dim = wrapped_env.observation_space.spaces['observation'].low.size

    net_size = variant['net_size']

    disc = Discriminator(
        input_size=obs_dim,
        output_size=variant['disc_kwargs']['num_skills'],
        hidden_sizes=[net_size, net_size],
        **variant['disc_kwargs'])

    env = DiscriminatorWrappedEnv(wrapped_env=wrapped_env,
                                  disc=disc,
                                  **variant['env_kwargs']
                                  )

    context_dim = env.context_dim
    action_dim = wrapped_env.action_space.low.size

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
    )
    replay_buffer = ObsDictPathReplayBuffer(
        env=env,
        max_path_length=variant['algo_kwargs']['max_path_length'],
        observation_key='observation',
        context_key='context',
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
    # point2d
    diayn_weight = 100.
    variant = dict(
        env_name='Point2D-center-v0',
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
            save_replay_buffer=True,
            save_environment=True,
    
            use_automatic_entropy_tuning=True,
            fixed_entropy=0.1
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
        ),
        disc_kwargs=dict(
            batch_size=128,
            num_batches_per_fit=1,
            num_skills=6,
            sampling_strategy='random',
            sampling_window=10,
        ),
        env_kwargs=dict(
            reward_params=dict(type='wrapped_env + tiayn'),
            diayn_weight=diayn_weight
        ),
        net_size=32
    )

    # variant = dict(
    #     env_name='HalfCheetahEnv-v0',
    #     algo_kwargs=dict(
    #         num_epochs=10000,
    #         num_steps_per_epoch=1000,
    #         num_steps_per_eval=1000,
    #         # num_updates_per_epoch=1000,
    #         num_updates_per_env_step=1,
    #         max_path_length=1000,
    #         batch_size=128,
    #         discount=0.99,

    #         soft_target_tau=0.01,
    #         policy_lr=3e-4,
    #         qf_lr=3e-4,
    #         vf_lr=3e-4,

    #         collection_mode='online',
    #         save_replay_buffer=True,
    #         save_environment=True,

    #         use_automatic_entropy_tuning=True,
    #         fixed_entropy=0.1
    #     ),
    #     replay_buffer_kwargs=dict(
    #         max_size=10000,
    #     ),
    #     disc_kwargs=dict(
    #         batch_size=128,
    #         num_batches_per_fit=1,
    #         num_skills=50,

    #         sampling_strategy='random',
    #         sampling_window=10,
    #     ),
    #     env_kwargs=dict(
    #         reward_params=dict(type='diayn')
    #     ),
    #     net_size=300
    # )

    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    # setup_logger('url-disc-half-cheetah', variant=variant)
    setup_logger('reward+{}*tiayn'.format(diayn_weight), variant=variant)
    experiment(variant)
