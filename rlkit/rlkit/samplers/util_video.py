from flask_opencv_streamer.streamer import Streamer
import numpy as np
from array2gif import write_gif
import time
import cv2
import pickle

def rollout(env, agent, max_path_length=np.inf, animated=False, skill=None, deterministic=False, 
            streamer=None):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :return:
    """
    # if animated:
    #    port = 3030
    #    require_login = False
    #    streamer = Streamer(port, require_login)

    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    path_length = 0
    o = env.reset()
    next_o = None
    episode = []

    episode_count = 0
    skill = 0

    if animated:
        episode.append(env._wrapped_env.sim.render(768, 588))
        # env.render()
        # frame = env._wrapped_env.sim.render('rgb_array', width=256, height=196)
        frame = env._wrapped_env.sim.render(768, 588)  # 256, 196
        # frame = env._wrapped_env.sim.render(256, 196)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        streamer.update_frame(frame)


    done = False
    while True: #path_length < max_path_length:
        if animated and not streamer.is_streaming:
            streamer.start_streaming()

        a, agent_info = agent.get_action(o, skill=skill, deterministic=deterministic)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        if path_length > max_path_length:
            d = True
            path_length = 0

        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if done:
            done = False
            print('Done with Episode!')
            episode_count += 1
            skill = episode_count % 5
            print(skill)        
            if episode_count == 10:
                pickle.dump( episode, open( "image_array.p", "wb" ) )
            # break
            o = env.reset()
        if d:
            done = True
        o = next_o
        if animated:
            if episode_count >= 1:
                episode.append(env._wrapped_env.sim.render(768, 588))
   
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
