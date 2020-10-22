from flask_opencv_streamer.streamer import Streamer
import cv2
import numpy as np
import gym
import time

port = 3030
require_login = False
streamer = Streamer(port, require_login)

env = gym.make("HalfCheetah-v2")
env.reset()

while True:
    if not streamer.is_streaming:
        streamer.start_streaming()

    env.step(env.action_space.sample())
    frame = env.render('rgb_array', width=256, height=196)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    streamer.update_frame(frame)
