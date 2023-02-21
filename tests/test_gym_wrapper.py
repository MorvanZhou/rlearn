import unittest

import gymnasium
import numpy as np
from tensorflow import keras

import rlearn


class CartPoleSmoothReward(rlearn.EnvWrapper):
    def __init__(self, render_mode=None, seed=None):
        self.env = gymnasium.make('CartPole-v1', render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self):
        s, _ = self.env.reset()
        return s

    def step(self, a):
        s_, _, done, _, _ = self.env.step(a)
        x, _, theta, _ = s_
        r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
        r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
        r = r1 + r2
        return s_, r, done

    def render(self):
        self.env.render()


class CartPoleDiscreteReward(CartPoleSmoothReward):
    def step(self, a):
        s_, _, done, _, _ = self.env.step(a)
        r = -1 if done else 0
        return s_, r, done

    def render(self):
        self.env.render()


class Pendulum(rlearn.EnvWrapper):
    def __init__(self, render_mode=None, ep_step=100, seed=None):
        self.env = gymnasium.make('Pendulum-v1', render_mode=render_mode)
        if seed is not None:
            self.env.reset(seed=seed)
        self.step_count = 0
        self.ep_step = ep_step

    def reset(self) -> np.ndarray:
        s, _ = self.env.reset()
        self.step_count = 0
        return s

    def step(self, a):
        s_, r, _, _, _ = self.env.step(a)
        r = (r + 8) / 8
        done = self.step_count >= self.ep_step
        self.step_count += 1
        return s_, r, done

    def render(self):
        self.env.render()


class GameWrapperTest(unittest.TestCase):
    def test_cartpole_dqn(self):
        trainer = rlearn.trainer.DQNTrainer()
        trainer.set_model_encoder(
            q=keras.Sequential([
                keras.layers.InputLayer(4),
                keras.layers.Dense(16),
            ]),
            action_num=2
        )
        trainer.set_replay_buffer(100)
        trainer.set_params(learning_rate=0.01)

        game = CartPoleSmoothReward()
        for _ in range(2):
            s = game.reset()
            self.assertIsInstance(s, np.ndarray)
            for _ in range(10):
                a = trainer.predict(s)
                s, r, done = game.step(a)
                self.assertIsInstance(s, np.ndarray)
                self.assertIsInstance(r, float)
                self.assertIsInstance(done, bool)
                if done:
                    break

    # def test_keyboard(self):
    #     from pynput.keyboard import Key, Listener
    #     from queue import Queue
    #
    #     pressed = Queue()
    #
    #     def on_release(key):
    #         print(key)
    #         if key == Key.left:
    #             pressed.put(0)
    #         elif key == Key.right:
    #             pressed.put(1)
    #
    #     # Collect events until released
    #     listener = Listener(on_release=on_release)
    #     listener.start()
    #     data = []
    #     try:
    #         listener.wait()
    #         game = CartPoleSmoothReward(render_mode="human")
    #         for _ in range(10):
    #             s = game.reset()
    #             for _ in range(200):
    #                 a = pressed.get()
    #                 s_, r, done = game.step(a)
    #                 data.append([s, a, r, s_, done])
    #                 if done:
    #                     break
    #                 s = s_
    #     finally:
    #         listener.stop()
